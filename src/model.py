"""
Heterogeneous Graph Neural Network Models

This module implements GNN architectures for lab value imputation:
- Relational Graph Convolutional Network (R-GCN)
- Heterogeneous Graph Transformer (HGT)
- Edge prediction heads for lab value regression

Key Design Decisions:
1. Separate embeddings for each node type (patient, lab, diagnosis, medication)
2. Type-specific message passing functions
3. Edge attributes for patient-lab edges (continuous lab values)
4. MLP-based edge regression head
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

try:
    from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
    from torch_geometric.nn import to_hetero, HGTConv
except ImportError:
    logging.warning("PyTorch Geometric not installed. Models will not work.")


# ============================================================================
# Heterogeneous R-GCN Model
# ============================================================================

class HeteroRGCN(nn.Module):
    """
    Relational Graph Convolutional Network for heterogeneous graphs.

    Architecture:
    1. Node-type-specific embeddings (for non-patient nodes)
    2. Multiple HeteroConv layers for message passing
    3. Edge regression head for lab value prediction

    Rationale:
        R-GCN extends GCN to handle multiple edge types (relations).
        Each relation has its own weight matrix, allowing the model to
        learn different message passing patterns for:
        - patient ↔ lab (continuous values)
        - patient ↔ diagnosis (disease indicators)
        - patient ↔ medication (treatment patterns)
    """

    def __init__(
        self,
        metadata: Tuple,  # (node_types, edge_types)
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        patient_feature_dim: int = None,  # Not used in Iteration 6 (learnable embeddings)
        use_batch_norm: bool = True,
        activation: str = "relu"
    ):
        """
        Args:
            metadata: Graph metadata (node_types, edge_types) from HeteroData
            hidden_dim: Hidden dimension for all node embeddings
            num_layers: Number of GNN layers
            dropout: Dropout probability
            patient_feature_dim: Dimension of patient input features
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ("relu", "elu", "leaky_relu")
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        node_types, edge_types = metadata

        # ====================================================================
        # Node Embeddings
        # ====================================================================

        # All node types: learnable embeddings
        # We'll get the number of nodes for each type at runtime
        self.embeddings = nn.ModuleDict()

        # Initialize embedding tables (will be set during first forward pass)
        self.embedding_dims = {}  # Store expected dimensions

        # Patient nodes: deeper MLP to transform learnable embeddings
        # This adds capacity to learn complex patient representations
        self.patient_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # L2 normalization to make similar patients cluster together
        self.patient_l2_norm = nn.functional.normalize

        logging.info(f"Initialized HeteroRGCN with hidden_dim={hidden_dim}, num_layers={num_layers}")

        # ====================================================================
        # GNN Layers
        # ====================================================================

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        for layer_idx in range(num_layers):
            # HeteroConv: applies separate GCNConv for each edge type
            conv_dict = {}

            for edge_type in edge_types:
                src_type, rel_type, dst_type = edge_type

                # Use SAGEConv for better handling of varying node degrees
                # SAGEConv concatenates neighbor aggregation with self-features
                conv_dict[edge_type] = SAGEConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggr='mean'  # Mean aggregation is scale-invariant
                )

            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

            # Batch normalization per node type
            if use_batch_norm:
                bn_dict = nn.ModuleDict({
                    node_type: nn.BatchNorm1d(hidden_dim)
                    for node_type in node_types
                })
                self.batch_norms.append(bn_dict)

        # ====================================================================
        # Activation Function
        # ====================================================================

        if activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # ====================================================================
        # Edge Regression Head
        # ====================================================================

        # Predict lab value from concatenated patient and lab embeddings
        self.edge_predictor = EdgeRegressionHead(
            input_dim=2 * hidden_dim,  # [h_patient; h_lab]
            hidden_dims=[64, 32],
            output_dim=1,
            dropout=dropout
        )

        # ====================================================================
        # Degree-Aware Hybrid: Tabular MLP for low-connectivity patients
        # ====================================================================

        # Fallback MLP for patients with < 6 labs (sparse graph context)
        # Uses only patient and lab embeddings without GNN propagation
        self.tabular_mlp = EdgeRegressionHead(
            input_dim=2 * hidden_dim,  # [h_patient; h_lab] (before GNN)
            hidden_dims=[64, 32],
            output_dim=1,
            dropout=dropout
        )
        self.degree_threshold = 6  # Hard gate: < 6 labs → tabular; >= 6 → GNN

    def _init_embeddings(self, data):
        """
        Initialize embedding tables based on actual node counts in data.

        Args:
            data: HeteroData object

        Rationale:
            We don't know the number of nodes for each type until we see the data.
            This method creates embedding tables on first forward pass.

        Iteration 6: All node types (including patients) use learnable embeddings.
        """
        for node_type in data.node_types:
            if node_type not in self.embeddings:
                num_nodes = data[node_type].num_nodes

                # Create embedding table
                embedding = nn.Embedding(num_nodes, self.hidden_dim)
                nn.init.xavier_uniform_(embedding.weight)

                self.embeddings[node_type] = embedding
                self.embedding_dims[node_type] = num_nodes

                logging.info(f"Created embedding for {node_type}: {num_nodes} nodes")

    def encode_nodes(self, data):
        """
        Encode all node types to hidden_dim embeddings.

        Args:
            data: HeteroData object

        Returns:
            Dictionary mapping node_type to embeddings
        """
        x_dict = {}

        # Determine device from model parameters
        device = next(self.parameters()).device

        # All nodes: lookup learnable embeddings
        for node_type in data.node_types:
            num_nodes = data[node_type].num_nodes
            # Create node indices [0, 1, 2, ..., num_nodes-1]
            node_indices = torch.arange(num_nodes, device=device)
            x_dict[node_type] = self.embeddings[node_type](node_indices)

        # Patient nodes: apply deeper transformation + L2 normalization
        if 'patient' in x_dict:
            patient_embeddings = self.patient_transform(x_dict['patient'])
            # L2 normalize so similar patients cluster together
            x_dict['patient'] = self.patient_l2_norm(patient_embeddings, p=2, dim=1)

        return x_dict

    def forward(self, data):
        """
        Forward pass through the model.

        Args:
            data: HeteroData object

        Returns:
            Dictionary of node embeddings for each node type
        """
        # Initialize embeddings if first forward pass
        if len(self.embeddings) == 0:
            self._init_embeddings(data)

        # Encode all nodes to hidden_dim
        x_dict = self.encode_nodes(data)

        # Message passing layers
        for layer_idx in range(self.num_layers):
            # Apply convolution
            x_dict = self.convs[layer_idx](x_dict, data.edge_index_dict)

            # Apply batch normalization
            if self.use_batch_norm:
                for node_type in x_dict.keys():
                    x_dict[node_type] = self.batch_norms[layer_idx][node_type](x_dict[node_type])

            # Apply activation
            x_dict = {key: self.activation(x) for key, x in x_dict.items()}

            # Apply dropout
            if layer_idx < self.num_layers - 1:  # Don't apply dropout after last layer
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training)
                         for key, x in x_dict.items()}

        return x_dict

    def predict_lab_values(self, data, patient_indices, lab_indices):
        """
        Predict lab values for given patient-lab pairs.

        Iteration 7: Degree-aware hybrid approach.
        - Low-connectivity patients (< 6 labs): Use tabular MLP on initial embeddings
        - High-connectivity patients (>= 6 labs): Use GNN head on propagated embeddings

        Args:
            data: HeteroData object (for edge_index_dict)
            patient_indices: Tensor of patient node indices
            lab_indices: Tensor of lab node indices

        Returns:
            Predicted lab values (continuous, normalized)
        """
        # Step 0: Initialize embeddings if needed
        if len(self.embeddings) == 0:
            self._init_embeddings(data)

        # Step 1: Get initial embeddings (before GNN propagation)
        initial_x_dict = self.encode_nodes(data)

        # Step 2: Compute patient degrees (number of lab connections)
        patient_lab_edges = data['patient', 'has_lab', 'lab'].edge_index
        patient_degrees = torch.bincount(patient_lab_edges[0], minlength=data['patient'].num_nodes)

        # Step 3: Get final embeddings (after GNN propagation)
        x_dict = self.forward(data)

        # Step 4: Degree-aware gating
        # Extract embeddings for specific patient-lab pairs
        initial_patient_embeds = initial_x_dict['patient'][patient_indices]
        initial_lab_embeds = initial_x_dict['lab'][lab_indices]

        final_patient_embeds = x_dict['patient'][patient_indices]
        final_lab_embeds = x_dict['lab'][lab_indices]

        # Get degrees for these specific patients
        edge_degrees = patient_degrees[patient_indices]

        # Predict using tabular MLP (for low-connectivity) or GNN head (for high-connectivity)
        low_degree_mask = edge_degrees < self.degree_threshold

        predictions = torch.zeros(len(patient_indices), device=patient_indices.device)

        if low_degree_mask.any():
            # Use tabular MLP on initial embeddings (no GNN propagation)
            initial_embeds = torch.cat([
                initial_patient_embeds[low_degree_mask],
                initial_lab_embeds[low_degree_mask]
            ], dim=1)
            predictions[low_degree_mask] = self.tabular_mlp(initial_embeds).squeeze(-1)

        if (~low_degree_mask).any():
            # Use GNN head on propagated embeddings
            final_embeds = torch.cat([
                final_patient_embeds[~low_degree_mask],
                final_lab_embeds[~low_degree_mask]
            ], dim=1)
            predictions[~low_degree_mask] = self.edge_predictor(final_embeds).squeeze(-1)

        return predictions


# ============================================================================
# Edge Regression Head
# ============================================================================

class EdgeRegressionHead(nn.Module):
    """
    MLP-based edge regression head for predicting continuous lab values.

    Architecture:
        Input: Concatenated patient and lab embeddings [h_patient; h_lab]
        Hidden layers: MLP with specified dimensions
        Output: Single continuous value (predicted lab value)

    Rationale:
        Lab value prediction is a regression task. We use an MLP to map
        the combined patient-lab representation to a scalar value.
        Multiple hidden layers allow non-linear transformations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        """
        Args:
            input_dim: Input dimension (2 * hidden_dim from node embeddings)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (1 for regression)
            dropout: Dropout probability
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, edge_embeds):
        """
        Args:
            edge_embeds: [num_edges, input_dim]

        Returns:
            predictions: [num_edges, output_dim]
        """
        return self.mlp(edge_embeds)


# ============================================================================
# Alternative: Heterogeneous Graph Transformer (HGT)
# ============================================================================

class HeteroGT(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) model.

    HGT uses attention mechanisms to weigh the importance of different
    neighbors based on their types and relations.

    Advantages over R-GCN:
    - Attention weights provide interpretability
    - Better handling of heterogeneous graphs with varying edge importance
    - Can capture long-range dependencies

    Trade-offs:
    - More parameters (higher memory, slower training)
    - May overfit on small datasets
    """

    def __init__(
        self,
        metadata: Tuple,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        patient_feature_dim: int = 3
    ):
        """
        Args:
            metadata: Graph metadata (node_types, edge_types)
            hidden_dim: Hidden dimension
            num_layers: Number of HGT layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            patient_feature_dim: Patient input feature dimension
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        node_types, edge_types = metadata

        # Node encoders
        self.patient_encoder = nn.Linear(patient_feature_dim, hidden_dim)
        self.embeddings = nn.ModuleDict()

        # HGT layers
        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads,
                dropout=dropout
            )
            self.convs.append(conv)

        # Edge predictor
        self.edge_predictor = EdgeRegressionHead(
            input_dim=2 * hidden_dim,
            hidden_dims=[64, 32],
            output_dim=1,
            dropout=dropout
        )

    def _init_embeddings(self, data):
        """Initialize embeddings for non-patient nodes."""
        for node_type in data.node_types:
            if node_type != 'patient' and node_type not in self.embeddings:
                num_nodes = data[node_type].num_nodes
                embedding = nn.Embedding(num_nodes, self.hidden_dim)
                nn.init.xavier_uniform_(embedding.weight)
                self.embeddings[node_type] = embedding

    def encode_nodes(self, data):
        """Encode all nodes."""
        x_dict = {}

        patient_features = data['patient'].x
        x_dict['patient'] = self.patient_encoder(patient_features)

        for node_type in data.node_types:
            if node_type != 'patient':
                num_nodes = data[node_type].num_nodes
                node_indices = torch.arange(num_nodes, device=patient_features.device)
                x_dict[node_type] = self.embeddings[node_type](node_indices)

        return x_dict

    def forward(self, data):
        """Forward pass."""
        if len(self.embeddings) == 0:
            self._init_embeddings(data)

        x_dict = self.encode_nodes(data)

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)

        return x_dict

    def predict_lab_values(self, data, patient_indices, lab_indices):
        """Predict lab values."""
        x_dict = self.forward(data)

        patient_embeds = x_dict['patient'][patient_indices]
        lab_embeds = x_dict['lab'][lab_indices]

        edge_embeds = torch.cat([patient_embeds, lab_embeds], dim=1)
        predictions = self.edge_predictor(edge_embeds)

        return predictions.squeeze(-1)


# ============================================================================
# Model Factory
# ============================================================================

def build_model(config: Dict, metadata: Tuple, patient_feature_dim: int):
    """
    Build GNN model based on configuration.

    Args:
        config: Configuration dictionary
        metadata: Graph metadata (node_types, edge_types)
        patient_feature_dim: Dimension of patient features

    Returns:
        GNN model instance

    Rationale:
        Factory pattern allows easy switching between model architectures
        by changing config['model']['architecture'].
    """
    model_config = config['model']
    architecture = model_config['architecture']

    common_args = {
        'metadata': metadata,
        'hidden_dim': model_config['hidden_dim'],
        'num_layers': model_config['num_layers'],
        'dropout': model_config['dropout'],
        'patient_feature_dim': patient_feature_dim
    }

    if architecture == "RGCN":
        model = HeteroRGCN(
            **common_args,
            use_batch_norm=model_config['use_batch_norm'],
            activation=model_config['activation']
        )
        logging.info("Built HeteroRGCN model")

    elif architecture == "HGT":
        model = HeteroGT(
            **common_args,
            num_heads=model_config.get('num_heads', 4)
        )
        logging.info("Built HeteroGT model")

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {num_params:,} trainable parameters")

    return model


# ============================================================================
# Loss Functions
# ============================================================================

def compute_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "mae"
) -> torch.Tensor:
    """
    Compute regression loss for lab value prediction.

    Args:
        predictions: Predicted lab values [num_edges]
        targets: True lab values [num_edges]
        loss_type: "mae" (Mean Absolute Error) or "mse" (Mean Squared Error)

    Returns:
        Scalar loss

    Rationale:
        MAE vs MSE trade-off:
        - MAE: Robust to outliers, treats all errors equally
        - MSE: Penalizes large errors more, smooth gradients

        For medical data with potential outliers, MAE is often preferred.
    """
    if loss_type == "mae":
        loss = F.l1_loss(predictions, targets)
    elif loss_type == "mse":
        loss = F.mse_loss(predictions, targets)
    elif loss_type == "huber":
        # Huber loss: MSE for small errors, MAE for large errors
        loss = F.huber_loss(predictions, targets)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    """
    Quick test of model architecture.
    """
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*70)
    print("Testing Model Architecture")
    print("="*70)

    # Create dummy metadata
    node_types = ['patient', 'lab', 'diagnosis', 'medication']
    edge_types = [
        ('patient', 'has_lab', 'lab'),
        ('lab', 'has_lab_rev', 'patient'),
        ('patient', 'has_diagnosis', 'diagnosis'),
        ('diagnosis', 'has_diagnosis_rev', 'patient'),
        ('patient', 'has_medication', 'medication'),
        ('medication', 'has_medication_rev', 'patient')
    ]
    metadata = (node_types, edge_types)

    # Create model
    model = HeteroRGCN(
        metadata=metadata,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        patient_feature_dim=3
    )

    print(f"\n✓ Model created")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test edge predictor
    dummy_embeds = torch.randn(10, 128)
    predictions = model.edge_predictor(dummy_embeds)
    print(f"✓ Edge predictor output shape: {predictions.shape}")

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)

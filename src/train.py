"""
Training Module with Mask-and-Recover Strategy

This module implements the training loop for GNN-based lab imputation:
1. Mask-and-recover: Hide some observed lab values, train model to predict them
2. Train/validation/test splits on edges
3. Early stopping and learning rate scheduling
4. Model checkpointing
5. Experiment logging

Key Rationale:
    The mask-and-recover strategy simulates missing data. By randomly masking
    20% of observed patient-lab edges, we train the model to impute values
    based on graph structure and remaining observations. This tests the model's
    ability to leverage relationships between patients, diagnoses, and medications.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import HeteroData

from model import build_model, compute_regression_loss
from utils import get_device, format_time, count_parameters


# ============================================================================
# Edge Masking for Training
# ============================================================================

class EdgeMasker:
    """
    Handles masking of patient-lab edges for mask-and-recover training.

    Strategy:
        1. Split edges into train/val/test sets
        2. During training, mask specified fraction of train edges
        3. Model must predict masked edge values using graph context
        4. Evaluate on val/test edges that were never seen during training

    Rationale:
        This simulates real missing data scenarios:
        - Training set: Model learns from observed relationships
        - Masked edges: Model practices imputation
        - Test set: Evaluates generalization to truly unseen patient-lab pairs
    """

    def __init__(
        self,
        data: HeteroData,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        mask_fraction: float = 0.2,
        seed: int = 42,
        cv_target_labs: Optional[list] = None
    ):
        """
        Args:
            data: HeteroData object
            train_split: Fraction for training
            val_split: Fraction for validation
            test_split: Fraction for testing
            mask_fraction: Fraction of training edges to mask
            seed: Random seed for reproducibility
            cv_target_labs: Optional list of CV lab names to predict. If provided,
                only edges to these labs will be masked/predicted. All other labs
                remain as context.
        """
        self.data = data
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.mask_fraction = mask_fraction
        self.seed = seed
        self.cv_target_labs = cv_target_labs

        # Validate splits
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Splits must sum to 1.0"

        # Get patient-lab edges
        self.edge_type = ('patient', 'has_lab', 'lab')
        self.edge_index = data[self.edge_type].edge_index
        self.edge_attr = data[self.edge_type].edge_attr

        self.num_edges = self.edge_index.shape[1]

        # Filter to CV target edges if specified
        self.cv_edge_mask = self._create_cv_edge_mask()

        if self.cv_edge_mask is not None:
            logging.info(f"CV target mode enabled:")
            logging.info(f"  Total edges: {self.num_edges}")
            logging.info(f"  CV target edges: {self.cv_edge_mask.sum()} ({100*self.cv_edge_mask.sum()/self.num_edges:.1f}%)")
            logging.info(f"  Context edges (not masked): {(~self.cv_edge_mask).sum()}")
        else:
            logging.info(f"Standard mode: All {self.num_edges} edges used for prediction")

        # Create splits
        self.train_mask, self.val_mask, self.test_mask = self._create_splits()

        logging.info(f"Edge splits created:")
        logging.info(f"  Train: {self.train_mask.sum()} edges ({100*train_split:.1f}%)")
        logging.info(f"  Val: {self.val_mask.sum()} edges ({100*val_split:.1f}%)")
        logging.info(f"  Test: {self.test_mask.sum()} edges ({100*test_split:.1f}%)")

    def _create_cv_edge_mask(self) -> Optional[torch.Tensor]:
        """
        Create mask identifying edges that connect to CV target labs.

        Returns:
            Boolean mask of shape [num_edges] where True indicates edge to CV lab,
            or None if cv_target_labs not specified
        """
        if self.cv_target_labs is None or len(self.cv_target_labs) == 0:
            return None

        # Get lab metadata to map node indices to lab names
        if not hasattr(self.data['lab'], 'metadata'):
            logging.warning("Lab metadata not found in graph. CV filtering disabled.")
            return None

        lab_metadata = self.data['lab'].metadata

        # Create set of CV target lab names for fast lookup
        cv_lab_names_set = set(self.cv_target_labs)

        # Find which lab node indices correspond to CV labs
        cv_lab_indices = set()
        matched_labs = []

        for lab_idx, metadata in lab_metadata.items():
            lab_name = metadata.get('label', '')
            if lab_name in cv_lab_names_set:
                cv_lab_indices.add(lab_idx)
                matched_labs.append(lab_name)

        logging.info(f"Matched {len(matched_labs)}/{len(cv_lab_names_set)} CV target labs:")
        for lab_name in sorted(matched_labs):
            logging.info(f"    - {lab_name}")

        if len(cv_lab_indices) == 0:
            logging.warning("No CV target labs found in graph. Using all labs.")
            return None

        # Create edge mask: True if edge connects to a CV lab
        # edge_index[1] contains the lab node indices
        lab_node_indices = self.edge_index[1]  # [num_edges]
        cv_edge_mask = torch.tensor(
            [lab_idx.item() in cv_lab_indices for lab_idx in lab_node_indices],
            dtype=torch.bool
        )

        return cv_edge_mask

    def _create_splits(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create train/val/test splits on edges.

        Returns:
            Tuple of boolean masks (train_mask, val_mask, test_mask)

        Rationale:
            We split at the edge level (not patient level) to ensure
            each set contains diverse patient-lab combinations.

            If CV target mode is enabled, only CV target edges are split.
            All other edges remain as context (not in any split).
        """
        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Determine which edges to split
        if self.cv_edge_mask is not None:
            # Only split CV target edges
            edges_to_split = torch.where(self.cv_edge_mask)[0]
            num_edges_to_split = len(edges_to_split)
        else:
            # Split all edges
            edges_to_split = torch.arange(self.num_edges)
            num_edges_to_split = self.num_edges

        # Random permutation of edges to split
        perm_indices = torch.randperm(num_edges_to_split)
        perm = edges_to_split[perm_indices]

        # Compute split boundaries
        n_train = int(self.train_split * num_edges_to_split)
        n_val = int(self.val_split * num_edges_to_split)

        # Create masks (all False initially)
        train_mask = torch.zeros(self.num_edges, dtype=torch.bool)
        val_mask = torch.zeros(self.num_edges, dtype=torch.bool)
        test_mask = torch.zeros(self.num_edges, dtype=torch.bool)

        # Assign splits only to edges being split
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train + n_val]] = True
        test_mask[perm[n_train + n_val:]] = True

        return train_mask, val_mask, test_mask

    def get_masked_data(self, split: str = 'train') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get edges and targets for a specific split.

        Args:
            split: 'train', 'val', or 'test'

        Returns:
            Tuple of (edge_indices, edge_values, mask, supervision_mask)
            - edge_indices: [2, num_edges] patient and lab node indices
            - edge_values: [num_edges] normalized lab values
            - mask: Boolean mask for this split
            - supervision_mask: Which edges to mask during training (train only)

        Rationale:
            During training, we create a temporary mask on training edges.
            During validation/testing, we use all edges but only evaluate
            on val/test edges.
        """
        if split == 'train':
            mask = self.train_mask

            # Randomly mask a fraction of training edges
            if self.mask_fraction > 0:
                # Create supervision mask (which edges to predict)
                torch.manual_seed(int(time.time()))  # Different mask each epoch
                supervision_mask = torch.rand(mask.sum()) < self.mask_fraction
            else:
                supervision_mask = torch.ones(mask.sum(), dtype=torch.bool)

        elif split == 'val':
            mask = self.val_mask
            supervision_mask = torch.ones(mask.sum(), dtype=torch.bool)

        elif split == 'test':
            mask = self.test_mask
            supervision_mask = torch.ones(mask.sum(), dtype=torch.bool)

        else:
            raise ValueError(f"Unknown split: {split}")

        # Extract edges and values for this split
        edge_indices = self.edge_index[:, mask]
        edge_values = self.edge_attr[mask].squeeze(-1)

        return edge_indices, edge_values, mask, supervision_mask


# ============================================================================
# Training Loop
# ============================================================================

class Trainer:
    """
    Handles model training with mask-and-recover strategy.

    Features:
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Training metrics logging
    """

    def __init__(
        self,
        model: nn.Module,
        data: HeteroData,
        masker: EdgeMasker,
        config: Dict,
        device: torch.device
    ):
        """
        Args:
            model: GNN model
            data: HeteroData graph
            masker: EdgeMasker for train/val/test splits
            config: Configuration dictionary
            device: torch device
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.masker = masker
        self.config = config
        self.device = device

        train_config = config['train']

        # Optimizer
        self.optimizer = self._build_optimizer(train_config['optimizer'])

        # Learning rate scheduler
        self.scheduler = self._build_scheduler(train_config.get('lr_scheduler', {}))

        # Loss function
        self.loss_fn = train_config['loss']

        # Training parameters
        self.epochs = train_config['epochs']
        self.early_stopping_patience = train_config['early_stopping_patience']

        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        # ====================================================================
        # Iteration 7: Lab-wise loss reweighting
        # ====================================================================
        # Compute variance per lab type on training set
        # Samples with low-variance labs get higher weight (prevent domination)
        self.lab_weights = self._compute_lab_weights()

        logging.info("Trainer initialized")
        logging.info(f"  Optimizer: {self.optimizer.__class__.__name__}")
        logging.info(f"  Loss function: {self.loss_fn}")
        logging.info(f"  Epochs: {self.epochs}")
        logging.info(f"  Early stopping patience: {self.early_stopping_patience}")
        logging.info(f"  Lab-wise reweighting: Enabled (variance-based)")

    def _build_optimizer(self, optimizer_config: Dict) -> optim.Optimizer:
        """Build optimizer from config."""
        optimizer_type = optimizer_config.get('type', 'adam').lower()

        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config['weight_decay']
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config['weight_decay'],
                momentum=optimizer_config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def _build_scheduler(self, scheduler_config: Dict) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler from config."""
        if not scheduler_config.get('enabled', False):
            return None

        scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')

        if scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                verbose=True
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

    def _compute_lab_weights(self) -> torch.Tensor:
        """
        Compute per-lab-type weights based on variance on training set.

        Iteration 7: Lab-wise loss reweighting to prevent high-variance labs
        from dominating the loss. Weight = 1 / Var(lab).

        Returns:
            Tensor of shape [num_lab_types] with inverse variance weights
        """
        # Get training edges and values
        edge_indices, edge_values, _, _ = self.masker.get_masked_data('train')
        lab_indices = edge_indices[1]  # Lab node indices

        # Compute variance per lab type
        num_labs = self.data['lab'].num_nodes
        lab_variances = torch.zeros(num_labs, device=self.device)

        for lab_idx in range(num_labs):
            lab_mask = (lab_indices == lab_idx)
            if lab_mask.sum() > 1:  # Need at least 2 samples for variance
                lab_values = edge_values[lab_mask]
                lab_variances[lab_idx] = lab_values.var()
            else:
                lab_variances[lab_idx] = 1.0  # Default variance

        # Compute weights: 1 / variance (with epsilon for numerical stability)
        epsilon = 1e-6
        lab_weights = 1.0 / (lab_variances + epsilon)

        # Normalize weights so they sum to num_labs (keeps loss scale similar)
        lab_weights = lab_weights * num_labs / lab_weights.sum()

        logging.info(f"  Computed lab-wise weights: min={lab_weights.min():.3f}, max={lab_weights.max():.3f}, mean={lab_weights.mean():.3f}")

        return lab_weights

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss

        Rationale:
            Each epoch:
            1. Get training edges with random masking
            2. Forward pass through GNN
            3. Predict masked lab values
            4. Compute loss only on masked edges
            5. Backpropagate and update weights
        """
        self.model.train()

        # Get training edges
        edge_indices, edge_values, _, supervision_mask = self.masker.get_masked_data('train')

        patient_indices = edge_indices[0]
        lab_indices = edge_indices[1]

        # Forward pass
        self.optimizer.zero_grad()

        predictions = self.model.predict_lab_values(
            self.data,
            patient_indices,
            lab_indices
        )

        # Compute loss only on supervision edges (masked edges)
        # Iteration 7: Apply lab-wise weights to balance loss contributions
        supervised_preds = predictions[supervision_mask]
        supervised_targets = edge_values[supervision_mask]
        supervised_lab_indices = lab_indices[supervision_mask]

        # Get weights for these specific labs
        sample_weights = self.lab_weights[supervised_lab_indices]

        # Compute weighted loss
        if self.loss_fn == 'mae':
            per_sample_loss = torch.abs(supervised_preds - supervised_targets)
        elif self.loss_fn == 'mse':
            per_sample_loss = (supervised_preds - supervised_targets) ** 2
        else:
            # Fallback to unweighted for other loss types
            loss = compute_regression_loss(supervised_preds, supervised_targets, loss_type=self.loss_fn)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        # Apply weights and take mean
        loss = (per_sample_loss * sample_weights).mean()

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def validate(self, split: str = 'val') -> float:
        """
        Validate on validation or test set.

        Args:
            split: 'val' or 'test'

        Returns:
            Average loss

        Rationale:
            Validation uses all graph edges for message passing,
            but only computes loss on val/test edges that were held out.
        """
        self.model.eval()

        # Get validation/test edges
        edge_indices, edge_values, _, _ = self.masker.get_masked_data(split)

        patient_indices = edge_indices[0]
        lab_indices = edge_indices[1]

        # Forward pass
        predictions = self.model.predict_lab_values(
            self.data,
            patient_indices,
            lab_indices
        )

        # Compute loss on all val/test edges
        loss = compute_regression_loss(
            predictions,
            edge_values,
            loss_type=self.loss_fn
        )

        return loss.item()

    def train(self, output_dir: Path) -> Dict:
        """
        Full training loop with early stopping.

        Args:
            output_dir: Directory to save checkpoints and logs

        Returns:
            Training history dictionary

        Rationale:
            Early stopping prevents overfitting by monitoring validation loss.
            If validation loss doesn't improve for N epochs, stop training.
        """
        logging.info("="*70)
        logging.info("Starting Training")
        logging.info("="*70)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate('val')

            # Record
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            epoch_time = time.time() - epoch_start

            # Log
            if epoch % self.config['logging']['log_interval'] == 0:
                logging.info(
                    f"Epoch {epoch}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Time: {epoch_time:.2f}s"
                )

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save best model
                best_model_path = output_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, best_model_path)
                logging.info(f"  → New best model saved (val_loss: {val_loss:.4f})")

            else:
                self.patience_counter += 1

                if self.patience_counter >= self.early_stopping_patience:
                    logging.info(f"\nEarly stopping triggered after {epoch} epochs")
                    break

            # Periodic checkpoint
            if self.config['logging'].get('save_checkpoints', False):
                checkpoint_interval = self.config['logging'].get('checkpoint_interval', 10)
                if epoch % checkpoint_interval == 0:
                    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, checkpoint_path)

        total_time = time.time() - start_time

        logging.info("\n" + "="*70)
        logging.info("Training Complete!")
        logging.info("="*70)
        logging.info(f"Total time: {format_time(total_time)}")
        logging.info(f"Best validation loss: {self.best_val_loss:.4f}")

        # Save training history
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logging.info(f"Training history saved to {history_path}")

        return history

    def load_best_model(self, output_dir: Path):
        """
        Load best model checkpoint.

        Args:
            output_dir: Directory containing best_model.pt
        """
        best_model_path = output_dir / "best_model.pt"

        if not best_model_path.exists():
            logging.warning(f"Best model not found at {best_model_path}")
            return

        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded best model from epoch {checkpoint['epoch']} (val_loss: {checkpoint['val_loss']:.4f})")


# ============================================================================
# Main Training Pipeline
# ============================================================================

def train_pipeline(config: Dict, graph_path: Path, output_dir: Path):
    """
    Complete training pipeline.

    Args:
        config: Configuration dictionary
        graph_path: Path to saved graph.pt
        output_dir: Where to save outputs

    Rationale:
        This is the main entry point for training. It:
        1. Loads the preprocessed graph
        2. Creates edge masks for train/val/test
        3. Builds the model
        4. Trains with early stopping
        5. Evaluates on test set
        6. Saves results
    """
    from utils import set_random_seeds, setup_logging

    # Setup
    setup_logging(
        level=config['logging']['level'],
        log_file=output_dir / "training.log" if config['logging']['save_to_file'] else None
    )

    set_random_seeds(config['train']['seed'])

    # Device
    device = get_device(config['train']['device'])

    # Load graph
    logging.info(f"Loading graph from {graph_path}...")
    graph = torch.load(graph_path, map_location=device)
    logging.info(f"Graph loaded successfully")

    # Get CV target labs if enabled
    cv_target_labs = None
    if config['feature_space']['labs'].get('cv_target_mode', False):
        cv_target_labs = config['feature_space']['labs'].get('cv_target_labs', [])
        if cv_target_labs:
            logging.info(f"CV target mode enabled with {len(cv_target_labs)} target labs")
        else:
            logging.warning("CV target mode enabled but no labs specified. Using all labs.")

    # Create edge masker
    masker = EdgeMasker(
        graph,
        train_split=config['train']['train_split'],
        val_split=config['train']['val_split'],
        test_split=config['train']['test_split'],
        mask_fraction=config['train']['mask_fraction'],
        seed=config['train']['seed'],
        cv_target_labs=cv_target_labs
    )

    # Build model
    metadata = (graph.node_types, graph.edge_types)
    # Iteration 6: Using learnable embeddings for all nodes (no patient features)
    patient_feature_dim = None

    model = build_model(config, metadata, patient_feature_dim)

    # Create trainer
    trainer = Trainer(model, graph, masker, config, device)

    # Train
    history = trainer.train(output_dir)

    # Load best model and evaluate on test set
    trainer.load_best_model(output_dir)
    test_loss = trainer.validate('test')

    logging.info(f"\nFinal Test Loss: {test_loss:.4f}")

    # Save test results
    test_results = {
        'test_loss': test_loss,
        'best_val_loss': trainer.best_val_loss,
        'num_epochs': len(history['train_loss'])
    }

    results_path = output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    logging.info(f"Test results saved to {results_path}")


# ============================================================================
# Command-line Interface
# ============================================================================

if __name__ == "__main__":
    """
    Train model from command line.

    Usage:
        python train.py
    """
    import sys
    sys.path.append(str(Path(__file__).parent))

    from utils import load_config
    from config_helper import load_and_process_config

    # Load configuration with experiment mode processing
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    config = load_and_process_config(str(config_path))

    # Paths
    graph_path = Path(config['data']['output_dir']) / "graph.pt"
    output_dir = Path(config['data']['output_dir'])

    if not graph_path.exists():
        logging.error(f"Graph file not found: {graph_path}")
        logging.error("Please run graph_build.py first")
        sys.exit(1)

    # Train
    train_pipeline(config, graph_path, output_dir)

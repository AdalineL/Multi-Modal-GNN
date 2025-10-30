"""
Heterogeneous Graph Construction Module

This module builds a multi-modal graph from preprocessed EHR data:
- Node types: Patient, Lab, Diagnosis, Medication
- Edge types: patient-lab (with values), patient-diagnosis, patient-medication
- Uses PyTorch Geometric's HeteroData structure

The graph structure enables GNNs to learn from:
1. Direct relationships (patient has lab result)
2. Indirect relationships (patients with similar diagnoses have similar lab patterns)
3. Multi-hop reasoning (diagnosis → patient → medication → other patients → labs)
"""

import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

try:
    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T
except ImportError:
    logging.warning("PyTorch Geometric not installed. Graph construction will fail.")


# ============================================================================
# Node Indexing
# ============================================================================

class NodeIndexer:
    """
    Maps entity IDs (SUBJECT_ID, ITEMID, etc.) to contiguous node indices.

    Rationale:
        PyTorch Geometric requires node indices to be contiguous integers [0, N-1].
        MIMIC-III uses non-contiguous IDs (e.g., SUBJECT_ID=10006, 10011, ...)

        This class:
        1. Creates bidirectional mappings: ID ↔ index
        2. Ensures consistency across different edge types
        3. Enables reverse lookup for interpretation
    """

    def __init__(self):
        self.id_to_index = {}
        self.index_to_id = {}
        self.next_index = 0

    def add(self, entity_id) -> int:
        """
        Add entity ID and return its index.
        If already exists, return existing index.

        Args:
            entity_id: Original entity identifier

        Returns:
            Contiguous integer index
        """
        # Convert numeric types to int first (handles both int and float)
        # This ensures 10006.0 and 10006 map to the same string "10006"
        if isinstance(entity_id, (int, float, np.integer, np.floating)):
            entity_id = int(entity_id)

        # Convert to string for consistent hashing
        entity_id = str(entity_id)

        if entity_id not in self.id_to_index:
            idx = self.next_index
            self.id_to_index[entity_id] = idx
            self.index_to_id[idx] = entity_id
            self.next_index += 1
            return idx
        else:
            return self.id_to_index[entity_id]

    def get_index(self, entity_id) -> Optional[int]:
        """Get index for entity ID (returns None if not found)."""
        # Convert numeric types to int first (same as add())
        if isinstance(entity_id, (int, float, np.integer, np.floating)):
            entity_id = int(entity_id)
        return self.id_to_index.get(str(entity_id))

    def get_id(self, index: int) -> Optional[str]:
        """Get entity ID for index (returns None if not found)."""
        return self.index_to_id.get(index)

    def __len__(self) -> int:
        """Total number of entities."""
        return self.next_index

    def __repr__(self) -> str:
        return f"NodeIndexer(num_entities={len(self)})"


# ============================================================================
# Graph Construction
# ============================================================================

def build_heterogeneous_graph(
    cohort: pd.DataFrame,
    labs: pd.DataFrame,
    diagnoses: pd.DataFrame,
    medications: pd.DataFrame,
    demographics: pd.DataFrame,
    labitems: pd.DataFrame,
    config: Dict
) -> HeteroData:
    """
    Construct heterogeneous graph from preprocessed data.

    Args:
        cohort: Patient cohort DataFrame
        labs: Normalized lab values (SUBJECT_ID, ITEMID, VALUE_NORMALIZED)
        diagnoses: Patient-diagnosis pairs (SUBJECT_ID, ICD3_CODE)
        medications: Patient-medication pairs (SUBJECT_ID, DRUG)
        demographics: Patient demographic features
        labitems: Lab item metadata
        config: Configuration dictionary

    Returns:
        HeteroData object containing graph structure and features

    Graph Schema:
        Node Types:
        - patient: One node per patient
        - lab: One node per unique lab test
        - diagnosis: One node per diagnosis code
        - medication: One node per medication

        Edge Types:
        - (patient, has_lab, lab): Edge with continuous attribute (lab value)
        - (lab, has_lab_rev, patient): Reverse edges for message passing
        - (patient, has_diagnosis, diagnosis): Binary edge
        - (diagnosis, has_diagnosis_rev, patient): Reverse edges
        - (patient, has_medication, medication): Binary edge
        - (medication, has_medication_rev, patient): Reverse edges
    """
    logging.info("="*70)
    logging.info("Building Heterogeneous Graph")
    logging.info("="*70)

    # Initialize HeteroData
    data = HeteroData()

    # ------------------------------------------------------------------------
    # Create Node Indexers
    # ------------------------------------------------------------------------
    logging.info("Creating node indexers...")

    indexers = {
        'patient': NodeIndexer(),
        'lab': NodeIndexer(),
        'diagnosis': NodeIndexer(),
        'medication': NodeIndexer()
    }

    # Index all entities
    for subject_id in cohort['SUBJECT_ID']:
        indexers['patient'].add(subject_id)

    for itemid in labs['ITEMID'].unique():
        indexers['lab'].add(itemid)

    for dx_code in diagnoses['ICD3_CODE'].unique():
        indexers['diagnosis'].add(dx_code)

    for drug in medications['DRUG'].unique():
        indexers['medication'].add(drug)

    logging.info(f"Node counts:")
    for node_type, indexer in indexers.items():
        logging.info(f"  {node_type}: {len(indexer)}")

    # ------------------------------------------------------------------------
    # Create Node Features
    # ------------------------------------------------------------------------
    logging.info("\nCreating node features...")

    # Patient features: learnable embeddings (Iteration 6 - pure learnable)
    # No handcrafted features, let model learn optimal representations
    data['patient'].num_nodes = len(indexers['patient'])
    logging.info(f"  Patient nodes: {data['patient'].num_nodes} (embeddings will be learned)")

    # Lab features: learnable embeddings (initialized in model)
    # Store metadata for now
    data['lab'].num_nodes = len(indexers['lab'])
    lab_metadata = create_lab_metadata(labitems, indexers['lab'])
    data['lab'].metadata = lab_metadata
    logging.info(f"  Lab nodes: {data['lab'].num_nodes} (embeddings will be learned)")

    # Diagnosis features: learnable embeddings (hybrid approach - avoid sparse one-hot)
    data['diagnosis'].num_nodes = len(indexers['diagnosis'])
    logging.info(f"  Diagnosis nodes: {data['diagnosis'].num_nodes} (embeddings will be learned)")

    # Medication features: learnable embeddings (hybrid approach - avoid sparse one-hot)
    data['medication'].num_nodes = len(indexers['medication'])
    logging.info(f"  Medication nodes: {data['medication'].num_nodes} (embeddings will be learned)")

    # ------------------------------------------------------------------------
    # Create Edges
    # ------------------------------------------------------------------------
    logging.info("\nCreating edges...")

    edge_config = config['graph']['edge_types']

    # Patient-Lab edges (with edge attributes = lab values)
    if edge_config['patient_lab']['enabled']:
        patient_lab_edges, lab_values = create_patient_lab_edges(
            labs, indexers['patient'], indexers['lab']
        )
        data['patient', 'has_lab', 'lab'].edge_index = patient_lab_edges
        data['patient', 'has_lab', 'lab'].edge_attr = lab_values
        logging.info(f"  (patient, has_lab, lab): {patient_lab_edges.shape[1]} edges")

        # Reverse edges for bidirectional message passing
        if edge_config['patient_lab']['bidirectional']:
            data['lab', 'has_lab_rev', 'patient'].edge_index = patient_lab_edges.flip(0)
            data['lab', 'has_lab_rev', 'patient'].edge_attr = lab_values
            logging.info(f"  (lab, has_lab_rev, patient): {patient_lab_edges.shape[1]} edges (reverse)")

    # Patient-Diagnosis edges
    if edge_config['patient_diagnosis']['enabled']:
        patient_dx_edges = create_patient_diagnosis_edges(
            diagnoses, indexers['patient'], indexers['diagnosis']
        )
        data['patient', 'has_diagnosis', 'diagnosis'].edge_index = patient_dx_edges
        logging.info(f"  (patient, has_diagnosis, diagnosis): {patient_dx_edges.shape[1]} edges")

        if edge_config['patient_diagnosis']['bidirectional']:
            data['diagnosis', 'has_diagnosis_rev', 'patient'].edge_index = patient_dx_edges.flip(0)
            logging.info(f"  (diagnosis, has_diagnosis_rev, patient): {patient_dx_edges.shape[1]} edges (reverse)")

    # Patient-Medication edges
    if edge_config['patient_medication']['enabled']:
        patient_med_edges = create_patient_medication_edges(
            medications, indexers['patient'], indexers['medication']
        )
        data['patient', 'has_medication', 'medication'].edge_index = patient_med_edges
        logging.info(f"  (patient, has_medication, medication): {patient_med_edges.shape[1]} edges")

        if edge_config['patient_medication']['bidirectional']:
            data['medication', 'has_medication_rev', 'patient'].edge_index = patient_med_edges.flip(0)
            logging.info(f"  (medication, has_medication_rev, patient): {patient_med_edges.shape[1]} edges (reverse)")

    # ------------------------------------------------------------------------
    # Add Metadata
    # ------------------------------------------------------------------------
    # Convert indexers to dict format (avoids pickling issues)
    data.indexers = {
        node_type: {
            'id_to_index': indexer.id_to_index,
            'index_to_id': indexer.index_to_id
        }
        for node_type, indexer in indexers.items()
    }
    data.config = config  # Store config

    # ------------------------------------------------------------------------
    # Validate Graph
    # ------------------------------------------------------------------------
    logging.info("\nValidating graph...")
    validate_graph(data)

    logging.info("\n" + "="*70)
    logging.info("Graph construction complete!")
    logging.info("="*70)

    return data


# ============================================================================
# Feature Creation
# ============================================================================

def create_patient_features(
    cohort: pd.DataFrame,
    demographics: pd.DataFrame,
    indexer: NodeIndexer
) -> torch.Tensor:
    """
    Create feature matrix for patient nodes.

    Args:
        cohort: Patient cohort DataFrame
        demographics: Demographic features DataFrame
        indexer: Patient node indexer

    Returns:
        Tensor of shape [num_patients, num_features]

    Rationale:
        Patient features serve as initial node embeddings.
        These are updated through GNN message passing.
    """
    # Merge cohort with demographics
    patient_data = cohort[['SUBJECT_ID']].merge(
        demographics,
        on='SUBJECT_ID',
        how='left'
    )

    # Sort by node index to ensure alignment
    patient_data['node_idx'] = patient_data['SUBJECT_ID'].apply(indexer.get_index)
    patient_data = patient_data.sort_values('node_idx')

    # Extract feature columns (exclude ID columns)
    feature_cols = [col for col in patient_data.columns
                   if col not in ['SUBJECT_ID', 'node_idx']]

    if len(feature_cols) == 0:
        # No features, create dummy feature
        logging.warning("No patient features found, using dummy feature")
        features = torch.ones((len(patient_data), 1))
    else:
        # Convert to float to handle boolean columns (GENDER_F, GENDER_M)
        features = torch.tensor(
            patient_data[feature_cols].astype(float).values,
            dtype=torch.float32
        )

    return features


def create_lab_metadata(
    labitems: pd.DataFrame,
    indexer: NodeIndexer
) -> Dict:
    """
    Create metadata dictionary for lab nodes.

    Args:
        labitems: Lab items DataFrame
        indexer: Lab node indexer

    Returns:
        Dictionary mapping node index to lab metadata
    """
    metadata = {}

    for _, row in labitems.iterrows():
        idx = indexer.get_index(row['ITEMID'])
        if idx is not None:
            metadata[idx] = {
                'itemid': row['ITEMID'],
                'label': row.get('LABEL', 'Unknown'),
                'fluid': row.get('FLUID', 'Unknown'),
                'category': row.get('CATEGORY', 'Unknown')
            }

    return metadata


def create_diagnosis_features(
    diagnoses: pd.DataFrame,
    indexer: NodeIndexer
) -> torch.Tensor:
    """
    Create feature matrix for diagnosis nodes from hierarchy and priority.

    Features:
    - Diagnosis category (one-hot, ~20 categories)
    - Diagnosis priority (one-hot, 3 categories: Primary, Major, Other)

    Args:
        diagnoses: Diagnosis DataFrame with DIAGNOSIS_CATEGORY, DIAGNOSIS_PRIORITY
        indexer: Diagnosis node indexer

    Returns:
        Tensor of shape [num_diagnoses, num_features]
    """
    import logging

    # Group by ICD3_CODE to get unique diagnosis nodes
    dx_features = diagnoses.groupby('ICD3_CODE').agg({
        'DIAGNOSIS_CATEGORY': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
        'DIAGNOSIS_PRIORITY': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Other'
    }).reset_index()

    # One-hot encode category
    category_dummies = pd.get_dummies(dx_features['DIAGNOSIS_CATEGORY'], prefix='DX_CAT')

    # One-hot encode priority
    priority_dummies = pd.get_dummies(dx_features['DIAGNOSIS_PRIORITY'], prefix='PRI')

    # Combine features
    feature_df = pd.concat([dx_features[['ICD3_CODE']], category_dummies, priority_dummies], axis=1)

    # Map to node indices and create tensor
    num_nodes = len(indexer)
    num_features = len(category_dummies.columns) + len(priority_dummies.columns)

    features = torch.zeros((num_nodes, num_features), dtype=torch.float32)

    for _, row in feature_df.iterrows():
        idx = indexer.get_index(row['ICD3_CODE'])
        if idx is not None:
            feature_values = row[1:].values.astype(float)
            features[idx] = torch.tensor(feature_values, dtype=torch.float32)

    logging.info(f"Created diagnosis features: {num_features} features for {num_nodes} nodes")
    return features


def create_medication_features(
    medications: pd.DataFrame,
    indexer: NodeIndexer
) -> torch.Tensor:
    """
    Create feature matrix for medication nodes from route, frequency, and flags.

    Features:
    - Route of administration (one-hot, ~10 categories)
    - Frequency (one-hot, ~15 categories)
    - PRN flag (binary)
    - IV admixture flag (binary)

    Args:
        medications: Medication DataFrame with ROUTE, FREQUENCY, PRN, IV_ADMIXTURE
        indexer: Medication node indexer

    Returns:
        Tensor of shape [num_medications, num_features]
    """
    import logging

    # Group by DRUG to get unique medication nodes (aggregate metadata)
    med_features = medications.groupby('DRUG').agg({
        'ROUTE': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
        'FREQUENCY': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
        'PRN': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'No',
        'IV_ADMIXTURE': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'No'
    }).reset_index()

    # One-hot encode route
    route_dummies = pd.get_dummies(med_features['ROUTE'], prefix='ROUTE')

    # One-hot encode frequency
    freq_dummies = pd.get_dummies(med_features['FREQUENCY'], prefix='FREQ')

    # Binary flags
    med_features['PRN_FLAG'] = (med_features['PRN'].str.lower() == 'yes').astype(int)
    med_features['IV_FLAG'] = (med_features['IV_ADMIXTURE'].str.lower() == 'yes').astype(int)

    # Combine features
    feature_df = pd.concat([
        med_features[['DRUG', 'PRN_FLAG', 'IV_FLAG']],
        route_dummies,
        freq_dummies
    ], axis=1)

    # Map to node indices and create tensor
    num_nodes = len(indexer)
    num_features = 2 + len(route_dummies.columns) + len(freq_dummies.columns)

    features = torch.zeros((num_nodes, num_features), dtype=torch.float32)

    for _, row in feature_df.iterrows():
        idx = indexer.get_index(row['DRUG'])
        if idx is not None:
            feature_values = row[1:].values.astype(float)
            features[idx] = torch.tensor(feature_values, dtype=torch.float32)

    logging.info(f"Created medication features: {num_features} features for {num_nodes} nodes")
    return features


# ============================================================================
# Edge Creation
# ============================================================================

def create_patient_lab_edges(
    labs: pd.DataFrame,
    patient_indexer: NodeIndexer,
    lab_indexer: NodeIndexer
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create patient-lab edges with edge attributes (normalized lab values).

    Args:
        labs: Lab DataFrame (SUBJECT_ID, ITEMID, VALUE_NORMALIZED)
        patient_indexer: Patient node indexer
        lab_indexer: Lab node indexer

    Returns:
        Tuple of (edge_index, edge_attr)
        - edge_index: [2, num_edges] tensor of (patient_idx, lab_idx) pairs
        - edge_attr: [num_edges, 1] tensor of lab values

    Rationale:
        Lab values are continuous and vary in clinical meaning.
        Storing them as edge attributes allows the GNN to weight
        message passing by lab value magnitude.
    """
    edge_list = []
    edge_attrs = []

    for _, row in labs.iterrows():
        patient_idx = patient_indexer.get_index(row['SUBJECT_ID'])
        lab_idx = lab_indexer.get_index(row['ITEMID'])

        if patient_idx is not None and lab_idx is not None:
            edge_list.append([patient_idx, lab_idx])
            edge_attrs.append(row['VALUE_NORMALIZED'])

    if len(edge_list) == 0:
        # No edges found, return empty tensors with correct shape
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).unsqueeze(1)

    return edge_index, edge_attr


def create_patient_diagnosis_edges(
    diagnoses: pd.DataFrame,
    patient_indexer: NodeIndexer,
    diagnosis_indexer: NodeIndexer
) -> torch.Tensor:
    """
    Create patient-diagnosis edges.

    Args:
        diagnoses: Diagnosis DataFrame (SUBJECT_ID, ICD3_CODE)
        patient_indexer: Patient node indexer
        diagnosis_indexer: Diagnosis node indexer

    Returns:
        edge_index: [2, num_edges] tensor
    """
    edge_list = []

    for _, row in diagnoses.iterrows():
        patient_idx = patient_indexer.get_index(row['SUBJECT_ID'])
        dx_idx = diagnosis_indexer.get_index(row['ICD3_CODE'])

        if patient_idx is not None and dx_idx is not None:
            edge_list.append([patient_idx, dx_idx])

    if len(edge_list) == 0:
        # No edges found, return empty tensor with correct shape
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return edge_index


def create_patient_medication_edges(
    medications: pd.DataFrame,
    patient_indexer: NodeIndexer,
    medication_indexer: NodeIndexer
) -> torch.Tensor:
    """
    Create patient-medication edges.

    Args:
        medications: Medication DataFrame (SUBJECT_ID, DRUG)
        patient_indexer: Patient node indexer
        medication_indexer: Medication node indexer

    Returns:
        edge_index: [2, num_edges] tensor
    """
    edge_list = []

    for _, row in medications.iterrows():
        patient_idx = patient_indexer.get_index(row['SUBJECT_ID'])
        med_idx = medication_indexer.get_index(row['DRUG'])

        if patient_idx is not None and med_idx is not None:
            edge_list.append([patient_idx, med_idx])

    if len(edge_list) == 0:
        # No edges found, return empty tensor with correct shape
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return edge_index


# ============================================================================
# Graph Validation
# ============================================================================

def validate_graph(data: HeteroData) -> None:
    """
    Validate graph structure and report potential issues.

    Args:
        data: HeteroData object

    Raises:
        ValueError: If critical validation fails
    """
    logging.info("Running graph validation...")

    # Check node counts
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        if num_nodes == 0:
            logging.warning(f"Node type '{node_type}' has 0 nodes!")

    # Check edge indices
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index

        if edge_index.shape[0] != 2:
            raise ValueError(f"Edge type {edge_type} has invalid shape: {edge_index.shape}")

        # Check for out-of-bounds indices
        src_type, _, dst_type = edge_type
        src_indices = edge_index[0]
        dst_indices = edge_index[1]

        max_src = data[src_type].num_nodes
        max_dst = data[dst_type].num_nodes

        # Only validate indices if there are edges
        if src_indices.numel() > 0:
            if src_indices.max() >= max_src:
                raise ValueError(f"Edge type {edge_type} has out-of-bounds source index: {src_indices.max()} >= {max_src}")

        if dst_indices.numel() > 0:
            if dst_indices.max() >= max_dst:
                raise ValueError(f"Edge type {edge_type} has out-of-bounds destination index: {dst_indices.max()} >= {max_dst}")

        logging.info(f"  Edge type {edge_type}: ✓ valid")

    logging.info("Graph validation passed!")


# ============================================================================
# Graph Statistics
# ============================================================================

def compute_graph_statistics(data: HeteroData) -> Dict:
    """
    Compute and log graph statistics.

    Args:
        data: HeteroData object

    Returns:
        Dictionary of statistics

    Rationale:
        Understanding graph properties helps:
        1. Detect data quality issues (isolated nodes, missing edges)
        2. Choose appropriate model architecture (depth, aggregation)
        3. Identify potential biases (degree imbalance)
    """
    stats = {}

    logging.info("\n" + "="*70)
    logging.info("GRAPH STATISTICS")
    logging.info("="*70)

    # Node counts
    stats['node_counts'] = {}
    logging.info("\nNode Counts:")
    for node_type in data.node_types:
        count = data[node_type].num_nodes
        stats['node_counts'][node_type] = count
        logging.info(f"  {node_type}: {count}")

    # Edge counts
    stats['edge_counts'] = {}
    logging.info("\nEdge Counts:")
    for edge_type in data.edge_types:
        count = data[edge_type].edge_index.shape[1]
        stats['edge_counts'][edge_type] = count
        logging.info(f"  {edge_type}: {count}")

    # Degree statistics for patient nodes
    logging.info("\nPatient Node Degree Statistics:")

    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type

        if src_type == 'patient':
            edge_index = data[edge_type].edge_index
            degrees = torch.bincount(edge_index[0], minlength=data['patient'].num_nodes)

            stats[f'patient_degree_{rel}'] = {
                'mean': degrees.float().mean().item(),
                'std': degrees.float().std().item(),
                'min': degrees.min().item(),
                'max': degrees.max().item(),
                'median': degrees.median().item()
            }

            logging.info(f"  {rel}:")
            logging.info(f"    Mean: {degrees.float().mean():.2f}")
            logging.info(f"    Std: {degrees.float().std():.2f}")
            logging.info(f"    Min: {degrees.min()}")
            logging.info(f"    Max: {degrees.max()}")
            logging.info(f"    Median: {degrees.median()}")

    # Density
    num_patients = data['patient'].num_nodes
    num_labs = data['lab'].num_nodes

    if ('patient', 'has_lab', 'lab') in data.edge_types:
        num_edges = data['patient', 'has_lab', 'lab'].edge_index.shape[1]
        max_edges = num_patients * num_labs
        density = num_edges / max_edges
        stats['density_patient_lab'] = density
        logging.info(f"\nPatient-Lab Graph Density: {density:.4f} ({num_edges}/{max_edges})")

    logging.info("="*70)

    return stats


# ============================================================================
# Main Pipeline
# ============================================================================

def build_graph_from_preprocessed(
    interim_dir: Path,
    config: Dict,
    output_path: Optional[Path] = None
) -> HeteroData:
    """
    Build graph from preprocessed parquet files.

    Args:
        interim_dir: Directory containing preprocessed files
        config: Configuration dictionary
        output_path: Where to save the graph (optional)

    Returns:
        HeteroData object
    """
    logging.info("Loading preprocessed data...")

    interim_dir = Path(interim_dir)

    # Load all preprocessed files
    cohort = pd.read_parquet(interim_dir / "cohort.parquet")
    labs = pd.read_parquet(interim_dir / "labs_normalized.parquet")
    diagnoses = pd.read_parquet(interim_dir / "diagnoses.parquet")
    medications = pd.read_parquet(interim_dir / "medications.parquet")
    demographics = pd.read_parquet(interim_dir / "demographics.parquet")
    labitems = pd.read_parquet(interim_dir / "labitems.parquet")

    logging.info("All preprocessed files loaded successfully")

    # Build graph
    graph = build_heterogeneous_graph(
        cohort, labs, diagnoses, medications, demographics, labitems, config
    )

    # Compute statistics
    stats = compute_graph_statistics(graph)

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, output_path)
        logging.info(f"Graph saved to {output_path}")

    return graph


# ============================================================================
# Command-line Interface
# ============================================================================

if __name__ == "__main__":
    """
    Build graph from command line.

    Usage:
        python graph_build.py
    """
    import sys
    sys.path.append(str(Path(__file__).parent))

    from utils import load_config, setup_logging, set_random_seeds

    # Load configuration
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    config = load_config(str(config_path))

    # Setup
    setup_logging(level=config['logging']['level'])
    set_random_seeds(config['train']['seed'])

    # Build graph
    interim_dir = Path(config['data']['interim_dir'])
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = build_graph_from_preprocessed(
        interim_dir,
        config,
        output_path=output_dir / "graph.pt"
    )

    logging.info("\nGraph construction complete!")

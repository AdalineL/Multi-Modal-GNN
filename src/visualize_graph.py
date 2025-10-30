"""
Graph Structure Visualization

This script visualizes the heterogeneous graph BEFORE training.
Shows:
1. Patient-centered subgraphs (local neighborhoods)
2. Graph statistics and connectivity
3. Node type distributions
4. Network visualization using NetworkX

Run this AFTER graph_build.py but BEFORE train.py to understand graph structure.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from typing import Dict, Optional, List, Set
import networkx as nx
from matplotlib.patches import Rectangle

# Import NodeIndexer for loading pickled graph
from graph_build import NodeIndexer

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# ============================================================================
# Patient-Centered Subgraph Visualization
# ============================================================================

def extract_patient_subgraph(
    graph,
    patient_idx: int,
    max_neighbors: int = 10
) -> Dict:
    """
    Extract k-hop neighborhood around a patient.

    Args:
        graph: HeteroData object
        patient_idx: Patient node index
        max_neighbors: Maximum neighbors to show per node type

    Returns:
        Dictionary containing subgraph information
    """
    subgraph = {
        'patient': patient_idx,
        'labs': [],
        'diagnoses': [],
        'medications': [],
        'lab_values': [],
        'lab_names': []
    }

    # Get patient's labs
    patient_lab_edges = graph['patient', 'has_lab', 'lab'].edge_index
    patient_lab_attrs = graph['patient', 'has_lab', 'lab'].edge_attr

    # Find edges connected to this patient
    mask = patient_lab_edges[0] == patient_idx
    if mask.sum() > 0:
        lab_indices = patient_lab_edges[1][mask][:max_neighbors]
        lab_values = patient_lab_attrs[mask][:max_neighbors].squeeze().cpu().numpy()

        subgraph['labs'] = lab_indices.cpu().numpy().tolist()
        subgraph['lab_values'] = lab_values.tolist()

        # Get lab names if available
        if hasattr(graph['lab'], 'metadata') and graph['lab'].metadata:
            for lab_idx in subgraph['labs']:
                lab_info = graph['lab'].metadata.get(int(lab_idx), {})
                subgraph['lab_names'].append(lab_info.get('label', f'Lab_{lab_idx}'))
        else:
            subgraph['lab_names'] = [f'Lab_{idx}' for idx in subgraph['labs']]

    # Get patient's diagnoses
    if ('patient', 'has_diagnosis', 'diagnosis') in graph.edge_types:
        patient_dx_edges = graph['patient', 'has_diagnosis', 'diagnosis'].edge_index
        mask = patient_dx_edges[0] == patient_idx
        if mask.sum() > 0:
            dx_indices = patient_dx_edges[1][mask][:max_neighbors]
            subgraph['diagnoses'] = dx_indices.cpu().numpy().tolist()

    # Get patient's medications
    if ('patient', 'has_medication', 'medication') in graph.edge_types:
        patient_med_edges = graph['patient', 'has_medication', 'medication'].edge_index
        mask = patient_med_edges[0] == patient_idx
        if mask.sum() > 0:
            med_indices = patient_med_edges[1][mask][:max_neighbors]
            subgraph['medications'] = med_indices.cpu().numpy().tolist()

    return subgraph


def plot_patient_subgraph(
    subgraph: Dict,
    patient_id: str,
    output_path: Optional[Path] = None
):
    """
    Visualize patient-centered subgraph using NetworkX.

    Args:
        subgraph: Subgraph dictionary from extract_patient_subgraph
        patient_id: Patient identifier (for title)
        output_path: Where to save the plot

    Rationale:
        Shows the actual graph structure for a single patient:
        - Which labs they have (with values)
        - Which diagnoses
        - Which medications
        This helps understand what information the GNN uses for prediction.
    """
    logging.info(f"Plotting subgraph for patient {patient_id}...")

    # Create directed graph
    G = nx.DiGraph()

    # Add patient node (central)
    patient_node = f"Patient_{patient_id}"
    G.add_node(patient_node, node_type='patient')

    # Add lab nodes and edges
    for i, (lab_idx, lab_name, lab_value) in enumerate(
        zip(subgraph['labs'], subgraph['lab_names'], subgraph['lab_values'])
    ):
        lab_node = f"{lab_name[:15]}"  # Truncate long names
        G.add_node(lab_node, node_type='lab')
        G.add_edge(patient_node, lab_node, value=lab_value, edge_type='has_lab')

    # Add diagnosis nodes and edges
    for dx_idx in subgraph['diagnoses']:
        dx_node = f"Dx_{dx_idx}"
        G.add_node(dx_node, node_type='diagnosis')
        G.add_edge(patient_node, dx_node, edge_type='has_diagnosis')

    # Add medication nodes and edges
    for med_idx in subgraph['medications']:
        med_node = f"Med_{med_idx}"
        G.add_node(med_node, node_type='medication')
        G.add_edge(patient_node, med_node, edge_type='has_medication')

    # Layout: circular with patient in center
    pos = {}

    # Patient at center
    pos[patient_node] = (0, 0)

    # Labs on the right
    labs = [n for n, d in G.nodes(data=True) if d['node_type'] == 'lab']
    for i, lab in enumerate(labs):
        angle = 2 * np.pi * i / max(len(labs), 1) + np.pi/2
        pos[lab] = (2 * np.cos(angle), 2 * np.sin(angle))

    # Diagnoses on the left
    diagnoses = [n for n, d in G.nodes(data=True) if d['node_type'] == 'diagnosis']
    for i, dx in enumerate(diagnoses):
        angle = 2 * np.pi * i / max(len(diagnoses), 1) + np.pi
        pos[dx] = (3 * np.cos(angle), 3 * np.sin(angle))

    # Medications at bottom
    medications = [n for n, d in G.nodes(data=True) if d['node_type'] == 'medication']
    for i, med in enumerate(medications):
        angle = 2 * np.pi * i / max(len(medications), 1) - np.pi/2
        pos[med] = (2.5 * np.cos(angle), 2.5 * np.sin(angle))

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Node colors by type
    node_colors = []
    for node, data in G.nodes(data=True):
        if data['node_type'] == 'patient':
            node_colors.append('#FF6B6B')  # Red
        elif data['node_type'] == 'lab':
            node_colors.append('#4ECDC4')  # Teal
        elif data['node_type'] == 'diagnosis':
            node_colors.append('#FFD93D')  # Yellow
        elif data['node_type'] == 'medication':
            node_colors.append('#95E1D3')  # Light green

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=800,
        alpha=0.9,
        ax=ax
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        arrows=True,
        arrowsize=15,
        width=2,
        alpha=0.6,
        ax=ax
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_weight='bold',
        ax=ax
    )

    # Add edge labels (lab values)
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        if 'value' in data:
            edge_labels[(u, v)] = f"{data['value']:.2f}"

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=7,
        ax=ax
    )

    # Title and legend
    ax.set_title(f'Patient {patient_id} Subgraph\n'
                f'{len(labs)} Labs | {len(diagnoses)} Diagnoses | {len(medications)} Medications',
                fontsize=14, fontweight='bold', pad=20)

    # Legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc='#FF6B6B', label='Patient'),
        Rectangle((0, 0), 1, 1, fc='#4ECDC4', label='Lab Test'),
        Rectangle((0, 0), 1, 1, fc='#FFD93D', label='Diagnosis'),
        Rectangle((0, 0), 1, 1, fc='#95E1D3', label='Medication')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved to {output_path}")

    plt.close()


# ============================================================================
# Graph Overview Visualization
# ============================================================================

def plot_graph_overview(
    graph,
    output_path: Optional[Path] = None
):
    """
    Create comprehensive overview of graph structure.

    Args:
        graph: HeteroData object
        output_path: Where to save

    Shows:
    - Node counts by type
    - Edge counts by type
    - Degree distributions
    - Connectivity statistics
    """
    logging.info("Creating graph overview visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Heterogeneous Graph Overview', fontsize=16, fontweight='bold', y=1.00)

    # ========================================================================
    # 1. Node Type Distribution
    # ========================================================================
    node_counts = {}
    for node_type in graph.node_types:
        node_counts[node_type] = graph[node_type].num_nodes

    ax = axes[0, 0]
    colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#95E1D3']
    bars = ax.bar(node_counts.keys(), node_counts.values(), color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Number of Nodes', fontsize=12)
    ax.set_title('Node Type Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ========================================================================
    # 2. Edge Type Distribution
    # ========================================================================
    edge_counts = {}
    for edge_type in graph.edge_types:
        src, rel, dst = edge_type
        label = f"{src}-{rel}-{dst}"
        edge_counts[label] = graph[edge_type].edge_index.shape[1]

    ax = axes[0, 1]
    # Truncate labels for readability
    labels = [label[:25] + '...' if len(label) > 25 else label for label in edge_counts.keys()]
    bars = ax.barh(range(len(edge_counts)), list(edge_counts.values()), color='#6C5CE7', alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(edge_counts)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Number of Edges', fontsize=12)
    ax.set_title('Edge Type Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {int(width):,}',
                ha='left', va='center', fontsize=9, fontweight='bold')

    # ========================================================================
    # 3. Patient Node Degree Distribution
    # ========================================================================
    ax = axes[1, 0]

    # Calculate patient degrees (total connections)
    patient_degrees = torch.zeros(graph['patient'].num_nodes)
    for edge_type in graph.edge_types:
        src_type, _, _ = edge_type
        if src_type == 'patient':
            edge_index = graph[edge_type].edge_index
            degrees = torch.bincount(edge_index[0], minlength=graph['patient'].num_nodes)
            patient_degrees += degrees

    patient_degrees = patient_degrees.cpu().numpy()

    ax.hist(patient_degrees, bins=30, color='#FF6B6B', alpha=0.7, edgecolor='black')
    ax.axvline(patient_degrees.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {patient_degrees.mean():.1f}')
    ax.axvline(np.median(patient_degrees), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(patient_degrees):.1f}')
    ax.set_xlabel('Degree (Total Connections)', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title('Patient Node Degree Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # 4. Graph Statistics Summary
    # ========================================================================
    ax = axes[1, 1]
    ax.axis('off')

    # Compute statistics
    total_nodes = sum(node_counts.values())
    total_edges = sum(edge_counts.values())

    # Patient-lab density
    num_patients = graph['patient'].num_nodes
    num_labs = graph['lab'].num_nodes
    num_patient_lab_edges = graph['patient', 'has_lab', 'lab'].edge_index.shape[1]
    density = num_patient_lab_edges / (num_patients * num_labs) if num_patients * num_labs > 0 else 0

    # Average labs per patient
    avg_labs_per_patient = patient_degrees.mean()

    # Statistics text
    stats_text = f"""
    GRAPH STATISTICS
    {'='*40}

    Total Nodes: {total_nodes:,}
    Total Edges: {total_edges:,}

    Node Breakdown:
    • Patients: {node_counts.get('patient', 0):,}
    • Lab Tests: {node_counts.get('lab', 0):,}
    • Diagnoses: {node_counts.get('diagnosis', 0):,}
    • Medications: {node_counts.get('medication', 0):,}

    Patient Statistics:
    • Avg. connections: {avg_labs_per_patient:.1f}
    • Min connections: {patient_degrees.min():.0f}
    • Max connections: {patient_degrees.max():.0f}

    Graph Density:
    • Patient-Lab: {density:.4f} ({density*100:.2f}%)

    Edge Distribution:
    • Patient-Lab: {num_patient_lab_edges:,}
    • Patient-Diagnosis: {graph['patient', 'has_diagnosis', 'diagnosis'].edge_index.shape[1] if ('patient', 'has_diagnosis', 'diagnosis') in graph.edge_types else 0:,}
    • Patient-Medication: {graph['patient', 'has_medication', 'medication'].edge_index.shape[1] if ('patient', 'has_medication', 'medication') in graph.edge_types else 0:,}
    """

    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved to {output_path}")

    plt.close()


# ============================================================================
# Network Visualization (Sampled)
# ============================================================================

def plot_network_sample(
    graph,
    num_patients: int = 20,
    output_path: Optional[Path] = None
):
    """
    Visualize a small sample of the full network.

    Args:
        graph: HeteroData object
        num_patients: Number of patients to include
        output_path: Where to save

    Rationale:
        Full graph is too large to visualize. Sample shows structure.
    """
    logging.info(f"Creating network visualization with {num_patients} patients...")

    # Create NetworkX graph
    G = nx.Graph()  # Undirected for simplicity

    # Sample patients
    patient_indices = np.random.choice(
        graph['patient'].num_nodes,
        size=min(num_patients, graph['patient'].num_nodes),
        replace=False
    )

    # Track which labs/diagnoses/medications are connected
    connected_labs = set()
    connected_diagnoses = set()
    connected_medications = set()

    # Add patient-lab edges
    patient_lab_edges = graph['patient', 'has_lab', 'lab'].edge_index.cpu().numpy()
    for patient_idx in patient_indices:
        G.add_node(f"P{patient_idx}", node_type='patient')

        mask = patient_lab_edges[0] == patient_idx
        for lab_idx in patient_lab_edges[1][mask]:
            connected_labs.add(lab_idx)
            G.add_node(f"L{lab_idx}", node_type='lab')
            G.add_edge(f"P{patient_idx}", f"L{lab_idx}")

    # Add patient-diagnosis edges
    if ('patient', 'has_diagnosis', 'diagnosis') in graph.edge_types:
        patient_dx_edges = graph['patient', 'has_diagnosis', 'diagnosis'].edge_index.cpu().numpy()
        for patient_idx in patient_indices:
            mask = patient_dx_edges[0] == patient_idx
            for dx_idx in patient_dx_edges[1][mask]:
                connected_diagnoses.add(dx_idx)
                G.add_node(f"D{dx_idx}", node_type='diagnosis')
                G.add_edge(f"P{patient_idx}", f"D{dx_idx}")

    # Add patient-medication edges
    if ('patient', 'has_medication', 'medication') in graph.edge_types:
        patient_med_edges = graph['patient', 'has_medication', 'medication'].edge_index.cpu().numpy()
        for patient_idx in patient_indices:
            mask = patient_med_edges[0] == patient_idx
            for med_idx in patient_med_edges[1][mask]:
                connected_medications.add(med_idx)
                G.add_node(f"M{med_idx}", node_type='medication')
                G.add_edge(f"P{patient_idx}", f"M{med_idx}")

    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 12))

    # Separate nodes by type
    patients = [n for n, d in G.nodes(data=True) if d['node_type'] == 'patient']
    labs = [n for n, d in G.nodes(data=True) if d['node_type'] == 'lab']
    diagnoses = [n for n, d in G.nodes(data=True) if d['node_type'] == 'diagnosis']
    medications = [n for n, d in G.nodes(data=True) if d['node_type'] == 'medication']

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=patients, node_color='#FF6B6B',
                          node_size=300, alpha=0.9, label='Patient', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=labs, node_color='#4ECDC4',
                          node_size=200, alpha=0.8, label='Lab', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=diagnoses, node_color='#FFD93D',
                          node_size=200, alpha=0.8, label='Diagnosis', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=medications, node_color='#95E1D3',
                          node_size=200, alpha=0.8, label='Medication', ax=ax)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=1, ax=ax)

    ax.set_title(f'Network Sample: {len(patients)} Patients | '
                f'{len(labs)} Labs | {len(diagnoses)} Diagnoses | {len(medications)} Medications',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved to {output_path}")

    plt.close()


# ============================================================================
# Main Function
# ============================================================================

def visualize_graph_structure(graph_path: Path, output_dir: Path, num_patient_examples: int = 5):
    """
    Generate all graph structure visualizations.

    Args:
        graph_path: Path to saved graph.pt
        output_dir: Where to save visualizations
        num_patient_examples: Number of patient subgraphs to plot
    """
    logging.info("="*70)
    logging.info("GRAPH STRUCTURE VISUALIZATION")
    logging.info("="*70)

    # Load graph
    logging.info(f"Loading graph from {graph_path}...")
    graph = torch.load(graph_path, map_location='cpu')
    logging.info("Graph loaded successfully")

    # Create output directory
    viz_dir = output_dir / "graph_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 1. Graph overview
    logging.info("\n1. Creating graph overview...")
    plot_graph_overview(graph, output_path=viz_dir / "graph_overview.png")

    # 2. Network sample
    logging.info("\n2. Creating network sample visualization...")
    plot_network_sample(graph, num_patients=20, output_path=viz_dir / "network_sample.png")

    # 3. Patient-centered subgraphs
    logging.info(f"\n3. Creating {num_patient_examples} patient subgraph examples...")
    num_patients = graph['patient'].num_nodes

    # Sample diverse patients (different degree levels)
    patient_lab_edges = graph['patient', 'has_lab', 'lab'].edge_index
    patient_degrees = torch.bincount(patient_lab_edges[0], minlength=num_patients).cpu().numpy()

    # Get patients with low, medium, high connectivity
    sorted_indices = np.argsort(patient_degrees)
    sample_indices = [
        sorted_indices[len(sorted_indices)//4],      # Low
        sorted_indices[len(sorted_indices)//2],      # Medium
        sorted_indices[3*len(sorted_indices)//4],    # High
    ]

    # Add random samples
    sample_indices.extend(np.random.choice(num_patients, size=min(2, num_patient_examples-3), replace=False))

    for i, patient_idx in enumerate(sample_indices[:num_patient_examples]):
        subgraph = extract_patient_subgraph(graph, patient_idx, max_neighbors=15)

        # Get patient ID from indexer if available
        if hasattr(graph, 'indexers') and 'patient' in graph.indexers:
            # Indexers are stored as dicts with 'index_to_id' key
            patient_id = graph.indexers['patient']['index_to_id'].get(patient_idx, f'Patient_{patient_idx}')
        else:
            patient_id = str(patient_idx)

        plot_patient_subgraph(
            subgraph,
            patient_id,
            output_path=viz_dir / f"patient_subgraph_{i+1}.png"
        )

    logging.info("\n" + "="*70)
    logging.info("GRAPH VISUALIZATION COMPLETE!")
    logging.info("="*70)
    logging.info(f"All visualizations saved to: {viz_dir}")
    logging.info(f"\nGenerated files:")
    logging.info(f"  • graph_overview.png - Overall graph statistics")
    logging.info(f"  • network_sample.png - Network visualization sample")
    logging.info(f"  • patient_subgraph_*.png - {num_patient_examples} patient examples")


# ============================================================================
# Command-line Interface
# ============================================================================

if __name__ == "__main__":
    """
    Visualize graph structure from command line.

    Usage:
        python visualize_graph.py
    """
    import sys
    sys.path.append(str(Path(__file__).parent))

    from utils import load_config, setup_logging

    # Load configuration
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    config = load_config(str(config_path))

    # Setup
    setup_logging(level=config['logging']['level'])

    # Paths
    graph_path = Path(config['data']['output_dir']) / "graph.pt"
    output_dir = Path(config['data']['output_dir'])

    if not graph_path.exists():
        logging.error(f"Graph not found: {graph_path}")
        logging.error("Please run graph_build.py first to create the graph.")
        sys.exit(1)

    # Visualize
    visualize_graph_structure(
        graph_path,
        output_dir,
        num_patient_examples=5
    )

    logging.info("\n✓ Graph visualization complete!")
    logging.info(f"✓ Check {output_dir}/graph_visualizations/ for outputs")

"""
Visualization Module

This module provides visualization tools for:
1. Graph structure (degree distributions, connectivity)
2. Training curves (loss over epochs)
3. Prediction quality (parity plots, error distributions)
4. Node embeddings (UMAP/t-SNE projections)
5. Example subgraphs (patient-centered views)

Rationale:
    Visualizations help:
    - Understand data characteristics (graph structure)
    - Monitor training (convergence, overfitting)
    - Interpret predictions (where model succeeds/fails)
    - Explain model behavior (embedding clusters, attention patterns)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch_geometric.data import HeteroData

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


# ============================================================================
# Graph Structure Visualizations
# ============================================================================

def plot_degree_distribution(
    graph: HeteroData,
    node_type: str = 'patient',
    output_path: Optional[Path] = None
):
    """
    Plot degree distribution for a specific node type.

    Args:
        graph: HeteroData object
        node_type: Node type to analyze
        output_path: Where to save the plot

    Rationale:
        Degree distribution reveals:
        - Data completeness (how many labs per patient)
        - Graph connectivity (isolated vs well-connected nodes)
        - Potential biases (some patients have way more data)
    """
    logging.info(f"Plotting degree distribution for {node_type} nodes...")

    degrees = []

    # Compute degrees from all edge types connected to this node
    for edge_type in graph.edge_types:
        src_type, _, dst_type = edge_type

        if src_type == node_type:
            edge_index = graph[edge_type].edge_index
            node_degrees = torch.bincount(edge_index[0], minlength=graph[node_type].num_nodes)
            degrees.append(node_degrees.cpu().numpy())

    # Sum degrees across all edge types
    total_degrees = np.sum(degrees, axis=0)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(total_degrees, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel(f'Degree (number of connections)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{node_type.capitalize()} Node Degree Distribution')
    axes[0].axvline(total_degrees.mean(), color='red', linestyle='--',
                   label=f'Mean: {total_degrees.mean():.1f}')
    axes[0].legend()

    # Box plot
    axes[1].boxplot(total_degrees, vert=True)
    axes[1].set_ylabel('Degree')
    axes[1].set_title(f'{node_type.capitalize()} Degree Box Plot')
    axes[1].set_xticklabels([node_type])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved to {output_path}")

    plt.close()


def plot_missingness_heatmap(
    labs_df: pd.DataFrame,
    labitems_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    max_labs: int = 50,
    max_patients: int = 100
):
    """
    Plot heatmap showing which labs are missing for which patients.

    Args:
        labs_df: Labs DataFrame (SUBJECT_ID, ITEMID, VALUE)
        labitems_df: Lab items DataFrame with labels
        output_path: Where to save
        max_labs: Maximum number of labs to show
        max_patients: Maximum number of patients to show

    Rationale:
        Missingness patterns reveal:
        - Which labs are commonly ordered together (clinical panels)
        - Which patients have sparse data (potentially harder to predict)
        - Systematic missingness (e.g., certain labs only for ICU patients)
    """
    logging.info("Creating missingness heatmap...")

    # Get top K most common labs
    top_labs = labs_df['ITEMID'].value_counts().head(max_labs).index.tolist()

    # Get subset of patients
    unique_patients = labs_df['SUBJECT_ID'].unique()[:max_patients]

    # Create presence/absence matrix
    matrix = []
    patient_ids = []

    for patient_id in unique_patients:
        patient_labs = labs_df[labs_df['SUBJECT_ID'] == patient_id]['ITEMID'].values
        row = [1 if lab_id in patient_labs else 0 for lab_id in top_labs]
        matrix.append(row)
        patient_ids.append(patient_id)

    matrix = np.array(matrix)

    # Get lab names
    lab_names = []
    for lab_id in top_labs:
        lab_info = labitems_df[labitems_df['ITEMID'] == lab_id]
        if len(lab_info) > 0:
            name = lab_info.iloc[0]['LABEL']
            # Truncate long names
            name = name[:20] if len(name) > 20 else name
            lab_names.append(name)
        else:
            lab_names.append(f"Lab {lab_id}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(
        matrix,
        cmap='RdYlGn',
        cbar_kws={'label': 'Lab Present'},
        ax=ax,
        xticklabels=lab_names,
        yticklabels=[f"P{i}" for i in range(len(patient_ids))],
        linewidths=0.1,
        linecolor='gray'
    )

    ax.set_xlabel('Lab Test')
    ax.set_ylabel('Patient')
    ax.set_title(f'Lab Missingness Pattern (Top {max_labs} Labs, {max_patients} Patients)')

    plt.xticks(rotation=90)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved to {output_path}")

    plt.close()


# ============================================================================
# Training Curve Visualizations
# ============================================================================

def plot_training_curves(
    history: Dict,
    output_path: Optional[Path] = None
):
    """
    Plot training and validation loss curves.

    Args:
        history: Training history dictionary
        output_path: Where to save

    Rationale:
        Loss curves diagnose training dynamics:
        - Convergence: Are losses plateauing?
        - Overfitting: Is val loss increasing while train loss decreases?
        - Instability: Are there sudden spikes?
    """
    logging.info("Plotting training curves...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MAE)')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Find best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    axes[0].axvline(best_epoch, color='red', linestyle='--', alpha=0.5,
                   label=f'Best Epoch: {best_epoch}')
    axes[0].text(best_epoch, best_val_loss, f'  {best_val_loss:.4f}',
                verticalalignment='bottom')

    # Learning rate
    if 'learning_rates' in history:
        axes[1].plot(epochs, history['learning_rates'], marker='o', markersize=3, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved to {output_path}")

    plt.close()


# ============================================================================
# Prediction Quality Visualizations
# ============================================================================

def plot_parity(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Parity Plot: Predicted vs True Lab Values"
):
    """
    Plot parity plot (predicted vs true values).

    Args:
        predictions: Predicted values
        targets: True values
        output_path: Where to save
        title: Plot title

    Rationale:
        Parity plots show prediction quality:
        - Perfect predictions lie on diagonal line (y=x)
        - Scatter around diagonal indicates error
        - Systematic bias shows as offset from diagonal
        - Heteroscedasticity shows as varying spread
    """
    logging.info("Creating parity plot...")

    # Compute metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    r2 = 1 - np.sum((targets - predictions)**2) / np.sum((targets - np.mean(targets))**2)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot with transparency
    ax.scatter(targets, predictions, alpha=0.3, s=10)

    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Labels and title
    ax.set_xlabel('True Lab Value (Normalized)')
    ax.set_ylabel('Predicted Lab Value (Normalized)')
    ax.set_title(title)

    # Add metrics text box
    metrics_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}'
    ax.text(0.05, 0.95, metrics_text,
           transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10)

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved to {output_path}")

    plt.close()


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Optional[Path] = None
):
    """
    Plot distribution of prediction errors.

    Args:
        predictions: Predicted values
        targets: True values
        output_path: Where to save

    Rationale:
        Error distribution reveals:
        - Central tendency: Is mean error zero? (unbiased)
        - Spread: How much variability in errors?
        - Outliers: Are there extreme errors?
        - Skewness: Is model systematically over/under-predicting?
    """
    logging.info("Plotting error distribution...")

    errors = predictions - targets

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].axvline(errors.mean(), color='green', linestyle='--', linewidth=2,
                   label=f'Mean Error: {errors.mean():.4f}')
    axes[0].set_xlabel('Prediction Error (Predicted - True)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Q-Q plot (check if errors are normally distributed)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normal Distribution)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved to {output_path}")

    plt.close()


def plot_per_lab_performance(
    per_lab_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    top_k: int = 20
):
    """
    Plot per-lab performance metrics.

    Args:
        per_lab_df: DataFrame with per-lab metrics
        output_path: Where to save
        top_k: Number of labs to show

    Rationale:
        Identifies which lab tests are easy vs hard to predict.
        Helps prioritize improvements and understand model limitations.
    """
    logging.info(f"Plotting per-lab performance (top {top_k} best and worst)...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top K best
    best_labs = per_lab_df.head(top_k)
    axes[0].barh(range(len(best_labs)), best_labs['mae'], color='green', alpha=0.7)
    axes[0].set_yticks(range(len(best_labs)))
    axes[0].set_yticklabels(best_labs['lab_name'], fontsize=8)
    axes[0].set_xlabel('MAE')
    axes[0].set_title(f'Top {top_k} Best Predicted Labs')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')

    # Top K worst
    worst_labs = per_lab_df.tail(top_k)
    axes[1].barh(range(len(worst_labs)), worst_labs['mae'], color='red', alpha=0.7)
    axes[1].set_yticks(range(len(worst_labs)))
    axes[1].set_yticklabels(worst_labs['lab_name'], fontsize=8)
    axes[1].set_xlabel('MAE')
    axes[1].set_title(f'Top {top_k} Worst Predicted Labs')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved to {output_path}")

    plt.close()


# ============================================================================
# Embedding Visualizations
# ============================================================================

def plot_embeddings_umap(
    embeddings: torch.Tensor,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    title: str = "UMAP Projection of Node Embeddings"
):
    """
    Plot UMAP projection of node embeddings.

    Args:
        embeddings: Node embedding tensor [num_nodes, embedding_dim]
        labels: Optional labels for coloring (e.g., node type, cluster)
        label_names: Names for each label value
        output_path: Where to save
        title: Plot title

    Rationale:
        UMAP visualization reveals learned structure:
        - Clusters: Do similar nodes group together?
        - Separation: Are different node types/classes well-separated?
        - Outliers: Are there unusual nodes?

        This helps validate that the GNN is learning meaningful representations.
    """
    logging.info("Computing UMAP projection...")

    try:
        from umap import UMAP
    except ImportError:
        logging.warning("UMAP not installed. Skipping embedding visualization.")
        return

    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Compute UMAP
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        # Color by labels
        scatter = ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.6,
            s=20
        )

        # Add legend
        if label_names is not None:
            handles, _ = scatter.legend_elements()
            ax.legend(handles, label_names, title="Labels", loc='best')
        else:
            plt.colorbar(scatter, ax=ax, label='Label')
    else:
        # No labels, just show points
        ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            alpha=0.6,
            s=20
        )

    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved to {output_path}")

    plt.close()


# ============================================================================
# Main Visualization Pipeline
# ============================================================================

def create_all_visualizations(
    graph: HeteroData,
    history: Dict,
    predictions: np.ndarray,
    targets: np.ndarray,
    per_lab_df: pd.DataFrame,
    output_dir: Path,
    config: Dict
):
    """
    Generate all visualization outputs.

    Args:
        graph: HeteroData graph
        history: Training history
        predictions: Model predictions
        targets: True values
        per_lab_df: Per-lab metrics
        output_dir: Where to save plots
        config: Configuration dictionary
    """
    logging.info("="*70)
    logging.info("Generating Visualizations")
    logging.info("="*70)

    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Graph structure
    if config['visualization'].get('plot_degree_distribution', True):
        plot_degree_distribution(
            graph,
            node_type='patient',
            output_path=viz_dir / "patient_degree_distribution.png"
        )

    # Training curves
    plot_training_curves(
        history,
        output_path=viz_dir / "training_curves.png"
    )

    # Prediction quality
    if config['visualization'].get('generate_parity_plots', True):
        plot_parity(
            predictions,
            targets,
            output_path=viz_dir / "parity_plot.png"
        )

    plot_error_distribution(
        predictions,
        targets,
        output_path=viz_dir / "error_distribution.png"
    )

    # Per-lab performance
    if per_lab_df is not None:
        top_k = config['visualization'].get('top_labs_to_plot', 10)
        plot_per_lab_performance(
            per_lab_df,
            output_path=viz_dir / "per_lab_performance.png",
            top_k=top_k
        )

    logging.info(f"\nAll visualizations saved to {viz_dir}")


# ============================================================================
# Command-line Interface
# ============================================================================

if __name__ == "__main__":
    """
    Generate visualizations from command line.

    Usage:
        python visualize.py
    """
    import sys
    sys.path.append(str(Path(__file__).parent))

    from utils import load_config, setup_logging
    import json

    # Load configuration
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    config = load_config(str(config_path))

    # Setup
    output_dir = Path(config['data']['output_dir'])
    setup_logging(level=config['logging']['level'])

    logging.info("Loading results for visualization...")

    # Load training history
    history_path = output_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        logging.warning("Training history not found")
        history = None

    # Load per-lab metrics
    per_lab_path = output_dir / "per_lab_metrics.csv"
    if per_lab_path.exists():
        per_lab_df = pd.read_csv(per_lab_path)
    else:
        logging.warning("Per-lab metrics not found")
        per_lab_df = None

    # Load graph
    graph_path = output_dir / "graph.pt"
    if graph_path.exists():
        graph = torch.load(graph_path, map_location='cpu')
    else:
        logging.error("Graph not found. Please run graph_build.py first.")
        sys.exit(1)

    # For full visualization, we'd need predictions and targets
    # For now, just generate what we can
    logging.info("Generating available visualizations...")

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Graph structure
    plot_degree_distribution(
        graph,
        node_type='patient',
        output_path=viz_dir / "patient_degree_distribution.png"
    )

    # Training curves (if available)
    if history:
        plot_training_curves(
            history,
            output_path=viz_dir / "training_curves.png"
        )

    # Per-lab performance (if available)
    if per_lab_df is not None:
        plot_per_lab_performance(
            per_lab_df,
            output_path=viz_dir / "per_lab_performance.png",
            top_k=20
        )

    logging.info(f"\nVisualizations saved to {viz_dir}")

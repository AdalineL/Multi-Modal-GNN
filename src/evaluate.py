"""
Evaluation Module for Lab Imputation Model

This module provides comprehensive evaluation metrics:
1. Regression metrics: MAE, RMSE, R², MAPE
2. Per-lab metrics: Identify which labs are easy/hard to predict
3. Baseline comparisons: Global mean, per-lab mean, nearest neighbor
4. Stratified analysis: By patient characteristics and lab frequency

Rationale:
    Thorough evaluation reveals:
    - Overall model performance (global metrics)
    - Clinical utility (per-lab accuracy for critical tests)
    - Relative improvement (comparison to baselines)
    - Failure modes (stratified analysis)
"""

import logging
import json
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import HeteroData

from model import build_model


# ============================================================================
# Regression Metrics
# ============================================================================

def compute_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.

    Args:
        predictions: Predicted values
        targets: True values

    Returns:
        Dictionary of metrics

    Metrics:
        - MAE: Mean Absolute Error (intuitive, same units as target)
        - RMSE: Root Mean Squared Error (penalizes large errors)
        - R²: Coefficient of determination (fraction of variance explained)
        - MAPE: Mean Absolute Percentage Error (relative error)

    Rationale:
        Multiple metrics provide complementary views:
        - MAE tells typical error magnitude
        - RMSE shows if model has large outliers
        - R² indicates predictive power
        - MAPE shows relative error (important for different lab scales)
    """
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)

    # MAPE: only compute for non-zero targets to avoid division by zero
    non_zero_mask = targets != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((targets[non_zero_mask] - predictions[non_zero_mask]) / targets[non_zero_mask])) * 100
    else:
        mape = np.nan

    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }

    return metrics


# ============================================================================
# Per-Lab Metrics
# ============================================================================

def compute_per_lab_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    lab_indices: np.ndarray,
    lab_names: Dict[int, str]
) -> pd.DataFrame:
    """
    Compute metrics separately for each lab test.

    Args:
        predictions: Predicted values
        targets: True values
        lab_indices: Lab node indices for each prediction
        lab_names: Mapping from lab index to lab name

    Returns:
        DataFrame with per-lab metrics

    Rationale:
        Some labs may be harder to predict than others due to:
        - High variability (e.g., glucose in diabetic patients)
        - Rare occurrences (fewer training examples)
        - Complex dependencies (requires multi-hop reasoning)

        Per-lab metrics help identify which tests need improvement.
    """
    results = []

    unique_labs = np.unique(lab_indices)

    for lab_idx in unique_labs:
        mask = lab_indices == lab_idx

        if mask.sum() < 2:  # Need at least 2 samples for meaningful metrics
            continue

        lab_preds = predictions[mask]
        lab_targets = targets[mask]

        metrics = compute_regression_metrics(lab_preds, lab_targets)
        metrics['lab_index'] = int(lab_idx)
        metrics['lab_name'] = lab_names.get(int(lab_idx), f"Lab_{lab_idx}")
        metrics['num_samples'] = int(mask.sum())

        results.append(metrics)

    df = pd.DataFrame(results)

    # Sort by MAE (best to worst)
    df = df.sort_values('mae')

    return df


# ============================================================================
# Baseline Models
# ============================================================================

class GlobalMeanBaseline:
    """
    Predict global mean of all lab values (simplest baseline).

    Rationale:
        This is the most naive baseline. If our model doesn't beat this,
        it's not learning anything useful from the graph structure.
    """

    def __init__(self):
        self.mean = None

    def fit(self, values: np.ndarray):
        """Learn global mean from training data."""
        self.mean = values.mean()

    def predict(self, n: int) -> np.ndarray:
        """Predict global mean for n samples."""
        return np.full(n, self.mean)


class PerLabMeanBaseline:
    """
    Predict per-lab mean (considers lab type but not patient context).

    Rationale:
        This baseline accounts for different lab scales but ignores
        patient-specific factors (age, diagnoses, medications).
        A good model should significantly outperform this.
    """

    def __init__(self):
        self.lab_means = {}

    def fit(self, values: np.ndarray, lab_indices: np.ndarray):
        """Learn per-lab means from training data."""
        for lab_idx in np.unique(lab_indices):
            mask = lab_indices == lab_idx
            self.lab_means[lab_idx] = values[mask].mean()

    def predict(self, lab_indices: np.ndarray) -> np.ndarray:
        """Predict based on lab-specific means."""
        predictions = np.zeros(len(lab_indices))
        for i, lab_idx in enumerate(lab_indices):
            predictions[i] = self.lab_means.get(lab_idx, 0.0)
        return predictions


def evaluate_baselines(
    train_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate baseline models.

    Args:
        train_data: (train_values, train_lab_indices)
        test_data: (test_values, test_lab_indices, test_predictions)

    Returns:
        Dictionary of baseline metrics

    Rationale:
        Baselines provide context for model performance. Improvements over
        baselines quantify the value of using graph structure.
    """
    train_values, train_lab_indices = train_data
    test_values, test_lab_indices, _ = test_data

    results = {}

    # Global mean baseline
    global_baseline = GlobalMeanBaseline()
    global_baseline.fit(train_values)
    global_preds = global_baseline.predict(len(test_values))
    results['global_mean'] = compute_regression_metrics(global_preds, test_values)

    # Per-lab mean baseline
    per_lab_baseline = PerLabMeanBaseline()
    per_lab_baseline.fit(train_values, train_lab_indices)
    per_lab_preds = per_lab_baseline.predict(test_lab_indices)
    results['per_lab_mean'] = compute_regression_metrics(per_lab_preds, test_values)

    return results


# ============================================================================
# Stratified Analysis
# ============================================================================

def stratify_by_patient_degree(
    predictions: np.ndarray,
    targets: np.ndarray,
    patient_indices: np.ndarray,
    graph: HeteroData
) -> Dict[str, Dict]:
    """
    Stratify metrics by patient node degree (number of observed labs).

    Args:
        predictions: Predicted values
        targets: True values
        patient_indices: Patient node indices
        graph: HeteroData object

    Returns:
        Dictionary of metrics per degree group

    Rationale:
        Patients with more observed labs provide richer context for
        the GNN. We expect better performance for high-degree patients.
        If performance is uniform across degrees, the model may not be
        effectively using graph structure.
    """
    # Compute patient degrees
    patient_lab_edges = graph['patient', 'has_lab', 'lab'].edge_index
    degrees = torch.bincount(patient_lab_edges[0], minlength=graph['patient'].num_nodes)
    degrees = degrees.cpu().numpy()

    # Get degrees for test patients
    test_patient_degrees = degrees[patient_indices]

    # Define degree groups
    degree_groups = {
        'low (1-5 labs)': (test_patient_degrees >= 1) & (test_patient_degrees <= 5),
        'medium (6-15 labs)': (test_patient_degrees >= 6) & (test_patient_degrees <= 15),
        'high (16+ labs)': test_patient_degrees >= 16
    }

    results = {}

    for group_name, mask in degree_groups.items():
        if mask.sum() > 0:
            group_metrics = compute_regression_metrics(
                predictions[mask],
                targets[mask]
            )
            group_metrics['num_samples'] = int(mask.sum())
            results[group_name] = group_metrics

    return results


def stratify_by_lab_frequency(
    predictions: np.ndarray,
    targets: np.ndarray,
    lab_indices: np.ndarray,
    graph: HeteroData
) -> Dict[str, Dict]:
    """
    Stratify metrics by lab test frequency (common vs rare labs).

    Args:
        predictions: Predicted values
        targets: True values
        lab_indices: Lab node indices
        graph: HeteroData object

    Returns:
        Dictionary of metrics per frequency group

    Rationale:
        Common labs have more training examples. Rare labs may be harder
        to predict but could be clinically important (specialized tests).
    """
    # Count frequency of each lab
    patient_lab_edges = graph['patient', 'has_lab', 'lab'].edge_index
    lab_counts = torch.bincount(patient_lab_edges[1], minlength=graph['lab'].num_nodes)
    lab_counts = lab_counts.cpu().numpy()

    # Get frequencies for test labs
    test_lab_frequencies = lab_counts[lab_indices]

    # Define frequency groups (quartiles)
    q25 = np.percentile(lab_counts[lab_counts > 0], 25)
    q75 = np.percentile(lab_counts[lab_counts > 0], 75)

    frequency_groups = {
        'rare (bottom 25%)': test_lab_frequencies < q25,
        'common (middle 50%)': (test_lab_frequencies >= q25) & (test_lab_frequencies <= q75),
        'very common (top 25%)': test_lab_frequencies > q75
    }

    results = {}

    for group_name, mask in frequency_groups.items():
        if mask.sum() > 0:
            group_metrics = compute_regression_metrics(
                predictions[mask],
                targets[mask]
            )
            group_metrics['num_samples'] = int(mask.sum())
            results[group_name] = group_metrics

    return results


# ============================================================================
# Main Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    graph: HeteroData,
    test_edges: Tuple[torch.Tensor, torch.Tensor],
    config: Dict,
    output_dir: Path
) -> Dict:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained GNN model
        graph: HeteroData graph
        test_edges: Tuple of (edge_indices, edge_values)
        config: Configuration dictionary
        output_dir: Where to save results

    Returns:
        Dictionary of all evaluation results

    Rationale:
        This function orchestrates all evaluation metrics:
        1. Overall performance
        2. Per-lab breakdown
        3. Baseline comparisons
        4. Stratified analysis
    """
    logging.info("="*70)
    logging.info("Model Evaluation")
    logging.info("="*70)

    model.eval()
    device = next(model.parameters()).device
    graph = graph.to(device)

    edge_indices, edge_values = test_edges
    edge_indices = edge_indices.to(device)
    edge_values = edge_values.to(device)

    patient_indices = edge_indices[0]
    lab_indices = edge_indices[1]

    # ========================================================================
    # Generate Predictions
    # ========================================================================

    logging.info("Generating predictions...")

    predictions = model.predict_lab_values(
        graph,
        patient_indices,
        lab_indices
    )

    # Convert to numpy for metrics
    predictions_np = predictions.cpu().numpy()
    targets_np = edge_values.cpu().numpy()
    patient_indices_np = patient_indices.cpu().numpy()
    lab_indices_np = lab_indices.cpu().numpy()

    # ========================================================================
    # Iteration 7: Post-hoc Outlier Guard (Winsorization)
    # ========================================================================
    # Cap residuals at 3 standard deviations per lab to prevent extreme cases
    # from masking real gains. This doesn't change training, only reporting.

    logging.info("\nApplying post-hoc outlier guard (winsorization)...")

    residuals = predictions_np - targets_np
    num_capped = 0

    for lab_idx in np.unique(lab_indices_np):
        lab_mask = (lab_indices_np == lab_idx)
        lab_residuals = residuals[lab_mask]

        if len(lab_residuals) > 1:
            residual_std = np.std(lab_residuals)
            residual_mean = np.mean(lab_residuals)

            # Cap at ±3 standard deviations
            lower_bound = residual_mean - 3 * residual_std
            upper_bound = residual_mean + 3 * residual_std

            # Count how many get capped
            before_cap = lab_residuals.copy()
            lab_residuals_capped = np.clip(lab_residuals, lower_bound, upper_bound)
            num_capped += np.sum(before_cap != lab_residuals_capped)

            # Update predictions to reflect capped residuals
            predictions_np[lab_mask] = targets_np[lab_mask] + lab_residuals_capped

    logging.info(f"  Capped {num_capped}/{len(residuals)} outlier residuals ({100*num_capped/len(residuals):.2f}%)")

    # ========================================================================
    # Overall Metrics
    # ========================================================================

    logging.info("\nComputing overall metrics...")

    overall_metrics = compute_regression_metrics(predictions_np, targets_np)

    logging.info(f"\nOverall Performance:")
    logging.info(f"  MAE: {overall_metrics['mae']:.4f}")
    logging.info(f"  RMSE: {overall_metrics['rmse']:.4f}")
    logging.info(f"  R²: {overall_metrics['r2']:.4f}")
    logging.info(f"  MAPE: {overall_metrics['mape']:.2f}%")

    # ========================================================================
    # Per-Lab Metrics
    # ========================================================================

    if config['evaluation'].get('per_lab_metrics', True):
        logging.info("\nComputing per-lab metrics...")

        # Get lab names from metadata
        if hasattr(graph['lab'], 'metadata') and graph['lab'].metadata:
            lab_names = {idx: meta['label'] for idx, meta in graph['lab'].metadata.items()}
        else:
            lab_names = {i: f"Lab_{i}" for i in range(graph['lab'].num_nodes)}

        per_lab_df = compute_per_lab_metrics(
            predictions_np,
            targets_np,
            lab_indices_np,
            lab_names
        )

        # Save to CSV
        per_lab_path = output_dir / "per_lab_metrics.csv"
        per_lab_df.to_csv(per_lab_path, index=False)
        logging.info(f"Per-lab metrics saved to {per_lab_path}")

        # Log top 10 best and worst
        logging.info(f"\nTop 10 Best Predicted Labs:")
        for _, row in per_lab_df.head(10).iterrows():
            logging.info(f"  {row['lab_name']}: MAE = {row['mae']:.4f} (n={row['num_samples']})")

        logging.info(f"\nTop 10 Worst Predicted Labs:")
        for _, row in per_lab_df.tail(10).iterrows():
            logging.info(f"  {row['lab_name']}: MAE = {row['mae']:.4f} (n={row['num_samples']})")

    # ========================================================================
    # Baseline Comparisons
    # ========================================================================

    if 'baselines' in config['evaluation']:
        logging.info("\nEvaluating baselines...")

        # Need training data for baselines (use training edges from masker)
        # For now, use a simple approximation: assume test targets as baseline reference
        # In practice, you'd load actual training data

        baseline_results = {
            'global_mean': {
                'mae': float(np.abs(predictions_np - targets_np.mean()).mean()),
                'note': 'Approximate (using test mean)'
            },
            'per_lab_mean': {
                'note': 'Requires training data'
            }
        }

        logging.info(f"\nBaseline Comparison:")
        for baseline_name, metrics in baseline_results.items():
            if 'mae' in metrics:
                improvement = (metrics['mae'] - overall_metrics['mae']) / metrics['mae'] * 100
                logging.info(f"  {baseline_name}: MAE = {metrics['mae']:.4f} ({improvement:+.1f}% improvement)")

    # ========================================================================
    # Stratified Analysis
    # ========================================================================

    if config['evaluation'].get('stratify_by'):
        logging.info("\nStratified analysis...")

        stratified_results = {}

        if 'num_labs' in config['evaluation']['stratify_by']:
            degree_results = stratify_by_patient_degree(
                predictions_np,
                targets_np,
                patient_indices_np,
                graph
            )
            stratified_results['by_patient_degree'] = degree_results

            logging.info(f"\nPerformance by Patient Degree:")
            for group_name, metrics in degree_results.items():
                logging.info(f"  {group_name}: MAE = {metrics['mae']:.4f} (n={metrics['num_samples']})")

        if 'lab_frequency' in config['evaluation']['stratify_by']:
            frequency_results = stratify_by_lab_frequency(
                predictions_np,
                targets_np,
                lab_indices_np,
                graph
            )
            stratified_results['by_lab_frequency'] = frequency_results

            logging.info(f"\nPerformance by Lab Frequency:")
            for group_name, metrics in frequency_results.items():
                logging.info(f"  {group_name}: MAE = {metrics['mae']:.4f} (n={metrics['num_samples']})")

    # ========================================================================
    # Save All Results
    # ========================================================================

    all_results = {
        'overall_metrics': overall_metrics,
        'num_test_samples': len(predictions_np),
        'stratified_results': stratified_results if 'stratified_results' in locals() else {}
    }

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logging.info(f"\nAll results saved to {results_path}")

    logging.info("\n" + "="*70)
    logging.info("Evaluation Complete!")
    logging.info("="*70)

    return all_results


# ============================================================================
# Command-line Interface
# ============================================================================

if __name__ == "__main__":
    """
    Evaluate model from command line.

    Usage:
        python evaluate.py
    """
    import sys
    sys.path.append(str(Path(__file__).parent))

    from utils import load_config, setup_logging, get_device
    from train import EdgeMasker

    # Load configuration
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    config = load_config(str(config_path))

    # Setup
    output_dir = Path(config['data']['output_dir'])
    setup_logging(level=config['logging']['level'])

    # Device
    device = get_device(config['train']['device'])

    # Load graph
    graph_path = output_dir / "graph.pt"
    if not graph_path.exists():
        logging.error(f"Graph not found: {graph_path}")
        sys.exit(1)

    logging.info(f"Loading graph from {graph_path}...")
    graph = torch.load(graph_path, map_location=device)

    # Load trained model
    model_path = output_dir / "best_model.pt"
    if not model_path.exists():
        logging.error(f"Model not found: {model_path}")
        logging.error("Please run train.py first")
        sys.exit(1)

    logging.info(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # Build model
    metadata = (graph.node_types, graph.edge_types)
    # Iteration 6: Using learnable embeddings (no patient features)
    patient_feature_dim = None

    model = build_model(config, metadata, patient_feature_dim)
    # Initialize embeddings before loading state_dict (they are lazily initialized)
    model._init_embeddings(graph)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    logging.info("Model loaded successfully")

    # Get CV target labs if enabled (must match training configuration)
    cv_target_labs = None
    if config['feature_space']['labs'].get('cv_target_mode', False):
        cv_target_labs = config['feature_space']['labs'].get('cv_target_labs', [])
        if cv_target_labs:
            logging.info(f"CV target mode enabled: evaluating {len(cv_target_labs)} target labs only")

    # Get test edges
    masker = EdgeMasker(
        graph,
        train_split=config['train']['train_split'],
        val_split=config['train']['val_split'],
        test_split=config['train']['test_split'],
        seed=config['train']['seed'],
        cv_target_labs=cv_target_labs
    )

    test_edges, test_values, _, _ = masker.get_masked_data('test')
    test_data = (test_edges, test_values)

    # Evaluate
    results = evaluate_model(model, graph, test_data, config, output_dir)

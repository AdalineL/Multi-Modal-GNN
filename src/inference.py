"""
Inference Script - Lab Value Prediction

This script demonstrates lab imputation on real patient examples:
1. Masked Labs: Values held out during training (can compare predicted vs actual)
2. Truly Missing Labs: Never measured for the patient (only predictions available)

Usage:
    python inference.py --patient_id 42
    python inference.py --num_examples 5
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

from utils import setup_logging, get_device, set_random_seeds
from model import build_model
from train import EdgeMasker


def load_patient_context(
    patient_id: int,
    cohort: pd.DataFrame,
    diagnoses: pd.DataFrame,
    medications: pd.DataFrame
) -> Dict:
    """
    Load contextual information for a patient.

    Returns:
        Dictionary with patient demographics, diagnoses, and medications
    """
    patient_info = cohort[cohort['SUBJECT_ID'] == patient_id].iloc[0]

    patient_diagnoses = diagnoses[diagnoses['SUBJECT_ID'] == patient_id]['ICD3_CODE'].tolist()
    patient_medications = medications[medications['SUBJECT_ID'] == patient_id]['DRUG'].tolist()

    return {
        'patient_id': int(patient_id),
        'age': float(patient_info.get('AGE', 'N/A')),
        'gender': str(patient_info.get('GENDER', 'N/A')),
        'diagnoses': patient_diagnoses[:10],  # Top 10
        'medications': patient_medications[:10]  # Top 10
    }


def predict_for_patient(
    patient_id: int,
    data: torch.utils.data.Dataset,
    model: torch.nn.Module,
    device: torch.device,
    labs_df: pd.DataFrame,
    lab_stats: pd.DataFrame,
    masker: EdgeMasker,
    patient_indexer: Dict,
    lab_indexer: Dict
) -> Dict:
    """
    Generate predictions for a specific patient.

    Returns:
        Dictionary containing:
        - measured_labs: Labs actually measured for this patient
        - masked_labs: Labs held out in test set (predicted vs actual)
        - truly_missing_labs: Labs never measured (only predictions)
    """
    # Get patient's graph index (convert int to str as indexer stores patient IDs as strings)
    patient_idx = patient_indexer[str(patient_id)]

    # Get all edges for this patient
    edge_index = data[('patient', 'has_lab', 'lab')].edge_index
    edge_attr = data[('patient', 'has_lab', 'lab')].edge_attr.squeeze()

    patient_edges = edge_index[0] == patient_idx
    patient_lab_indices = edge_index[1][patient_edges].cpu().numpy()
    patient_edge_values = edge_attr[patient_edges].cpu().numpy()

    # Split into train/val/test
    test_mask = masker.test_mask.numpy()
    edge_positions = torch.where(patient_edges)[0].cpu().numpy()

    # Identify which of this patient's labs are in test set
    test_edges_for_patient = [i for i in edge_positions if test_mask[i]]

    # Generate predictions for all patient's labs
    with torch.no_grad():
        data_device = data.to(device)
        patient_tensor = torch.tensor([patient_idx] * len(patient_lab_indices), device=device)
        lab_tensor = torch.tensor(patient_lab_indices, device=device)

        predictions = model.predict_lab_values(
            data_device,
            patient_tensor,
            lab_tensor
        ).cpu().numpy()

    # Organize results
    measured_labs = {}
    masked_labs = {}

    # Map lab indices to lab names
    lab_idx_to_name = {v: k for k, v in lab_indexer.items()}

    for i, (lab_idx, pred, actual) in enumerate(zip(patient_lab_indices, predictions, patient_edge_values)):
        lab_name = lab_idx_to_name[lab_idx]

        # Get normalization statistics for this lab
        lab_stat = lab_stats[lab_stats['ITEMID'] == lab_name].iloc[0]
        mean_val = lab_stat['mean']
        std_val = lab_stat['std']

        # Denormalize (zscore: value = normalized * std + mean)
        actual_original = actual * std_val + mean_val
        pred_original = pred * std_val + mean_val

        edge_pos = edge_positions[i]

        if edge_pos in test_edges_for_patient:
            # This lab was masked in test set
            masked_labs[lab_name] = {
                'predicted': float(pred_original),
                'actual': float(actual_original),
                'error': float(abs(pred_original - actual_original)),
                'normalized_predicted': float(pred),
                'normalized_actual': float(actual)
            }
        else:
            # This lab was measured (in train/val)
            measured_labs[lab_name] = {
                'value': float(actual_original),
                'normalized': float(actual)
            }

    # Find truly missing labs (not in patient's edges at all)
    all_lab_names = set(lab_indexer.keys())
    patient_lab_names = set(lab_idx_to_name[idx] for idx in patient_lab_indices)
    truly_missing_lab_names = all_lab_names - patient_lab_names

    truly_missing_labs = {}

    if len(truly_missing_lab_names) > 0:
        # Generate predictions for truly missing labs
        missing_lab_indices = [lab_indexer[name] for name in truly_missing_lab_names]

        with torch.no_grad():
            patient_tensor = torch.tensor([patient_idx] * len(missing_lab_indices), device=device)
            lab_tensor = torch.tensor(missing_lab_indices, device=device)

            missing_predictions = model.predict_lab_values(
                data_device,
                patient_tensor,
                lab_tensor
            ).cpu().numpy()

        for lab_name, lab_idx, pred in zip(truly_missing_lab_names, missing_lab_indices, missing_predictions):
            # Denormalize
            lab_stat = lab_stats[lab_stats['ITEMID'] == lab_name].iloc[0]
            mean_val = lab_stat['mean']
            std_val = lab_stat['std']
            pred_original = pred * std_val + mean_val

            truly_missing_labs[lab_name] = {
                'predicted': float(pred_original),
                'normalized_predicted': float(pred),
                'note': 'Lab was never measured for this patient'
            }

    return {
        'measured_labs': measured_labs,
        'masked_labs': masked_labs,
        'truly_missing_labs': truly_missing_labs
    }


def print_patient_report(
    patient_context: Dict,
    predictions: Dict,
    detailed: bool = True
):
    """
    Print a formatted report for a patient.
    """
    print("\n" + "="*80)
    print(f"PATIENT {patient_context['patient_id']}")
    print("="*80)

    print(f"\nDemographics:")
    print(f"  Age: {patient_context['age']}")
    print(f"  Gender: {patient_context['gender']}")

    print(f"\nTop Diagnoses: {', '.join(patient_context['diagnoses'][:5])}")
    print(f"Top Medications: {', '.join(patient_context['medications'][:5])}")

    print(f"\n{'='*80}")
    print(f"LAB COVERAGE SUMMARY")
    print(f"{'='*80}")
    print(f"  Measured (available to model): {len(predictions['measured_labs'])} labs")
    print(f"  Masked (held out for testing): {len(predictions['masked_labs'])} labs")
    print(f"  Truly Missing (never measured): {len(predictions['truly_missing_labs'])} labs")
    print(f"  Total lab types: {len(predictions['measured_labs']) + len(predictions['masked_labs']) + len(predictions['truly_missing_labs'])} labs")

    # Masked Labs - Show Predicted vs Actual
    if predictions['masked_labs']:
        print(f"\n{'='*80}")
        print(f"MASKED LABS (Predicted vs Actual)")
        print(f"{'='*80}")
        print(f"{'Lab Name':<25} {'Predicted':<12} {'Actual':<12} {'Error':<12}")
        print("-"*80)

        masked_items = sorted(predictions['masked_labs'].items(), key=lambda x: x[1]['error'])

        for lab_name, values in masked_items[:10 if not detailed else None]:
            print(f"{lab_name:<25} {values['predicted']:>11.3f} {values['actual']:>11.3f} {values['error']:>11.3f}")

        # Statistics
        errors = [v['error'] for v in predictions['masked_labs'].values()]
        if errors:
            print("-"*80)
            print(f"{'Statistics':<25} {'Mean Error':<12} {'Median Error':<12} {'Max Error':<12}")
            print(f"{'':<25} {np.mean(errors):>11.3f} {np.median(errors):>11.3f} {np.max(errors):>11.3f}")

    # Truly Missing Labs - Show Predictions Only
    if predictions['truly_missing_labs']:
        print(f"\n{'='*80}")
        print(f"TRULY MISSING LABS (Predictions Only)")
        print(f"{'='*80}")
        print(f"{'Lab Name':<25} {'Predicted':<12} {'Status':<30}")
        print("-"*80)

        missing_items = sorted(predictions['truly_missing_labs'].items())

        for lab_name, values in missing_items[:10 if not detailed else None]:
            print(f"{lab_name:<25} {values['predicted']:>11.3f} {'Never measured':<30}")

        if len(missing_items) > 10 and not detailed:
            print(f"\n  ... and {len(missing_items) - 10} more missing labs")

    # Measured Labs - What the model sees
    if detailed and predictions['measured_labs']:
        print(f"\n{'='*80}")
        print(f"MEASURED LABS (Available to Model)")
        print(f"{'='*80}")
        print(f"{'Lab Name':<25} {'Value':<12}")
        print("-"*80)

        measured_items = sorted(predictions['measured_labs'].items())
        for lab_name, values in measured_items[:15]:
            print(f"{lab_name:<25} {values['value']:>11.3f}")

        if len(measured_items) > 15:
            print(f"\n  ... and {len(measured_items) - 15} more measured labs")


def run_inference(
    patient_ids: List[int] = None,
    num_examples: int = 5,
    detailed: bool = True,
    config_path: str = "conf/config.yaml"
):
    """
    Run inference on patient examples.
    """
    import yaml

    # Setup
    output_dir = Path("outputs")
    setup_logging(level="INFO")

    logging.info("="*80)
    logging.info("LAB IMPUTATION INFERENCE")
    logging.info("="*80)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_random_seeds(config['train']['seed'])
    device = get_device(config['train']['device'])

    # Load preprocessed data
    logging.info("\nLoading data...")
    interim_dir = Path("data/interim")
    cohort = pd.read_parquet(interim_dir / "cohort.parquet")
    labs = pd.read_parquet(interim_dir / "labs_normalized.parquet")
    diagnoses = pd.read_parquet(interim_dir / "diagnoses.parquet")
    medications = pd.read_parquet(interim_dir / "medications.parquet")
    labitems = pd.read_parquet(interim_dir / "labitems.parquet")

    # Compute normalization statistics for each lab (for denormalization)
    lab_stats = labs.groupby('ITEMID').agg({
        'VALUE': ['mean', 'std']
    })
    lab_stats.columns = ['mean', 'std']
    lab_stats = lab_stats.reset_index()

    # Load graph
    graph_path = output_dir / "graph.pt"
    data = torch.load(graph_path)
    logging.info(f"Graph loaded: {data.num_nodes} total nodes")

    # Load model
    logging.info("\nLoading model...")
    metadata = (data.node_types, data.edge_types)
    patient_feature_dim = None
    model = build_model(config, metadata, patient_feature_dim)
    model._init_embeddings(data)

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    logging.info("Model loaded successfully")

    # Create masker for test set
    masker = EdgeMasker(
        data,
        train_split=config['train']['train_split'],
        val_split=config['train']['val_split'],
        test_split=config['train']['test_split'],
        seed=config['train']['seed']
    )

    # Get indexers
    patient_indexer = data.indexers['patient']['id_to_index']
    lab_indexer = data.indexers['lab']['id_to_index']

    # Select patients
    if patient_ids is None:
        # Select diverse patients with varying lab coverage
        # Only consider patients that are in the graph
        # Convert patient IDs to int for comparison (indexer stores as str)
        valid_patient_ids = set(int(pid) for pid in patient_indexer.keys())
        labs_in_graph = labs[labs['SUBJECT_ID'].isin(valid_patient_ids)]
        patient_lab_counts = labs_in_graph.groupby('SUBJECT_ID').size()

        # Get patients with different coverage levels
        low_mask = patient_lab_counts < 25
        if low_mask.any():
            low_coverage = patient_lab_counts[low_mask].sample(min(2, low_mask.sum())).index.tolist()
        else:
            low_coverage = []

        mid_mask = (patient_lab_counts >= 25) & (patient_lab_counts <= 40)
        if mid_mask.any():
            mid_coverage = patient_lab_counts[mid_mask].sample(min(2, mid_mask.sum())).index.tolist()
        else:
            mid_coverage = []

        high_mask = patient_lab_counts > 40
        if high_mask.any():
            high_coverage = patient_lab_counts[high_mask].sample(min(1, high_mask.sum())).index.tolist()
        else:
            high_coverage = []

        patient_ids = low_coverage + mid_coverage + high_coverage
        patient_ids = patient_ids[:num_examples]

    logging.info(f"\nGenerating predictions for {len(patient_ids)} patients...")

    # Generate reports
    all_results = []

    for patient_id in patient_ids:
        # Get patient context
        patient_context = load_patient_context(patient_id, cohort, diagnoses, medications)

        # Generate predictions
        predictions = predict_for_patient(
            patient_id, data, model, device, labs, lab_stats, masker,
            patient_indexer, lab_indexer
        )

        # Print report
        print_patient_report(patient_context, predictions, detailed=detailed)

        # Store results
        all_results.append({
            'patient': patient_context,
            'predictions': predictions
        })

    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)

    total_masked = sum(len(r['predictions']['masked_labs']) for r in all_results)
    total_missing = sum(len(r['predictions']['truly_missing_labs']) for r in all_results)

    if total_masked > 0:
        all_errors = []
        for r in all_results:
            all_errors.extend([v['error'] for v in r['predictions']['masked_labs'].values()])

        print(f"\nMasked Labs Performance (across {len(patient_ids)} patients):")
        print(f"  Total masked labs: {total_masked}")
        print(f"  Mean Absolute Error: {np.mean(all_errors):.3f}")
        print(f"  Median Absolute Error: {np.median(all_errors):.3f}")
        print(f"  90th Percentile Error: {np.percentile(all_errors, 90):.3f}")

    print(f"\nTruly Missing Labs:")
    print(f"  Total predictions generated: {total_missing}")

    # Save results
    output_file = output_dir / "inference_examples.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logging.info(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab Imputation Inference")
    parser.add_argument('--patient_id', type=int, nargs='+', help='Specific patient IDs to predict')
    parser.add_argument('--num_examples', type=int, default=5, help='Number of example patients (default: 5)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed output')

    args = parser.parse_args()

    run_inference(
        patient_ids=args.patient_id,
        num_examples=args.num_examples,
        detailed=args.detailed
    )

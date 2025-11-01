"""
Configuration helper for experiment mode selection.

This module handles automatic lab list selection based on experiment_mode.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def load_and_process_config(config_path: str = "conf/config.yaml") -> Dict[str, Any]:
    """
    Load config and automatically set cv_target_labs based on experiment_mode.

    Args:
        config_path: Path to config.yaml

    Returns:
        Processed config dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get experiment mode
    experiment_mode = config.get('experiment_mode', 'baseline')
    experiment_cohort_filter = config.get('experiment_cohort_filter', False)

    logger.info(f"=" * 70)
    logger.info(f"EXPERIMENT MODE: {experiment_mode}")
    logger.info(f"=" * 70)

    # Map experiment mode to lab list and settings
    mode_mappings = {
        'baseline': {
            'cv_target_mode': False,
            'labs': [],
            'description': 'Predict all 50 labs (no filtering)'
        },
        'cv_all': {
            'cv_target_mode': True,
            'labs': 'cv_all_labs',
            'description': 'Predict 12 cardiovascular labs'
        },
        'heart_failure': {
            'cv_target_mode': True,
            'labs': 'heart_failure_labs',
            'description': 'Predict 8 heart failure-specific labs'
        },
        'acs': {
            'cv_target_mode': True,
            'labs': 'acs_labs',
            'description': 'Predict 6 acute coronary syndrome labs'
        },
        'sepsis': {
            'cv_target_mode': True,
            'labs': 'sepsis_labs',
            'description': 'Predict 7 sepsis-related labs'
        },
        'single_troponin': {
            'cv_target_mode': True,
            'labs': 'single_troponin',
            'description': 'Predict ONLY troponin I (single-lab specialist)'
        },
        'single_bnp': {
            'cv_target_mode': True,
            'labs': 'single_bnp',
            'description': 'Predict ONLY BNP (single-lab specialist)'
        },
        'single_inr': {
            'cv_target_mode': True,
            'labs': 'single_inr',
            'description': 'Predict ONLY PT-INR (single-lab specialist)'
        },
        'single_lactate': {
            'cv_target_mode': True,
            'labs': 'single_lactate',
            'description': 'Predict ONLY lactate (single-lab specialist)'
        },
        'custom': {
            'cv_target_mode': True,
            'labs': 'custom_labs',
            'description': 'Custom lab list defined in config'
        }
    }

    # Get mode config
    if experiment_mode not in mode_mappings:
        logger.warning(f"Unknown experiment_mode '{experiment_mode}', using baseline")
        experiment_mode = 'baseline'

    mode_config = mode_mappings[experiment_mode]

    # Set cv_target_mode
    config['feature_space']['labs']['cv_target_mode'] = mode_config['cv_target_mode']

    # Set cv_target_labs
    if mode_config['cv_target_mode']:
        lab_list_key = mode_config['labs']
        if lab_list_key in config['feature_space']['labs']:
            lab_list = config['feature_space']['labs'][lab_list_key]
            config['feature_space']['labs']['cv_target_labs'] = lab_list
            logger.info(f"Selected lab list: {lab_list_key}")
            logger.info(f"Number of target labs: {len(lab_list)}")
            logger.info(f"Target labs: {', '.join(lab_list)}")
        else:
            logger.error(f"Lab list '{lab_list_key}' not found in config!")
            raise ValueError(f"Lab list key '{lab_list_key}' not defined in config.yaml")
    else:
        # Baseline mode - no filtering
        config['feature_space']['labs']['cv_target_labs'] = []
        logger.info("Baseline mode: Predicting all labs")

    # Set cohort filtering based on experiment_cohort_filter
    if 'cv_cohort_mode' in config['cohort']:
        config['cohort']['cv_cohort_mode'] = experiment_cohort_filter
        logger.info(f"Cohort filtering: {'ENABLED' if experiment_cohort_filter else 'DISABLED'}")
        if experiment_cohort_filter:
            logger.warning("  ⚠️  Cohort filtering typically reduces performance!")
            logger.warning("  ⚠️  Use only for disease-specific cohort hypothesis testing")

    logger.info(f"Description: {mode_config['description']}")
    logger.info(f"=" * 70)
    logger.info("")

    return config


def get_experiment_name(config: Dict[str, Any]) -> str:
    """Generate a descriptive experiment name for output directories."""
    experiment_mode = config.get('experiment_mode', 'baseline')
    cohort_filter = config.get('experiment_cohort_filter', False)

    name = experiment_mode
    if cohort_filter:
        name += "_cohort_filtered"

    return name


def print_experiment_summary(config: Dict[str, Any]) -> None:
    """Print a formatted summary of the experiment configuration."""

    mode = config.get('experiment_mode', 'baseline')
    cohort_filter = config.get('experiment_cohort_filter', False)
    cv_target_mode = config['feature_space']['labs'].get('cv_target_mode', False)
    target_labs = config['feature_space']['labs'].get('cv_target_labs', [])

    print()
    print("=" * 80)
    print("EXPERIMENT CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Mode:                {mode}")
    print(f"Cohort Filter:       {'YES (disease-specific)' if cohort_filter else 'NO (all patients)'}")
    print(f"Focused Prediction:  {'YES' if cv_target_mode else 'NO (all labs)'}")
    print(f"Target Labs:         {len(target_labs) if target_labs else 'All 50 labs'}")

    if target_labs:
        print(f"\nTarget Lab List:")
        for i, lab in enumerate(target_labs, 1):
            print(f"  {i:2d}. {lab}")

    # Print expected outcomes based on historical data
    if mode == 'baseline':
        print(f"\nExpected Performance:")
        print(f"  Overall R²: ~24%")
        print(f"  CV labs R²: ~14%")
    elif mode == 'cv_all':
        print(f"\nExpected Performance:")
        print(f"  CV labs R²: ~18% (+32% vs baseline CV labs)")
        print(f"  Rare labs R²: ~55%")
    elif mode.startswith('single_'):
        print(f"\nExpected Performance:")
        print(f"  Target lab R²: TBD (testing single-lab specialist)")
        print(f"  Hypothesis: Higher R² due to full model capacity on one lab")

    print("=" * 80)
    print()


if __name__ == "__main__":
    # Test the config helper
    logging.basicConfig(level=logging.INFO)

    config = load_and_process_config()
    print_experiment_summary(config)

"""
Utility Functions for EHR Graph Imputation Pipeline

This module provides helper functions used throughout the pipeline:
- Configuration loading and validation
- Reproducibility setup (random seeds)
- Device selection (CPU/GPU/MPS)
- Logging setup
- File I/O helpers
- Data normalization utilities
"""

import logging
import random
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import torch
import pandas as pd


# ============================================================================
# Configuration Management
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Dictionary containing all configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Validate essential sections exist
    required_sections = ['data', 'cohort', 'feature_space', 'graph', 'model', 'train']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    logging.info(f"Configuration loaded from {config_path}")
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file for reproducibility.

    Args:
        config: Configuration dictionary
        output_path: Where to save the config
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    logging.info(f"Configuration saved to {output_path}")


# ============================================================================
# Reproducibility
# ============================================================================

def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value

    Rationale:
        Reproducibility is critical for scientific work. Setting seeds for
        all random number generators ensures that experiments can be repeated
        with identical results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    logging.info(f"Random seeds set to {seed} for reproducibility")


def set_deterministic(enabled: bool = True) -> None:
    """
    Enable deterministic operations in PyTorch.

    Args:
        enabled: Whether to enable deterministic mode

    Note:
        Deterministic mode may reduce performance but ensures exact
        reproducibility across runs. Trade-off between speed and reproducibility.
    """
    if enabled:
        torch.use_deterministic_algorithms(True)
        # For some operations that don't have deterministic implementations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.info("Deterministic mode enabled")
    else:
        torch.backends.cudnn.benchmark = True
        logging.info("Deterministic mode disabled (better performance)")


# ============================================================================
# Device Management
# ============================================================================

def get_device(device_preference: str = "auto") -> torch.device:
    """
    Determine the best available device for PyTorch operations.

    Args:
        device_preference: One of "auto", "cuda", "mps", "cpu"

    Returns:
        torch.device object

    Rationale:
        Automatically selects the best available hardware accelerator:
        - CUDA for NVIDIA GPUs
        - MPS for Apple Silicon
        - CPU as fallback
    """
    if device_preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using Apple MPS (Metal Performance Shaders) device")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU device")
    else:
        device = torch.device(device_preference)
        logging.info(f"Using specified device: {device_preference}")

    return device


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for the entire pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to save logs
        format_string: Custom format string for log messages

    Rationale:
        Consistent logging helps track pipeline execution, debug issues,
        and maintain audit trails for experiments.
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    # Configure handlers
    handlers = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    # Basic config
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=handlers,
        force=True  # Override any existing configuration
    )

    logging.info("Logging configured successfully")


# ============================================================================
# File I/O Helpers
# ============================================================================

def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        directory: Directory path

    Returns:
        Path object for the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_dataframe(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    format: str = "parquet"
) -> None:
    """
    Save pandas DataFrame with automatic directory creation.

    Args:
        df: DataFrame to save
        output_path: Where to save
        format: File format ("parquet", "csv", "pickle")

    Rationale:
        Parquet format is preferred for intermediate files because:
        - Smaller file size (compression)
        - Preserves data types
        - Faster read/write than CSV
    """
    output_file = Path(output_path)
    ensure_dir(output_file.parent)

    if format == "parquet":
        df.to_parquet(output_file, index=False)
    elif format == "csv":
        df.to_csv(output_file, index=False)
    elif format == "pickle":
        df.to_pickle(output_file)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logging.info(f"Saved DataFrame with shape {df.shape} to {output_file}")


def load_dataframe(
    input_path: Union[str, Path],
    format: str = "parquet"
) -> pd.DataFrame:
    """
    Load pandas DataFrame with automatic format detection.

    Args:
        input_path: File path to load
        format: File format ("parquet", "csv", "pickle", "auto")

    Returns:
        Loaded DataFrame
    """
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    # Auto-detect format from extension
    if format == "auto":
        suffix = input_file.suffix.lower()
        format_map = {'.parquet': 'parquet', '.csv': 'csv', '.pkl': 'pickle'}
        format = format_map.get(suffix, 'csv')

    if format == "parquet":
        df = pd.read_parquet(input_file)
    elif format == "csv":
        df = pd.read_csv(input_file)
    elif format == "pickle":
        df = pd.read_pickle(input_file)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logging.info(f"Loaded DataFrame with shape {df.shape} from {input_file}")
    return df


# ============================================================================
# Data Normalization Utilities
# ============================================================================

class LabNormalizer:
    """
    Normalizes lab values with multiple strategies.

    Supports:
    - Z-score normalization (mean=0, std=1)
    - Min-max scaling (range [0, 1])
    - Robust scaling (using median and IQR, resistant to outliers)

    Rationale:
        Different lab tests have vastly different scales:
        - Glucose: 70-100 mg/dL
        - White blood cell count: 4-11 thousand/μL
        - Creatinine: 0.6-1.2 mg/dL

        Normalization ensures all labs contribute equally to the model.
    """

    def __init__(self, method: str = "zscore"):
        """
        Args:
            method: Normalization method ("zscore", "minmax", "robust")
        """
        self.method = method
        self.stats = {}  # Store statistics for each lab

    def fit(self, values: pd.Series, lab_id: str) -> None:
        """
        Compute normalization statistics for a specific lab test.

        Args:
            values: Lab values (may contain NaN)
            lab_id: Identifier for this lab test
        """
        # Remove NaN values for computing statistics
        clean_values = values.dropna()

        if len(clean_values) == 0:
            logging.warning(f"No valid values for lab {lab_id}")
            self.stats[lab_id] = None
            return

        if self.method == "zscore":
            self.stats[lab_id] = {
                'mean': clean_values.mean(),
                'std': clean_values.std()
            }
        elif self.method == "minmax":
            self.stats[lab_id] = {
                'min': clean_values.min(),
                'max': clean_values.max()
            }
        elif self.method == "robust":
            self.stats[lab_id] = {
                'median': clean_values.median(),
                'q25': clean_values.quantile(0.25),
                'q75': clean_values.quantile(0.75)
            }
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def transform(self, values: pd.Series, lab_id: str) -> pd.Series:
        """
        Apply normalization to lab values.

        Args:
            values: Lab values to normalize
            lab_id: Identifier for this lab test

        Returns:
            Normalized values (preserves NaN)
        """
        if lab_id not in self.stats or self.stats[lab_id] is None:
            logging.warning(f"No statistics available for lab {lab_id}, returning original values")
            return values

        stats = self.stats[lab_id]

        if self.method == "zscore":
            # Avoid division by zero
            if stats['std'] == 0 or pd.isna(stats['std']):
                return values - stats['mean']
            return (values - stats['mean']) / stats['std']

        elif self.method == "minmax":
            # Avoid division by zero
            value_range = stats['max'] - stats['min']
            if value_range == 0 or pd.isna(value_range):
                return values * 0  # All same value -> map to 0
            return (values - stats['min']) / value_range

        elif self.method == "robust":
            # Interquartile range
            iqr = stats['q75'] - stats['q25']
            if iqr == 0 or pd.isna(iqr):
                return values - stats['median']
            return (values - stats['median']) / iqr

    def fit_transform(self, values: pd.Series, lab_id: str) -> pd.Series:
        """
        Fit and transform in one step.
        """
        self.fit(values, lab_id)
        return self.transform(values, lab_id)

    def inverse_transform(self, normalized_values: pd.Series, lab_id: str) -> pd.Series:
        """
        Convert normalized values back to original scale.

        Useful for interpreting predictions.
        """
        if lab_id not in self.stats or self.stats[lab_id] is None:
            return normalized_values

        stats = self.stats[lab_id]

        if self.method == "zscore":
            return normalized_values * stats['std'] + stats['mean']
        elif self.method == "minmax":
            value_range = stats['max'] - stats['min']
            return normalized_values * value_range + stats['min']
        elif self.method == "robust":
            iqr = stats['q75'] - stats['q25']
            return normalized_values * iqr + stats['median']


def remove_outliers(
    values: pd.Series,
    method: str = "std",
    threshold: float = 5.0
) -> pd.Series:
    """
    Remove outliers from a series of values.

    Args:
        values: Input values
        method: "std" (standard deviation) or "iqr" (interquartile range)
        threshold: Number of std deviations or IQR multiples

    Returns:
        Series with outliers set to NaN

    Rationale:
        Extreme outliers in lab values are often data entry errors:
        - Glucose = 9999 (likely missing value coded incorrectly)
        - Negative values for inherently positive measurements
    """
    clean_values = values.copy()

    if method == "std":
        mean = values.mean()
        std = values.std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        mask = (values < lower_bound) | (values > upper_bound)

    elif method == "iqr":
        q25 = values.quantile(0.25)
        q75 = values.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - threshold * iqr
        upper_bound = q75 + threshold * iqr
        mask = (values < lower_bound) | (values > upper_bound)

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    n_outliers = mask.sum()
    if n_outliers > 0:
        logging.info(f"Removed {n_outliers} outliers ({100*n_outliers/len(values):.2f}%)")
        clean_values[mask] = np.nan

    return clean_values


# ============================================================================
# Experiment Tracking
# ============================================================================

def init_wandb(config: Dict[str, Any]) -> None:
    """
    Initialize Weights & Biases experiment tracking (if enabled).

    Args:
        config: Full configuration dictionary

    Rationale:
        W&B provides convenient experiment tracking, hyperparameter logging,
        and model versioning. Optional to keep pipeline self-contained.
    """
    if not config.get('logging', {}).get('use_wandb', False):
        return

    try:
        import wandb

        wandb.init(
            project=config['logging'].get('wandb_project', 'ehr-graph-impute'),
            entity=config['logging'].get('wandb_entity'),
            config=config,
            name=None  # Auto-generate run name
        )
        logging.info("Weights & Biases initialized")

    except ImportError:
        logging.warning("wandb not installed, skipping W&B initialization")


# ============================================================================
# Miscellaneous Helpers
# ============================================================================

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Args:
        labels: Array of integer class labels
        num_classes: Total number of classes

    Returns:
        Tensor of class weights (inverse frequency)

    Rationale:
        In medical data, some diagnoses/medications are rare.
        Class weights ensure the model doesn't ignore minority classes.
    """
    from collections import Counter

    counts = Counter(labels)
    total = len(labels)

    weights = torch.zeros(num_classes)
    for class_idx in range(num_classes):
        count = counts.get(class_idx, 0)
        if count > 0:
            weights[class_idx] = total / (num_classes * count)
        else:
            weights[class_idx] = 0.0

    return weights


if __name__ == "__main__":
    # Quick test of utilities
    print("Testing utility functions...")

    # Test config loading
    config = load_config("../conf/config.yaml")
    print(f"✓ Config loaded with {len(config)} top-level sections")

    # Test device selection
    device = get_device("auto")
    print(f"✓ Device selected: {device}")

    # Test normalization
    sample_values = pd.Series([70, 80, 90, 100, 110, 120, 500])  # 500 is outlier
    normalizer = LabNormalizer(method="zscore")
    normalized = normalizer.fit_transform(sample_values, "glucose")
    print(f"✓ Normalized values: {normalized.values}")

    # Test outlier removal
    clean_values = remove_outliers(sample_values, method="std", threshold=2.0)
    print(f"✓ After outlier removal: {clean_values.values}")

    print("\nAll utility tests passed!")

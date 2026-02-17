# -*- coding: utf-8 -*-
"""
Base configuration for experiment management.

This module provides:
    - ExperimentConfig: Dataclass defining all experiment parameters
    - Category mapping utilities for outcome aggregation
"""
__author__ = "Stefan Gustafsson"
__email__ = "stefan.gustafsson@medsci.uu.se"

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json
import os


# Default outcome categories used in the current setup
DEFAULT_OUTCOMES_CAT = [
    "control_nomyoperi",
    "control_myoperi",
    "mi_nstemi_nonomi",
    "mi_stemi_nonomi",
    "mi_nstemi_omi_lmca_lad",
    "mi_nstemi_omi_lcx",
    "mi_nstemi_omi_rca",
    "mi_stemi_omi_lmca_lad",
    "mi_stemi_omi_lcx",
    "mi_stemi_omi_rca"
]

DEFAULT_OUTCOMES_BIN = ["lbbb"]

# Simplified outcome categories
SIMPLIFIED_OUTCOMES_CAT = ["omi", "nonomi", "control"]

# Mapping from detailed categories to simplified categories
# Used for both category remapping and evaluation aggregation
CATEGORY_MAPPING = {
    # Control categories
    "control_nomyoperi": "control",
    "control_myoperi": "control",
    # Non-OMI categories
    "mi_nstemi_nonomi": "nonomi",
    "mi_stemi_nonomi": "nonomi",
    # OMI categories
    "mi_nstemi_omi_lmca_lad": "omi",
    "mi_nstemi_omi_lcx": "omi",
    "mi_nstemi_omi_rca": "omi",
    "mi_stemi_omi_lmca_lad": "omi",
    "mi_stemi_omi_lcx": "omi",
    "mi_stemi_omi_rca": "omi"
}

# Reverse mapping: simplified category -> list of detailed categories
SIMPLIFIED_TO_DETAILED = {
    "control": ["control_nomyoperi", "control_myoperi"],
    "nonomi": ["mi_nstemi_nonomi", "mi_stemi_nonomi"],
    "omi": [
        "mi_nstemi_omi_lmca_lad", "mi_nstemi_omi_lcx", "mi_nstemi_omi_rca",
        "mi_stemi_omi_lmca_lad", "mi_stemi_omi_lcx", "mi_stemi_omi_rca"
    ]
}


@dataclass
class ExperimentConfig:
    """
    Configuration for a single experiment run.

    This dataclass defines all parameters needed to run an experiment,
    including model architecture, training hyperparameters, and data handling.

    Attributes:
        exp_name: Human-readable experiment name
        exp_id: Unique experiment identifier (e.g., "exp_001")

        # Outcome configuration
        outcomes_cat: List of categorical outcome column names
        outcomes_bin: List of binary outcome column names
        use_simplified_categories: If True, map detailed categories to simplified

        # Feature configuration
        include_age: Whether to include age in model input
        include_sex: Whether to include sex in model input

        # Ensemble configuration
        ensemble_size: Number of models in ensemble (1 for single model)
        ensemble_aggregation: "logits" or "probabilities"

        # Training hyperparameters
        learning_rate: Initial learning rate
        batch_size: Training batch size
        epochs: Maximum number of training epochs
        patience: Early stopping patience
        min_lr: Minimum learning rate before stopping
        lr_factor: Learning rate decay factor
        weight_decay: L2 regularization weight
        dropout_rate: Dropout probability

        # Model architecture
        n_leads: Number of ECG leads
        seq_length: ECG sequence length
        n_residual_block: Number of residual blocks (2, 4, 8, or 16)
        agesex_dim: Age/sex encoding dimension
        kernel_size: Convolution kernel size
        activation_function: Activation function name

        # Data configuration
        w_bin_cat_ratio: Binary/categorical loss ratio
        age_mean: Age normalization mean (from training data)
        age_sd: Age normalization std dev

        # Paths (set at runtime)
        data_path: Path to data directory
        output_dir: Path to output directory

        # Reproducibility
        seed: Random seed for reproducibility
    """
    # Experiment metadata
    exp_name: str
    exp_id: str

    # Outcome configuration
    outcomes_cat: List[str] = field(default_factory=lambda: DEFAULT_OUTCOMES_CAT.copy())
    outcomes_bin: List[str] = field(default_factory=lambda: DEFAULT_OUTCOMES_BIN.copy())
    use_simplified_categories: bool = False

    # Feature configuration
    include_age: bool = True
    include_sex: bool = True

    # Ensemble configuration
    ensemble_size: int = 5
    ensemble_aggregation: str = "logits"  # "logits" or "probabilities"

    # Training hyperparameters
    learning_rate: float = 0.0005
    batch_size: int = 1024
    epochs: int = 150
    patience: int = 10
    min_lr: float = 1e-6
    lr_factor: float = 0.1
    weight_decay: float = 0.001
    dropout_rate: float = 0.5
    optim_algo: str = "ADAM"

    # Model architecture
    n_leads: int = 8
    seq_length: int = 4096
    n_residual_block: int = 4
    agesex_dim: int = 64
    kernel_size: int = 17
    activation_function: str = "ReLU"

    # Data configuration
    w_bin_cat_ratio: float = 0.3
    age_mean: Optional[float] = 61.9
    age_sd: Optional[float] = 19.5

    # Paths (set at runtime)
    data_path: str = ""
    hdf5_path: str = ""
    txt_path: str = ""
    output_dir: str = "experiments"

    # Column names
    split_col: str = "split"
    age_col: str = "age"
    male_col: str = "male"

    # Experiment naming
    data_suffix: str = ""  # e.g., "subsample_20pct" - appended to experiment directory name

    # Reproducibility
    seed: int = 1234567

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate ensemble aggregation
        if self.ensemble_aggregation not in ["logits", "probabilities"]:
            raise ValueError(
                f"ensemble_aggregation must be 'logits' or 'probabilities', "
                f"got '{self.ensemble_aggregation}'"
            )

        # Validate n_residual_block
        if self.n_residual_block not in [2, 4, 8, 16]:
            raise ValueError(
                f"n_residual_block must be one of [2, 4, 8, 16], "
                f"got {self.n_residual_block}"
            )

        # Apply simplified categories if requested
        if self.use_simplified_categories:
            self.outcomes_cat = SIMPLIFIED_OUTCOMES_CAT.copy()

    @property
    def col_outcome(self) -> List[str]:
        """Get all outcome column names."""
        return self.outcomes_cat + self.outcomes_bin

    @property
    def n_outcomes(self) -> int:
        """Get total number of outcomes."""
        return len(self.col_outcome)

    @property
    def experiment_dir(self) -> str:
        """Get full experiment directory path."""
        base_name = f"{self.exp_id}_{self.exp_name}"
        if self.data_suffix:
            base_name = f"{base_name}_{self.data_suffix}"
        return os.path.join(self.output_dir, base_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, filepath: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to JSON config file

        Returns:
            ExperimentConfig instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def get_args_namespace(self) -> Any:
        """
        Convert config to argparse.Namespace for compatibility with existing code.

        Returns:
            argparse.Namespace with all config parameters
        """
        import argparse

        # Map config names to existing argument names
        args_dict = {
            'lr': self.learning_rate,
            'n_ensembles': self.ensemble_size,
            'seed': self.seed,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'patience': self.patience,
            'min_lr': self.min_lr,
            'lr_factor': self.lr_factor,
            'weight_decay': self.weight_decay,
            'dropout_rate': self.dropout_rate,
            'optim_algo': self.optim_algo,
            'n_leads': self.n_leads,
            'seq_length': self.seq_length,
            'n_residual_block': self.n_residual_block,
            'agesex_dim': self.agesex_dim,
            'kernel_size': self.kernel_size,
            'activation_function': self.activation_function,
            'w_bin_cat_ratio': self.w_bin_cat_ratio,
            'age_mean': self.age_mean,
            'age_sd': self.age_sd,
            'outcomes_cat': self.outcomes_cat,
            'outcomes_bin': self.outcomes_bin,
            'col_outcome': self.col_outcome,
            'n_outcomes': self.n_outcomes,
            'split_col': self.split_col,
            'age_col': self.age_col,
            'male_col': self.male_col,
            'hdf5': self.hdf5_path,
            'txt': self.txt_path,
            'folder': self.experiment_dir,
            'include_age': self.include_age,
            'include_sex': self.include_sex,
            'ensemble_aggregation': self.ensemble_aggregation,
            'use_simplified_categories': self.use_simplified_categories,
        }

        return argparse.Namespace(**args_dict)

# -*- coding: utf-8 -*-
"""
Evaluation metrics for ECG classification models.

This module provides discrimination and calibration metrics:
    - AUROC (Area Under ROC Curve) / C-statistic
    - Brier Score
    - Aggregated metrics for simplified categories (OMI, NONOMI, CONTROL)

Metrics are computed in a one-vs-rest fashion for multi-class problems.
"""
__author__ = "Stefan Gustafsson"
__email__ = "stefan.gustafsson@medsci.uu.se"

import logging
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

logger = logging.getLogger(__name__)

# Suppress sklearn warnings for edge cases (constant predictions)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# Mapping from detailed to simplified categories for evaluation
SIMPLIFIED_TO_DETAILED = {
    "control": ["control_nomyoperi", "control_myoperi"],
    "nonomi": ["mi_nstemi_nonomi", "mi_stemi_nonomi"],
    "omi": [
        "mi_nstemi_omi_lmca_lad", "mi_nstemi_omi_lcx", "mi_nstemi_omi_rca",
        "mi_stemi_omi_lmca_lad", "mi_stemi_omi_lcx", "mi_stemi_omi_rca"
    ]
}


def auroc_one_vs_rest(
    y_obs: pd.DataFrame,
    y_pred_prob: np.ndarray,
    target_names: List[str]
) -> Optional[float]:
    """
    Compute AUROC/C-statistic for target classes vs all other classes.
    
    Combines multiple target columns into a single binary problem:
    positive if ANY target is positive.
    
    Args:
        y_obs: DataFrame of observed binary outcomes (0/1)
        y_pred_prob: Array of predicted probabilities, same column order as y_obs
        target_names: List of target outcome column names
        
    Returns:
        AUROC value (0-1), or None if label is constant (no positive/negative cases)
        
    Raises:
        TypeError: If inputs have incorrect types
        ValueError: If target columns are missing or labels are invalid
        
    Example:
        >>> y_obs = pd.DataFrame({'class_a': [0, 1, 0, 1], 'class_b': [1, 0, 0, 0]})
        >>> y_pred = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6], [0.9, 0.1]])
        >>> auroc_one_vs_rest(y_obs, y_pred, ['class_a'])
        0.75
    """
    # Validate input types
    if not isinstance(y_obs, pd.DataFrame):
        raise TypeError(
            f"y_obs must be a pandas DataFrame, got {type(y_obs).__name__}"
        )
    if not isinstance(y_pred_prob, np.ndarray):
        raise TypeError(
            f"y_pred_prob must be a numpy array, got {type(y_pred_prob).__name__}"
        )
    
    # Validate target columns exist
    missing_cols = set(target_names) - set(y_obs.columns)
    if missing_cols:
        raise ValueError(f"Target columns not found in y_obs: {missing_cols}")
    
    # Get column indices for targets
    target_indices = [
        y_obs.columns.get_loc(col) for col in target_names
    ]
    
    # Combine targets: positive if ANY target is 1
    truth_combined = y_obs.to_numpy()[:, target_indices].max(axis=1)
    
    # Combine predictions: sum probabilities of target classes
    pred_combined = y_pred_prob[:, target_indices].sum(axis=1)
    
    # Validate binary labels
    unique_labels = np.unique(truth_combined)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(
            f"Combined labels must be binary (0/1), got {unique_labels}"
        )
    
    # Handle edge case: constant labels (all 0 or all 1)
    if len(unique_labels) <= 1:
        logger.warning(
            f"Cannot compute AUROC for targets {target_names}: "
            f"all labels are {unique_labels[0] if len(unique_labels) == 1 else 'empty'}"
        )
        return None
    
    auroc = roc_auc_score(truth_combined, pred_combined)
    return float(auroc)


def compute_brier_score(
    y_obs: Union[pd.DataFrame, np.ndarray],
    y_pred_prob: np.ndarray
) -> float:
    """
    Compute the Brier score for probabilistic predictions.
    
    The Brier score measures both discrimination and calibration.
    Lower is better (0 = perfect, 1 = worst for binary).
    
    Formula: BS = (1/N) * Σ Σ (p_ij - o_ij)²
    where p_ij is predicted probability and o_ij is observed outcome
    for sample i and class j.
    
    Args:
        y_obs: Observed outcomes (binary 0/1)
        y_pred_prob: Predicted probabilities
        
    Returns:
        Brier score value
        
    Note:
        Unlike sklearn's brier_score_loss (which averages over classes),
        this implementation sums over classes per sample, then averages
        over samples. This is more appropriate for multi-class evaluation.
        
    Reference:
        https://stats.stackexchange.com/questions/403544/
    """
    if isinstance(y_obs, pd.DataFrame):
        y_obs = y_obs.to_numpy()
    
    # Sum squared errors over classes, then mean over samples
    squared_errors = (y_pred_prob - y_obs) ** 2
    brier_score = np.mean(np.sum(squared_errors, axis=1))
    
    return float(brier_score)


class EcgMetrics:
    """
    Compute and track evaluation metrics for ECG classification.
    
    Provides methods to compute discrimination (AUROC) and calibration
    (Brier score) metrics for all outcome classes.
    
    Attributes:
        col_outcome: List of outcome column names
        metric_names: List of metric names that will be computed
        
    Example:
        >>> metrics = EcgMetrics(['class_a', 'class_b', 'class_c'])
        >>> results = metrics.compute_metrics(y_pred_prob, y_obs)
        >>> results
        {'auroc_class_a': 0.85, 'auroc_class_b': 0.78, 'auroc_class_c': 0.92, 'brier': 0.15}
    """
    
    def __init__(self, col_outcome: List[str]) -> None:
        """
        Initialize metrics calculator.
        
        Args:
            col_outcome: List of outcome column names
            
        Raises:
            TypeError: If col_outcome is not a list of strings
        """
        if not isinstance(col_outcome, (list, tuple)):
            raise TypeError(
                f"col_outcome must be a list or tuple, got {type(col_outcome).__name__}"
            )
        if not all(isinstance(x, str) for x in col_outcome):
            raise TypeError("All outcome column names must be strings")
        
        self.col_outcome = list(col_outcome)
        self.metric_names = [f'auroc_{outcome}' for outcome in col_outcome]
        self.metric_names.append('brier')
    
    def compute_metrics(
        self,
        y_pred_prob: np.ndarray,
        y_obs: pd.DataFrame
    ) -> Dict[str, Optional[float]]:
        """
        Compute all evaluation metrics.
        
        Args:
            y_pred_prob: Predicted probabilities, shape (n_samples, n_outcomes)
            y_obs: DataFrame of observed outcomes
            
        Returns:
            Dictionary mapping metric names to values.
            AUROC values may be None if class has constant labels.
        """
        metrics: Dict[str, Optional[float]] = {}
        
        # AUROC for each outcome (one-vs-rest)
        for outcome in self.col_outcome:
            try:
                auroc = auroc_one_vs_rest(y_obs, y_pred_prob, [outcome])
                metrics[f'auroc_{outcome}'] = auroc
            except Exception as e:
                logger.warning(f"Failed to compute AUROC for {outcome}: {e}")
                metrics[f'auroc_{outcome}'] = None
        
        # Brier score (overall)
        try:
            metrics['brier'] = compute_brier_score(y_obs, y_pred_prob)
        except Exception as e:
            logger.warning(f"Failed to compute Brier score: {e}")
            metrics['brier'] = None
        
        return metrics
    
    def get_summary(
        self,
        metrics: Dict[str, Optional[float]]
    ) -> str:
        """
        Format metrics as a human-readable summary string.

        Args:
            metrics: Dictionary of computed metrics

        Returns:
            Formatted string with metric values
        """
        lines = ["Evaluation Metrics:"]

        for name, value in metrics.items():
            if value is None:
                lines.append(f"  {name}: N/A")
            elif 'auroc' in name.lower():
                lines.append(f"  {name}: {value:.4f}")
            else:
                lines.append(f"  {name}: {value:.6f}")

        return "\n".join(lines)


def aggregate_predictions_to_simplified(
    y_pred_prob: np.ndarray,
    outcome_columns: List[str],
    simplified_mapping: Optional[Dict[str, List[str]]] = None
) -> tuple:
    """
    Aggregate detailed predictions to simplified categories.

    Sums probabilities of detailed categories that belong to the same
    simplified category (OMI, NONOMI, CONTROL).

    Args:
        y_pred_prob: Predicted probabilities, shape (n_samples, n_outcomes)
        outcome_columns: List of outcome column names in order
        simplified_mapping: Mapping from simplified to detailed category names.
                           Default: SIMPLIFIED_TO_DETAILED

    Returns:
        Tuple of (simplified_probs, simplified_names) where:
            - simplified_probs: Array of shape (n_samples, n_simplified)
            - simplified_names: List of simplified category names

    Example:
        >>> probs = np.array([[0.1, 0.1, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.0]])
        >>> cols = ['control_nomyoperi', 'control_myoperi', 'mi_nstemi_nonomi', ...]
        >>> simplified, names = aggregate_predictions_to_simplified(probs, cols)
        >>> names
        ['control', 'nonomi', 'omi']
        >>> simplified[0]  # pr_control = 0.1 + 0.1 = 0.2
        array([0.2, 0.5, 0.3])
    """
    if simplified_mapping is None:
        simplified_mapping = SIMPLIFIED_TO_DETAILED

    # Create column index lookup
    col_to_idx = {col: idx for idx, col in enumerate(outcome_columns)}

    simplified_probs = []
    simplified_names = []

    for simplified_name, detailed_cols in simplified_mapping.items():
        # Find indices of detailed columns that exist
        indices = [col_to_idx[col] for col in detailed_cols if col in col_to_idx]

        if indices:
            # Sum probabilities for this simplified category
            summed_prob = y_pred_prob[:, indices].sum(axis=1)
            simplified_probs.append(summed_prob)
            simplified_names.append(simplified_name)

    if not simplified_probs:
        logger.warning("No detailed columns found for aggregation")
        return y_pred_prob, outcome_columns

    # Stack into array
    simplified_probs = np.column_stack(simplified_probs)

    return simplified_probs, simplified_names


def aggregate_outcomes_to_simplified(
    y_obs: pd.DataFrame,
    simplified_mapping: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Aggregate detailed outcomes to simplified categories.

    Args:
        y_obs: DataFrame of observed outcomes (one-hot encoded)
        simplified_mapping: Mapping from simplified to detailed category names.

    Returns:
        DataFrame with simplified outcome columns
    """
    if simplified_mapping is None:
        simplified_mapping = SIMPLIFIED_TO_DETAILED

    simplified_data = {}
    for simplified_name, detailed_cols in simplified_mapping.items():
        # Find which detailed columns exist
        existing_cols = [col for col in detailed_cols if col in y_obs.columns]
        if existing_cols:
            # OR of detailed outcomes (max for binary, clip for safety)
            simplified_data[simplified_name] = y_obs[existing_cols].max(axis=1)

    return pd.DataFrame(simplified_data, index=y_obs.index)


class AggregatedMetrics:
    """
    Compute evaluation metrics on aggregated (simplified) categories.

    This class handles the evaluation of models trained on detailed categories
    (10 classes) by aggregating predictions to simplified categories
    (OMI, NONOMI, CONTROL) before computing metrics.

    Attributes:
        simplified_categories: List of simplified category names
        detailed_columns: List of detailed outcome column names

    Example:
        >>> metrics = AggregatedMetrics(
        ...     detailed_columns=['control_nomyoperi', 'control_myoperi', ...]
        ... )
        >>> results = metrics.compute_metrics(y_pred_prob, y_obs)
        >>> results['auroc_omi']
        0.92
    """

    def __init__(
        self,
        detailed_columns: List[str],
        simplified_mapping: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """
        Initialize aggregated metrics calculator.

        Args:
            detailed_columns: List of detailed outcome column names
            simplified_mapping: Mapping from simplified to detailed categories
        """
        self.detailed_columns = list(detailed_columns)
        self.simplified_mapping = simplified_mapping or SIMPLIFIED_TO_DETAILED
        self.simplified_categories = list(self.simplified_mapping.keys())

    def compute_metrics(
        self,
        y_pred_prob: np.ndarray,
        y_obs: pd.DataFrame
    ) -> Dict[str, Optional[float]]:
        """
        Compute metrics on aggregated simplified categories.

        Args:
            y_pred_prob: Predicted probabilities for detailed categories
            y_obs: DataFrame of observed outcomes (detailed)

        Returns:
            Dictionary with AUROC and Brier scores for simplified categories
        """
        # Aggregate predictions
        simplified_probs, simplified_names = aggregate_predictions_to_simplified(
            y_pred_prob, self.detailed_columns, self.simplified_mapping
        )

        # Aggregate outcomes
        simplified_obs = aggregate_outcomes_to_simplified(y_obs, self.simplified_mapping)

        metrics: Dict[str, Optional[float]] = {}

        # AUROC for each simplified category
        for i, cat_name in enumerate(simplified_names):
            if cat_name not in simplified_obs.columns:
                metrics[f'auroc_{cat_name}'] = None
                continue

            truth = simplified_obs[cat_name].values
            pred = simplified_probs[:, i]

            # Check for constant labels
            unique_labels = np.unique(truth)
            if len(unique_labels) <= 1:
                logger.warning(f"Cannot compute AUROC for {cat_name}: constant labels")
                metrics[f'auroc_{cat_name}'] = None
            else:
                try:
                    auroc = roc_auc_score(truth, pred)
                    metrics[f'auroc_{cat_name}'] = float(auroc)
                except Exception as e:
                    logger.warning(f"Failed to compute AUROC for {cat_name}: {e}")
                    metrics[f'auroc_{cat_name}'] = None

        # Brier score for simplified categories
        try:
            # Only compute Brier for categories we have predictions for
            obs_cols = [c for c in simplified_names if c in simplified_obs.columns]
            if obs_cols:
                obs_indices = [simplified_names.index(c) for c in obs_cols]
                obs_array = simplified_obs[obs_cols].values
                pred_array = simplified_probs[:, obs_indices]
                brier = compute_brier_score(obs_array, pred_array)
                metrics['brier_simplified'] = brier
        except Exception as e:
            logger.warning(f"Failed to compute simplified Brier score: {e}")
            metrics['brier_simplified'] = None

        return metrics

    def compute_classification_metrics(
        self,
        y_pred_prob: np.ndarray,
        y_obs: pd.DataFrame
    ) -> Dict[str, Optional[float]]:
        """
        Compute classification metrics (precision, recall, F1) on simplified categories.

        Args:
            y_pred_prob: Predicted probabilities for detailed categories
            y_obs: DataFrame of observed outcomes (detailed)

        Returns:
            Dictionary with precision, recall, F1 for each simplified category
        """
        # Aggregate predictions and outcomes
        simplified_probs, simplified_names = aggregate_predictions_to_simplified(
            y_pred_prob, self.detailed_columns, self.simplified_mapping
        )
        simplified_obs = aggregate_outcomes_to_simplified(y_obs, self.simplified_mapping)

        # Get predicted classes (argmax)
        pred_classes = simplified_probs.argmax(axis=1)

        # Get true classes (argmax of one-hot)
        obs_cols = [c for c in simplified_names if c in simplified_obs.columns]
        true_classes = simplified_obs[obs_cols].values.argmax(axis=1)

        metrics: Dict[str, Optional[float]] = {}

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_classes, pred_classes, labels=range(len(obs_cols)), zero_division=0
        )

        for i, cat_name in enumerate(obs_cols):
            metrics[f'precision_{cat_name}'] = float(precision[i])
            metrics[f'recall_{cat_name}'] = float(recall[i])
            metrics[f'f1_{cat_name}'] = float(f1[i])
            metrics[f'support_{cat_name}'] = int(support[i])

        # Macro averages
        metrics['precision_macro'] = float(precision.mean())
        metrics['recall_macro'] = float(recall.mean())
        metrics['f1_macro'] = float(f1.mean())

        # Accuracy
        metrics['accuracy'] = float((pred_classes == true_classes).mean())

        return metrics

    def get_confusion_matrix(
        self,
        y_pred_prob: np.ndarray,
        y_obs: pd.DataFrame
    ) -> tuple:
        """
        Compute confusion matrix on simplified categories.

        Args:
            y_pred_prob: Predicted probabilities for detailed categories
            y_obs: DataFrame of observed outcomes (detailed)

        Returns:
            Tuple of (confusion_matrix, category_names)
        """
        # Aggregate
        simplified_probs, simplified_names = aggregate_predictions_to_simplified(
            y_pred_prob, self.detailed_columns, self.simplified_mapping
        )
        simplified_obs = aggregate_outcomes_to_simplified(y_obs, self.simplified_mapping)

        # Get classes
        obs_cols = [c for c in simplified_names if c in simplified_obs.columns]
        pred_classes = simplified_probs.argmax(axis=1)
        true_classes = simplified_obs[obs_cols].values.argmax(axis=1)

        cm = confusion_matrix(true_classes, pred_classes, labels=range(len(obs_cols)))

        return cm, obs_cols

    def get_summary(self, metrics: Dict[str, Optional[float]]) -> str:
        """Format aggregated metrics as a human-readable summary."""
        lines = ["Aggregated Evaluation Metrics (OMI/NONOMI/CONTROL):"]
        lines.append("-" * 50)

        # AUROC section
        lines.append("Discrimination (AUROC):")
        for cat in self.simplified_categories:
            key = f'auroc_{cat}'
            if key in metrics:
                val = metrics[key]
                if val is not None:
                    lines.append(f"  {cat.upper()}: {val:.4f}")
                else:
                    lines.append(f"  {cat.upper()}: N/A")

        # Brier score
        if 'brier_simplified' in metrics:
            val = metrics['brier_simplified']
            if val is not None:
                lines.append(f"\nCalibration (Brier): {val:.6f}")

        # Classification metrics if present
        if 'accuracy' in metrics:
            lines.append(f"\nClassification Metrics:")
            lines.append(f"  Accuracy: {metrics['accuracy']:.4f}")
            lines.append(f"  Macro F1: {metrics.get('f1_macro', 'N/A'):.4f}")

        return "\n".join(lines)

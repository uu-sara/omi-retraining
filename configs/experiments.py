# -*- coding: utf-8 -*-
"""
Experiment definitions for OMI detection model comparison.

This module defines 5 experiment variations:
    1. Baseline: All features, all categories, 5-model ensemble, logit aggregation
    2. Simplified Categories: Train with simplified OMI/NONOMI/CONTROL
    3. No Demographics: Remove age and sex from model
    4. Single Model: No ensemble (ensemble_size=1)
    5. Probability Averaging: Different ensemble aggregation method

Each experiment uses the ExperimentConfig dataclass for configuration.
"""

from typing import Dict, List
from .base_config import (
    ExperimentConfig,
    DEFAULT_OUTCOMES_CAT,
    DEFAULT_OUTCOMES_BIN,
    SIMPLIFIED_OUTCOMES_CAT
)


def get_baseline_config() -> ExperimentConfig:
    """
    Experiment 1: Baseline configuration.

    - All 10 outcome categories (detailed)
    - Include age and sex
    - 5-model ensemble
    - Aggregate logits, then softmax

    This is the standard training setup.
    """
    return ExperimentConfig(
        exp_name="baseline",
        exp_id="exp_001",
        outcomes_cat=DEFAULT_OUTCOMES_CAT.copy(),
        outcomes_bin=DEFAULT_OUTCOMES_BIN.copy(),
        use_simplified_categories=False,
        include_age=True,
        include_sex=True,
        ensemble_size=5,
        ensemble_aggregation="logits",
    )


def get_simplified_categories_config() -> ExperimentConfig:
    """
    Experiment 2: Simplified outcome categories.

    - 3 outcome categories: OMI, NONOMI, CONTROL
    - Include age and sex
    - 5-model ensemble
    - Aggregate logits

    Tests whether training with simplified categories
    affects performance vs training with detailed categories.
    """
    return ExperimentConfig(
        exp_name="simplified_categories",
        exp_id="exp_002",
        outcomes_cat=SIMPLIFIED_OUTCOMES_CAT.copy(),
        outcomes_bin=DEFAULT_OUTCOMES_BIN.copy(),
        use_simplified_categories=True,
        include_age=True,
        include_sex=True,
        ensemble_size=5,
        ensemble_aggregation="logits",
    )


def get_no_demographics_config() -> ExperimentConfig:
    """
    Experiment 3: No demographics (age/sex removed).

    - All 10 outcome categories
    - No age, no sex
    - 5-model ensemble
    - Aggregate logits

    Tests the contribution of demographic features to model performance.
    """
    return ExperimentConfig(
        exp_name="no_demographics",
        exp_id="exp_003",
        outcomes_cat=DEFAULT_OUTCOMES_CAT.copy(),
        outcomes_bin=DEFAULT_OUTCOMES_BIN.copy(),
        use_simplified_categories=False,
        include_age=False,
        include_sex=False,
        ensemble_size=5,
        ensemble_aggregation="logits",
    )


def get_single_model_config() -> ExperimentConfig:
    """
    Experiment 4: Single model (no ensemble).

    - All 10 outcome categories
    - Include age and sex
    - Single model (ensemble_size=1)
    - N/A aggregation (single model)

    Tests the benefit of ensemble learning.
    """
    return ExperimentConfig(
        exp_name="single_model",
        exp_id="exp_004",
        outcomes_cat=DEFAULT_OUTCOMES_CAT.copy(),
        outcomes_bin=DEFAULT_OUTCOMES_BIN.copy(),
        use_simplified_categories=False,
        include_age=True,
        include_sex=True,
        ensemble_size=1,
        ensemble_aggregation="logits",  # N/A for single model, but kept for consistency
    )


def get_probability_averaging_config() -> ExperimentConfig:
    """
    Experiment 5: Probability averaging (different aggregation).

    - All 10 outcome categories
    - Include age and sex
    - 5-model ensemble
    - Aggregate probabilities (softmax first, then average)

    Tests the effect of ensemble aggregation strategy.
    """
    return ExperimentConfig(
        exp_name="probability_averaging",
        exp_id="exp_005",
        outcomes_cat=DEFAULT_OUTCOMES_CAT.copy(),
        outcomes_bin=DEFAULT_OUTCOMES_BIN.copy(),
        use_simplified_categories=False,
        include_age=True,
        include_sex=True,
        ensemble_size=5,
        ensemble_aggregation="probabilities",
    )


# Dictionary of all experiment configurations
EXPERIMENTS: Dict[str, ExperimentConfig] = {
    "baseline": get_baseline_config(),
    "simplified_categories": get_simplified_categories_config(),
    "no_demographics": get_no_demographics_config(),
    "single_model": get_single_model_config(),
    "probability_averaging": get_probability_averaging_config(),
}


def get_experiment(name: str) -> ExperimentConfig:
    """
    Get experiment configuration by name.

    Args:
        name: Experiment name (baseline, simplified_categories, no_demographics,
              single_model, probability_averaging)

    Returns:
        ExperimentConfig for the requested experiment

    Raises:
        ValueError: If experiment name is not recognized
    """
    if name not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS.keys())
        raise ValueError(
            f"Unknown experiment '{name}'. Available experiments: {available}"
        )

    # Return a fresh copy to avoid shared state issues
    config = EXPERIMENTS[name]
    return ExperimentConfig(**config.to_dict())


def get_all_experiments() -> List[ExperimentConfig]:
    """
    Get all experiment configurations.

    Returns:
        List of ExperimentConfig instances for all defined experiments
    """
    return [get_experiment(name) for name in EXPERIMENTS.keys()]


def list_experiments() -> List[str]:
    """
    List available experiment names.

    Returns:
        List of experiment names
    """
    return list(EXPERIMENTS.keys())


# Summary table for documentation
EXPERIMENT_SUMMARY = """
+-----+------------------------+-------------+----------+----------+---------------+
| ID  | Name                   | Categories  | Age/Sex  | Ensemble | Aggregation   |
+-----+------------------------+-------------+----------+----------+---------------+
| 001 | baseline               | 10 detailed | Yes      | 5        | logits        |
| 002 | simplified_categories  | 3 simple    | Yes      | 5        | logits        |
| 003 | no_demographics        | 10 detailed | No       | 5        | logits        |
| 004 | single_model           | 10 detailed | Yes      | 1        | N/A           |
| 005 | probability_averaging  | 10 detailed | Yes      | 5        | probabilities |
+-----+------------------------+-------------+----------+----------+---------------+

Evaluation Note:
All experiments are evaluated on the same metrics: AUROC and Brier score for
OMI, NONOMI, and CONTROL. For experiments with detailed categories, probabilities
are aggregated at evaluation time:
  - pr_control = pr_control_nomyoperi + pr_control_myoperi
  - pr_nonomi = pr_nstemi_nonomi + pr_stemi_nonomi
  - pr_omi = sum of all OMI subcategories
"""

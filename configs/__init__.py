# -*- coding: utf-8 -*-
"""
Experiment configuration module.

Provides configuration dataclasses and experiment definitions
for the OMI detection model comparison study.
"""
from .base_config import (
    ExperimentConfig,
    DEFAULT_OUTCOMES_CAT,
    DEFAULT_OUTCOMES_BIN,
    SIMPLIFIED_OUTCOMES_CAT,
    CATEGORY_MAPPING,
    SIMPLIFIED_TO_DETAILED,
)
from .experiments import (
    get_experiment,
    get_all_experiments,
    list_experiments,
    EXPERIMENTS,
    EXPERIMENT_SUMMARY,
)

__all__ = [
    'ExperimentConfig',
    'DEFAULT_OUTCOMES_CAT',
    'DEFAULT_OUTCOMES_BIN',
    'SIMPLIFIED_OUTCOMES_CAT',
    'CATEGORY_MAPPING',
    'SIMPLIFIED_TO_DETAILED',
    'get_experiment',
    'get_all_experiments',
    'list_experiments',
    'EXPERIMENTS',
    'EXPERIMENT_SUMMARY',
]

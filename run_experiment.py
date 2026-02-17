#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI entry point for running experiments.

Usage:
    # Run a specific experiment
    python run_experiment.py --config baseline

    # Run experiment with custom data paths
    python run_experiment.py --config baseline \
        --hdf5 /path/to/data.hdf5 \
        --txt /path/to/outcomes.txt

    # Run all experiments
    python run_experiment.py --all

    # List available experiments
    python run_experiment.py --list
"""
__author__ = "Stefan Gustafsson"
__email__ = "stefan.gustafsson@medsci.uu.se"

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import get_experiment, get_all_experiments, list_experiments, EXPERIMENT_SUMMARY
from src.experiment_runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OMI detection experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXPERIMENT_SUMMARY
    )

    # Experiment selection
    exp_group = parser.add_mutually_exclusive_group(required=True)
    exp_group.add_argument(
        "--config", "-c",
        type=str,
        choices=list_experiments(),
        help="Name of experiment configuration to run"
    )
    exp_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all defined experiments"
    )
    exp_group.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available experiment configurations"
    )

    # Data paths
    parser.add_argument(
        "--hdf5",
        type=str,
        default="",
        help="Path to HDF5 file with ECG data"
    )
    parser.add_argument(
        "--txt",
        type=str,
        default="",
        help="Path to text file with outcomes and splits"
    )

    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="experiments",
        help="Base directory for experiment outputs (default: experiments)"
    )

    # Testing
    parser.add_argument(
        "--test-name",
        type=str,
        default="test",
        help="Name of test split for evaluation (default: test)"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train, skip evaluation"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate (requires pre-trained models)"
    )

    # Overrides
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to append to experiment directory name (e.g., 'subsample_20pct')"
    )

    return parser.parse_args()


def run_single_experiment(
    config_name: str,
    args: argparse.Namespace
) -> None:
    """Run a single experiment."""
    logger.info(f"Loading experiment configuration: {config_name}")

    # Get configuration
    config = get_experiment(config_name)

    # Apply overrides
    if args.hdf5:
        config.hdf5_path = args.hdf5
    if args.txt:
        config.txt_path = args.txt
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.suffix:
        config.data_suffix = args.suffix

    # Validate data paths
    if not config.hdf5_path or not config.txt_path:
        logger.error("Data paths not specified. Use --hdf5 and --txt arguments.")
        sys.exit(1)

    # Create runner
    runner = ExperimentRunner(config)

    # Run experiment
    if args.eval_only:
        logger.info("Running evaluation only...")
        results = runner.evaluate(test_name=args.test_name)
    elif args.train_only:
        logger.info("Running training only...")
        results = runner.train()
    else:
        logger.info("Running full experiment (train + evaluate)...")
        results = runner.run(test_name=args.test_name)

    logger.info(f"Experiment {config_name} completed!")
    return results


def main():
    args = parse_args()

    # Handle list command
    if args.list:
        print("\nAvailable Experiments:")
        print("=" * 60)
        for name in list_experiments():
            config = get_experiment(name)
            print(f"\n{config.exp_id}: {name}")
            print(f"  - Ensemble size: {config.ensemble_size}")
            print(f"  - Aggregation: {config.ensemble_aggregation}")
            print(f"  - Demographics: age={config.include_age}, sex={config.include_sex}")
            print(f"  - Simplified categories: {config.use_simplified_categories}")
        print("\n" + EXPERIMENT_SUMMARY)
        return

    # Run experiments
    if args.all:
        logger.info("Running all experiments...")
        experiments = list_experiments()
        for exp_name in experiments:
            try:
                run_single_experiment(exp_name, args)
            except Exception as e:
                logger.error(f"Experiment {exp_name} failed: {e}")
                continue
    else:
        run_single_experiment(args.config, args)

    logger.info("All done!")


if __name__ == "__main__":
    main()

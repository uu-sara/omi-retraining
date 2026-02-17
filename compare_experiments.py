#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI entry point for comparing experiment results.

Usage:
    # Compare all experiments in default directory
    python compare_experiments.py

    # Compare experiments from specific directory
    python compare_experiments.py --experiments-dir /path/to/experiments

    # Output to specific results directory
    python compare_experiments.py --results-dir /path/to/results

    # Find best experiment by a specific metric
    python compare_experiments.py --best auroc_omi
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_comparator import ExperimentComparator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare results across OMI detection experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--experiments-dir", "-e",
        type=str,
        default="experiments",
        help="Directory containing experiment folders (default: experiments)"
    )

    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="experiments/results",
        help="Directory for comparison outputs (default: experiments/results)"
    )

    parser.add_argument(
        "--best", "-b",
        type=str,
        default=None,
        metavar="METRIC",
        help="Find best experiment by specified metric (e.g., agg_auroc_omi)"
    )

    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Only generate comparison table (skip plots)"
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Specific metrics to include in comparison table"
    )

    parser.add_argument(
        "--outcomes-file",
        type=str,
        default="example_data/test_outcomes.txt",
        help="Path to outcomes file with demographics (default: example_data/test_outcomes.txt)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create comparator
    comparator = ExperimentComparator(
        experiments_dir=args.experiments_dir,
        results_dir=args.results_dir,
        outcomes_file=args.outcomes_file,
    )

    # Load experiments
    experiments = comparator.load_all_experiments()

    if not experiments:
        logger.error(f"No experiments found in {args.experiments_dir}")
        sys.exit(1)

    print(f"\nFound {len(experiments)} experiments:")
    for name in experiments.keys():
        print(f"  - {name}")

    # Find best if requested
    if args.best:
        best_name, best_data = comparator.get_best_experiment(
            metric=args.best,
            higher_is_better=True
        )
        if best_name:
            print(f"\nBest experiment by {args.best}:")
            print(f"  Name: {best_name}")
            print(f"  Value: {best_data['metrics'].get(args.best):.4f}")
        return

    # Generate comparisons
    if args.table_only:
        df = comparator.create_comparison_table(
            metrics_to_compare=args.metrics,
            save_csv=True
        )
        print("\nComparison Table:")
        print("=" * 80)
        print(df.to_string(index=False))
    else:
        outputs = comparator.generate_all_comparisons()
        print("\nGenerated outputs:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

    logger.info("Comparison complete!")


if __name__ == "__main__":
    main()

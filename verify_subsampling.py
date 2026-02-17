#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify subsampling results by comparing class distributions and overlaps.

Usage:
    python verify_subsampling.py

    # Or with custom paths:
    python verify_subsampling.py \
        --original data.txt \
        --subsampled data_subsample_20pct.txt data_subsample_1pct.txt
"""
__author__ = "Generated for OMI retraining project"

import argparse
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import numpy as np


# Class columns to analyze
OUTCOME_COLUMNS = [
    'control_nomyoperi',
    'control_myoperi',
    'mi_nstemi_nonomi',
    'mi_stemi_nonomi',
    'mi_nstemi_omi_lmca_lad',
    'mi_nstemi_omi_lcx',
    'mi_nstemi_omi_rca',
    'mi_stemi_omi_lmca_lad',
    'mi_stemi_omi_lcx',
    'mi_stemi_omi_rca',
    'lbbb',
]

# Groupings for summary
CLASS_GROUPS = {
    'Control (nomyoperi)': ['control_nomyoperi'],
    'Control (myoperi)': ['control_myoperi'],
    'NONOMI': ['mi_nstemi_nonomi', 'mi_stemi_nonomi'],
    'OMI': [
        'mi_nstemi_omi_lmca_lad', 'mi_nstemi_omi_lcx', 'mi_nstemi_omi_rca',
        'mi_stemi_omi_lmca_lad', 'mi_stemi_omi_lcx', 'mi_stemi_omi_rca'
    ],
    'LBBB': ['lbbb'],
}


def load_data(path: str) -> pd.DataFrame:
    """Load outcomes file."""
    df = pd.read_csv(path, sep='\t')
    return df


def count_classes(df: pd.DataFrame, columns: List[str]) -> Dict[str, int]:
    """Count samples in each class."""
    counts = {}
    for col in columns:
        if col in df.columns:
            counts[col] = int(df[col].sum())
        else:
            counts[col] = 0
    return counts


def count_groups(df: pd.DataFrame, groups: Dict[str, List[str]]) -> Dict[str, int]:
    """Count samples in grouped categories."""
    counts = {}
    for group_name, columns in groups.items():
        available = [c for c in columns if c in df.columns]
        if available:
            # Any column in group is 1
            counts[group_name] = int(df[available].any(axis=1).sum())
        else:
            counts[group_name] = 0
    return counts


def count_by_split(df: pd.DataFrame, split_col: str = 'split') -> Dict[str, int]:
    """Count samples per split."""
    if split_col not in df.columns:
        return {'all': len(df)}
    return df[split_col].value_counts().to_dict()


def print_table(headers: List[str], rows: List[List], title: str = None):
    """Print a formatted table."""
    if title:
        print(f"\n{'=' * 80}")
        print(f" {title}")
        print('=' * 80)

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def analyze_overlap(
    original_ids: Set,
    subsampled_ids: Set,
    name: str
) -> Dict:
    """Analyze overlap between original and subsampled datasets."""
    in_both = original_ids & subsampled_ids
    only_in_original = original_ids - subsampled_ids
    only_in_subsampled = subsampled_ids - original_ids

    return {
        'name': name,
        'original_count': len(original_ids),
        'subsampled_count': len(subsampled_ids),
        'in_both': len(in_both),
        'only_in_original': len(only_in_original),
        'only_in_subsampled': len(only_in_subsampled),
        'is_proper_subset': len(only_in_subsampled) == 0,
        'retention_pct': 100 * len(in_both) / len(original_ids) if original_ids else 0
    }


def main():
    parser = argparse.ArgumentParser(
        description='Verify subsampling results by comparing class distributions.'
    )
    parser.add_argument(
        '--original',
        type=str,
        default='all_omi_data/omi_analysis_data_20250217.txt',
        help='Path to original outcomes file'
    )
    parser.add_argument(
        '--subsampled',
        type=str,
        nargs='+',
        default=[
            'all_omi_data/subsets/omi_analysis_data_20250217_subsample_20pct.txt',
            'all_omi_data/subsets/omi_analysis_data_20250217_subsample_1pct.txt'
        ],
        help='Paths to subsampled outcomes files'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory containing the data files'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load all files
    print("\n" + "=" * 80)
    print(" Loading Data Files")
    print("=" * 80)

    files = {}
    original_path = data_dir / args.original
    if original_path.exists():
        files['Original'] = load_data(str(original_path))
        print(f"  Loaded: {args.original} ({len(files['Original'])} records)")
    else:
        print(f"  WARNING: Original file not found: {original_path}")

    for sub_path in args.subsampled:
        full_path = data_dir / sub_path
        if full_path.exists():
            # Extract percentage from filename for label
            name = Path(sub_path).stem
            if 'subsample_' in name:
                label = name.split('subsample_')[-1].replace('pct', '%').replace('_', '.')
            else:
                label = name
            files[label] = load_data(str(full_path))
            print(f"  Loaded: {sub_path} ({len(files[label])} records)")
        else:
            print(f"  WARNING: File not found: {full_path}")

    if len(files) == 0:
        print("\nNo files found. Exiting.")
        return

    # ==========================================================================
    # Class counts per file
    # ==========================================================================
    print_table(
        headers=['Class'] + list(files.keys()),
        rows=[
            [col] + [count_classes(df, [col]).get(col, 0) for df in files.values()]
            for col in OUTCOME_COLUMNS if any(col in df.columns for df in files.values())
        ],
        title="Detailed Class Counts"
    )

    # ==========================================================================
    # Grouped class counts
    # ==========================================================================
    print_table(
        headers=['Group'] + list(files.keys()),
        rows=[
            [group] + [count_groups(df, {group: cols}).get(group, 0) for df in files.values()]
            for group, cols in CLASS_GROUPS.items()
        ],
        title="Grouped Class Counts"
    )

    # ==========================================================================
    # Counts by split
    # ==========================================================================
    all_splits = set()
    for df in files.values():
        if 'split' in df.columns:
            all_splits.update(df['split'].unique())

    if all_splits:
        rows = []
        for split in sorted(all_splits):
            row = [split]
            for df in files.values():
                if 'split' in df.columns:
                    row.append(int((df['split'] == split).sum()))
                else:
                    row.append('-')
            rows.append(row)

        # Add total row
        rows.append(['TOTAL'] + [len(df) for df in files.values()])

        print_table(
            headers=['Split'] + list(files.keys()),
            rows=rows,
            title="Counts by Data Split"
        )

    # ==========================================================================
    # Retention analysis (what % of each class was kept)
    # ==========================================================================
    if 'Original' in files:
        original_df = files['Original']
        other_files = {k: v for k, v in files.items() if k != 'Original'}

        if other_files:
            headers = ['Class', 'Original'] + [f'{k} (count)' for k in other_files.keys()] + \
                      [f'{k} (%)' for k in other_files.keys()]

            rows = []
            for col in OUTCOME_COLUMNS:
                if col not in original_df.columns:
                    continue

                orig_count = int(original_df[col].sum())
                row = [col, orig_count]

                # Counts
                for name, df in other_files.items():
                    if col in df.columns:
                        row.append(int(df[col].sum()))
                    else:
                        row.append(0)

                # Percentages
                for name, df in other_files.items():
                    if col in df.columns and orig_count > 0:
                        sub_count = int(df[col].sum())
                        row.append(f"{100 * sub_count / orig_count:.1f}%")
                    else:
                        row.append('-')

                rows.append(row)

            print_table(headers=headers, rows=rows, title="Class Retention Analysis")

    # ==========================================================================
    # Overlap analysis
    # ==========================================================================
    if 'Original' in files and len(files) > 1:
        print("\n" + "=" * 80)
        print(" Overlap Analysis")
        print("=" * 80)

        original_df = files['Original']
        original_ids = set(original_df['id_record'])

        for name, df in files.items():
            if name == 'Original':
                continue

            subsampled_ids = set(df['id_record'])
            overlap = analyze_overlap(original_ids, subsampled_ids, name)

            print(f"\n  {name} vs Original:")
            print(f"    Original records:      {overlap['original_count']:,}")
            print(f"    Subsampled records:    {overlap['subsampled_count']:,}")
            print(f"    Records in both:       {overlap['in_both']:,}")
            print(f"    Only in original:      {overlap['only_in_original']:,}")
            print(f"    Only in subsampled:    {overlap['only_in_subsampled']:,}")
            print(f"    Is proper subset:      {'Yes' if overlap['is_proper_subset'] else 'NO - PROBLEM!'}")
            print(f"    Overall retention:     {overlap['retention_pct']:.2f}%")

        # Check overlap between subsampled files
        subsampled_files = {k: v for k, v in files.items() if k != 'Original'}
        if len(subsampled_files) > 1:
            print("\n  Overlap between subsampled files:")

            file_names = list(subsampled_files.keys())
            for i, name1 in enumerate(file_names):
                for name2 in file_names[i+1:]:
                    ids1 = set(subsampled_files[name1]['id_record'])
                    ids2 = set(subsampled_files[name2]['id_record'])

                    in_both = len(ids1 & ids2)
                    only_in_1 = len(ids1 - ids2)
                    only_in_2 = len(ids2 - ids1)

                    # Check if smaller is subset of larger
                    if len(ids1) < len(ids2):
                        smaller, larger = name1, name2
                        is_subset = ids1.issubset(ids2)
                    else:
                        smaller, larger = name2, name1
                        is_subset = ids2.issubset(ids1)

                    print(f"\n    {name1} vs {name2}:")
                    print(f"      Records in {name1}: {len(ids1):,}")
                    print(f"      Records in {name2}: {len(ids2):,}")
                    print(f"      Records in both:    {in_both:,}")
                    print(f"      Only in {name1}:    {only_in_1:,}")
                    print(f"      Only in {name2}:    {only_in_2:,}")
                    print(f"      {smaller} is subset of {larger}: {'Yes' if is_subset else 'No'}")

    # ==========================================================================
    # Summary statistics
    # ==========================================================================
    print("\n" + "=" * 80)
    print(" Summary")
    print("=" * 80)

    for name, df in files.items():
        total = len(df)
        rare_cols = [c for c in OUTCOME_COLUMNS if c != 'control_nomyoperi' and c in df.columns]
        rare_count = int(df[rare_cols].any(axis=1).sum()) if rare_cols else 0
        majority_count = int(df['control_nomyoperi'].sum()) if 'control_nomyoperi' in df.columns else 0

        print(f"\n  {name}:")
        print(f"    Total records:     {total:,}")
        print(f"    Rare classes:      {rare_count:,} ({100*rare_count/total:.1f}%)")
        print(f"    Majority class:    {majority_count:,} ({100*majority_count/total:.1f}%)")
        print(f"    Class imbalance:   1:{majority_count/rare_count:.1f}" if rare_count > 0 else "")

    print("\n" + "=" * 80)
    print(" Done!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

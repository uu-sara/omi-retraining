#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Count samples by split and outcome class.

Usage:
    python count_by_split.py /path/to/outcomes.txt
"""

import argparse
import pandas as pd

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


def main():
    parser = argparse.ArgumentParser(description='Count samples by split and outcome class')
    parser.add_argument('txt_file', type=str, help='Path to outcomes txt file')
    parser.add_argument('--split-col', type=str, default='split', help='Name of split column')
    args = parser.parse_args()

    df = pd.read_csv(args.txt_file, sep='\t')

    # Find which outcome columns exist
    outcome_cols = [c for c in OUTCOME_COLUMNS if c in df.columns]

    # Melt to long format
    df_long = df.melt(
        id_vars=[args.split_col],
        value_vars=outcome_cols,
        var_name='outcome',
        value_name='value'
    )

    # Filter to only rows where value == 1 and count
    counts = df_long[df_long['value'] == 1].groupby([args.split_col, 'outcome']).size().reset_index(name='count')

    # Pivot for easier reading
    pivot = counts.pivot(index='outcome', columns=args.split_col, values='count').fillna(0).astype(int)

    # Reorder rows to match OUTCOME_COLUMNS order
    pivot = pivot.reindex([c for c in OUTCOME_COLUMNS if c in pivot.index])

    # Add total row
    pivot.loc['TOTAL'] = pivot.sum()

    print("\nCounts by split and outcome:")
    print("=" * 80)
    print(pivot.to_string())
    print()


if __name__ == "__main__":
    main()

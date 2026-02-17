#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Subsample HDF5 ECG dataset by reducing control_nomyoperi (majority class).

Creates new HDF5 and text files with all rare classes preserved and only
a percentage of the control_nomyoperi samples retained.

Usage:
    python subsample_hdf5.py \
        --hdf5 /path/to/data.hdf5 \
        --txt /path/to/outcomes.txt \
        --output-dir /path/to/output \
        --percentages 1 20

This will create:
    - data_subsample_1pct.hdf5 + outcomes_subsample_1pct.txt
    - data_subsample_20pct.hdf5 + outcomes_subsample_20pct.txt
"""
__author__ = "Generated for OMI retraining project"

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Columns that indicate rare classes (keep all samples with any of these = 1)
RARE_CLASS_COLUMNS = [
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

MAJORITY_CLASS_COLUMN = 'control_nomyoperi'


def identify_rare_and_majority(
    df: pd.DataFrame,
    rare_columns: List[str],
    majority_column: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify indices of rare class samples and majority class samples.

    Args:
        df: DataFrame with outcome columns
        rare_columns: List of column names indicating rare classes
        majority_column: Column name for majority class

    Returns:
        Tuple of (rare_indices, majority_indices) as numpy arrays
    """
    # Find which rare columns exist in the dataframe
    available_rare = [col for col in rare_columns if col in df.columns]

    if not available_rare:
        logger.warning(f"No rare class columns found. Available: {df.columns.tolist()}")

    # A sample is rare if ANY rare class column is 1
    is_rare = df[available_rare].any(axis=1) if available_rare else pd.Series(False, index=df.index)

    # A sample is majority if it's control_nomyoperi AND not rare
    if majority_column in df.columns:
        is_majority = (df[majority_column] == 1) & ~is_rare
    else:
        # Fallback: majority is anything not rare
        is_majority = ~is_rare

    rare_indices = np.where(is_rare)[0]
    majority_indices = np.where(is_majority)[0]

    return rare_indices, majority_indices


def subsample_dataset(
    hdf5_path: str,
    txt_path: str,
    output_dir: str,
    percentage: float,
    seed: int = 42,
    traces_dset: str = 'ecg_normalized',
    record_id_dset: str = 'id_record'
) -> Tuple[str, str]:
    """
    Create a subsampled version of the dataset.

    Args:
        hdf5_path: Path to input HDF5 file
        txt_path: Path to input outcomes text file
        output_dir: Directory for output files
        percentage: Percentage of majority class to keep (1-100)
        seed: Random seed for reproducibility
        traces_dset: Name of traces dataset in HDF5
        record_id_dset: Name of record ID dataset in HDF5

    Returns:
        Tuple of (output_hdf5_path, output_txt_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filenames
    pct_str = f"{percentage:.0f}pct" if percentage >= 1 else f"{percentage:.1f}pct".replace('.', '_')
    hdf5_stem = Path(hdf5_path).stem
    txt_stem = Path(txt_path).stem

    output_hdf5 = output_dir / f"{hdf5_stem}_subsample_{pct_str}.hdf5"
    output_txt = output_dir / f"{txt_stem}_subsample_{pct_str}.txt"

    logger.info(f"Creating {percentage}% subsample...")
    logger.info(f"  Input HDF5: {hdf5_path}")
    logger.info(f"  Input TXT:  {txt_path}")

    # Load outcomes file
    df = pd.read_csv(txt_path, sep='\t')
    logger.info(f"  Loaded {len(df)} records from outcomes file")

    # Open HDF5 and get record IDs
    with h5py.File(hdf5_path, 'r') as f_in:
        hdf5_record_ids = np.array(f_in[record_id_dset])
        traces_shape = f_in[traces_dset].shape
        traces_dtype = f_in[traces_dset].dtype
        logger.info(f"  HDF5 traces shape: {traces_shape}, dtype: {traces_dtype}")

    # Align dataframe with HDF5 order
    df = df.set_index('id_record')
    df = df.reindex(hdf5_record_ids)
    df = df.reset_index()

    # Identify rare and majority class samples
    rare_indices, majority_indices = identify_rare_and_majority(
        df, RARE_CLASS_COLUMNS, MAJORITY_CLASS_COLUMN
    )

    logger.info(f"  Rare class samples: {len(rare_indices)}")
    logger.info(f"  Majority class samples: {len(majority_indices)}")

    # Subsample majority class
    rng = np.random.RandomState(seed)
    n_keep = max(1, int(len(majority_indices) * percentage / 100.0))
    sampled_majority = rng.choice(majority_indices, size=n_keep, replace=False)

    logger.info(f"  Keeping {n_keep} majority samples ({percentage}%)")

    # Combine and sort indices for efficient HDF5 reading
    selected_indices = np.sort(np.concatenate([rare_indices, sampled_majority]))
    total_selected = len(selected_indices)

    logger.info(f"  Total selected samples: {total_selected}")

    # Create subsampled outcomes file
    df_selected = df.iloc[selected_indices].copy()
    df_selected.to_csv(output_txt, sep='\t', index=False)
    logger.info(f"  Written: {output_txt}")

    # Create subsampled HDF5 file
    with h5py.File(hdf5_path, 'r') as f_in:
        with h5py.File(output_hdf5, 'w') as f_out:
            # Copy traces for selected indices
            logger.info("  Copying traces (this may take a while)...")

            # Read in chunks to avoid memory issues
            chunk_size = 10000
            new_shape = (total_selected,) + traces_shape[1:]
            traces_out = f_out.create_dataset(
                traces_dset,
                shape=new_shape,
                dtype=traces_dtype,
                chunks=(min(100, total_selected),) + traces_shape[1:]
            )

            for i in range(0, total_selected, chunk_size):
                chunk_indices = selected_indices[i:i + chunk_size]
                chunk_data = f_in[traces_dset][chunk_indices]
                traces_out[i:i + len(chunk_indices)] = chunk_data

                if (i + chunk_size) % 50000 == 0 or i + chunk_size >= total_selected:
                    logger.info(f"    Processed {min(i + chunk_size, total_selected)}/{total_selected} samples")

            # Copy record IDs for selected indices
            selected_record_ids = hdf5_record_ids[selected_indices]

            # Determine the dtype for record IDs
            original_id_dtype = f_in[record_id_dset].dtype
            if original_id_dtype.kind == 'S':  # byte string
                f_out.create_dataset(record_id_dset, data=selected_record_ids, dtype=original_id_dtype)
            elif original_id_dtype.kind == 'O':  # object (variable length string)
                dt = h5py.special_dtype(vlen=str)
                f_out.create_dataset(record_id_dset, data=selected_record_ids, dtype=dt)
            else:
                f_out.create_dataset(record_id_dset, data=selected_record_ids)

            # Copy any other datasets that exist
            for key in f_in.keys():
                if key not in [traces_dset, record_id_dset]:
                    if isinstance(f_in[key], h5py.Dataset):
                        if f_in[key].shape[0] == traces_shape[0]:
                            # This dataset has same length as traces, subsample it
                            f_out.create_dataset(key, data=f_in[key][selected_indices])
                            logger.info(f"    Also subsampled dataset: {key}")
                        else:
                            # Copy as-is
                            f_out.create_dataset(key, data=f_in[key][:])
                            logger.info(f"    Copied dataset as-is: {key}")

    logger.info(f"  Written: {output_hdf5}")

    # Report file sizes
    input_size_gb = Path(hdf5_path).stat().st_size / (1024**3)
    output_size_gb = output_hdf5.stat().st_size / (1024**3)
    logger.info(f"  Size reduction: {input_size_gb:.2f} GB -> {output_size_gb:.2f} GB")

    return str(output_hdf5), str(output_txt)


def main():
    parser = argparse.ArgumentParser(
        description='Subsample HDF5 ECG dataset by reducing majority class (control_nomyoperi).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create 1% and 20% subsamples
    python subsample_hdf5.py --hdf5 data.hdf5 --txt outcomes.txt --output-dir ./subsampled --percentages 1 20

    # Create just a 5% subsample with custom seed
    python subsample_hdf5.py --hdf5 data.hdf5 --txt outcomes.txt --output-dir ./subsampled --percentages 5 --seed 123
        """
    )

    parser.add_argument(
        '--hdf5',
        type=str,
        required=True,
        help='Path to input HDF5 file containing ECG traces'
    )
    parser.add_argument(
        '--txt',
        type=str,
        required=True,
        help='Path to input tab-separated outcomes file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory for output files'
    )
    parser.add_argument(
        '--percentages',
        type=float,
        nargs='+',
        default=[1, 20],
        help='Percentages of majority class to keep (default: 1 20)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--traces-dataset',
        type=str,
        default='ecg_normalized',
        help='Name of traces dataset in HDF5 (default: ecg_normalized)'
    )
    parser.add_argument(
        '--record-id-dataset',
        type=str,
        default='id_record',
        help='Name of record ID dataset in HDF5 (default: id_record)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.hdf5).exists():
        logger.error(f"HDF5 file not found: {args.hdf5}")
        sys.exit(1)

    if not Path(args.txt).exists():
        logger.error(f"Outcomes file not found: {args.txt}")
        sys.exit(1)

    for pct in args.percentages:
        if not 0 < pct <= 100:
            logger.error(f"Percentage must be between 0 and 100, got: {pct}")
            sys.exit(1)

    # Process each percentage
    logger.info("=" * 60)
    logger.info("HDF5 Subsampling Tool")
    logger.info("=" * 60)
    logger.info(f"Rare class columns: {RARE_CLASS_COLUMNS}")
    logger.info(f"Majority class column: {MAJORITY_CLASS_COLUMN}")
    logger.info(f"Percentages to create: {args.percentages}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 60)

    created_files = []

    for pct in args.percentages:
        logger.info("")
        hdf5_out, txt_out = subsample_dataset(
            hdf5_path=args.hdf5,
            txt_path=args.txt,
            output_dir=args.output_dir,
            percentage=pct,
            seed=args.seed,
            traces_dset=args.traces_dataset,
            record_id_dset=args.record_id_dataset
        )
        created_files.append((pct, hdf5_out, txt_out))

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for pct, hdf5_out, txt_out in created_files:
        logger.info(f"  {pct}% subsample:")
        logger.info(f"    HDF5: {hdf5_out}")
        logger.info(f"    TXT:  {txt_out}")
    logger.info("")
    logger.info("Done!")


if __name__ == "__main__":
    main()

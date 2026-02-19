#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract specific splits from HDF5+txt dataset into separate files.

Usage:
    python extract_splits.py \
        --hdf5 /path/to/data.hdf5 \
        --txt /path/to/outcomes.txt \
        --output-dir /path/to/output \
        --splits test_rand test_temp

This will create:
    - data_test_rand.hdf5 + outcomes_test_rand.txt
    - data_test_temp.hdf5 + outcomes_test_temp.txt
"""

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


def extract_split(
    hdf5_path: str,
    txt_path: str,
    output_dir: str,
    split_name: str,
    split_col: str = 'split',
    traces_dset: str = 'ecg_normalized',
    record_id_dset: str = 'id_record'
) -> Tuple[str, str]:
    """
    Extract a specific split from the dataset into new files.

    Args:
        hdf5_path: Path to input HDF5 file
        txt_path: Path to input outcomes text file
        output_dir: Directory for output files
        split_name: Name of split to extract (e.g., 'test_rand')
        split_col: Name of split column in text file
        traces_dset: Name of traces dataset in HDF5
        record_id_dset: Name of record ID dataset in HDF5

    Returns:
        Tuple of (output_hdf5_path, output_txt_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filenames
    hdf5_stem = Path(hdf5_path).stem
    txt_stem = Path(txt_path).stem

    output_hdf5 = output_dir / f"{hdf5_stem}_{split_name}.hdf5"
    output_txt = output_dir / f"{txt_stem}_{split_name}.txt"

    logger.info(f"Extracting split '{split_name}'...")
    logger.info(f"  Input HDF5: {hdf5_path}")
    logger.info(f"  Input TXT:  {txt_path}")

    # Load outcomes file
    df = pd.read_csv(txt_path, sep='\t')
    logger.info(f"  Loaded {len(df)} records from outcomes file")

    # Check split column exists
    if split_col not in df.columns:
        raise ValueError(f"Split column '{split_col}' not found in outcomes file")

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

    # Find indices for the requested split
    split_mask = df[split_col] == split_name
    selected_indices = np.where(split_mask)[0]
    n_selected = len(selected_indices)

    if n_selected == 0:
        logger.warning(f"  No records found for split '{split_name}'")
        logger.info(f"  Available splits: {df[split_col].unique().tolist()}")
        return None, None

    logger.info(f"  Found {n_selected} records for split '{split_name}'")

    # Create extracted outcomes file
    df_selected = df.iloc[selected_indices].copy()
    df_selected.to_csv(output_txt, sep='\t', index=False)
    logger.info(f"  Written: {output_txt}")

    # Create extracted HDF5 file
    with h5py.File(hdf5_path, 'r') as f_in:
        with h5py.File(output_hdf5, 'w') as f_out:
            # Copy traces for selected indices
            logger.info("  Copying traces...")

            # Read in chunks to avoid memory issues
            chunk_size = 10000
            new_shape = (n_selected,) + traces_shape[1:]
            traces_out = f_out.create_dataset(
                traces_dset,
                shape=new_shape,
                dtype=traces_dtype,
                chunks=(min(100, n_selected),) + traces_shape[1:]
            )

            for i in range(0, n_selected, chunk_size):
                chunk_indices = selected_indices[i:i + chunk_size]
                chunk_data = f_in[traces_dset][chunk_indices]
                traces_out[i:i + len(chunk_indices)] = chunk_data

                if (i + chunk_size) % 50000 == 0 or i + chunk_size >= n_selected:
                    logger.info(f"    Processed {min(i + chunk_size, n_selected)}/{n_selected} samples")

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
                            # This dataset has same length as traces, extract it
                            f_out.create_dataset(key, data=f_in[key][selected_indices])
                            logger.info(f"    Also extracted dataset: {key}")
                        else:
                            # Copy as-is
                            f_out.create_dataset(key, data=f_in[key][:])
                            logger.info(f"    Copied dataset as-is: {key}")

    logger.info(f"  Written: {output_hdf5}")

    # Report file sizes
    input_size_gb = Path(hdf5_path).stat().st_size / (1024**3)
    output_size_gb = output_hdf5.stat().st_size / (1024**3)
    logger.info(f"  Size: {output_size_gb:.2f} GB (from {input_size_gb:.2f} GB)")

    return str(output_hdf5), str(output_txt)


def main():
    parser = argparse.ArgumentParser(
        description='Extract specific splits from HDF5+txt dataset into separate files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract test_rand and test_temp splits
    python extract_splits.py --hdf5 data.hdf5 --txt outcomes.txt --output-dir ./test_sets --splits test_rand test_temp

    # Extract just one split
    python extract_splits.py --hdf5 data.hdf5 --txt outcomes.txt --output-dir ./test_sets --splits test_rand
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
        '--splits',
        type=str,
        nargs='+',
        default=['test_rand', 'test_temp'],
        help='Split names to extract (default: test_rand test_temp)'
    )
    parser.add_argument(
        '--split-col',
        type=str,
        default='split',
        help='Name of split column in text file (default: split)'
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

    # Show available splits first
    df = pd.read_csv(args.txt, sep='\t')
    logger.info("=" * 60)
    logger.info("Extract Splits Tool")
    logger.info("=" * 60)
    logger.info(f"Available splits in data: {df[args.split_col].unique().tolist()}")
    logger.info(f"Splits to extract: {args.splits}")
    logger.info("=" * 60)

    # Process each split
    created_files = []

    for split_name in args.splits:
        logger.info("")
        hdf5_out, txt_out = extract_split(
            hdf5_path=args.hdf5,
            txt_path=args.txt,
            output_dir=args.output_dir,
            split_name=split_name,
            split_col=args.split_col,
            traces_dset=args.traces_dataset,
            record_id_dset=args.record_id_dataset
        )
        if hdf5_out and txt_out:
            created_files.append((split_name, hdf5_out, txt_out))

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for split_name, hdf5_out, txt_out in created_files:
        logger.info(f"  {split_name}:")
        logger.info(f"    HDF5: {hdf5_out}")
        logger.info(f"    TXT:  {txt_out}")
    logger.info("")
    logger.info("Done!")


if __name__ == "__main__":
    main()

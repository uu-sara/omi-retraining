# -*- coding: utf-8 -*-
"""
ECG dataset and batch dataloader for training and evaluation.

Based on code by Stefan Gustafsson (stefan.gustafsson@medsci.uu.se) for the OMI model.

This module provides:
    - OMIDataset: Loads ECG traces from HDF5 and outcomes from text file
    - BatchDataloader: Iterator for batch-wise data loading
    - Category mapping utilities for simplified outcome classification

Data Format:
    - ECG traces stored in HDF5 with shape (n_samples, seq_length, n_leads)
    - Outcomes and splits stored in tab-separated text file
    - Age is normalized (mean-centered, unit variance) to prevent gradient imbalance
"""

import logging
from typing import List, Optional, Tuple, Union, Dict

import h5py
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# Category mapping from detailed to simplified outcomes
CATEGORY_MAPPING: Dict[str, str] = {
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


def map_to_simplified_categories(
    outcomes: pd.DataFrame,
    mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Map detailed outcome categories to simplified categories.

    Combines multiple detailed outcome columns into simplified categories
    by summing the values (for one-hot encoded outcomes, this is equivalent
    to a logical OR).

    Args:
        outcomes: DataFrame with one-hot encoded outcome columns
        mapping: Dictionary mapping detailed column names to simplified names.
                 Default: CATEGORY_MAPPING

    Returns:
        DataFrame with simplified outcome columns

    Example:
        >>> df = pd.DataFrame({
        ...     'control_nomyoperi': [1, 0, 0],
        ...     'control_myoperi': [0, 0, 0],
        ...     'mi_nstemi_nonomi': [0, 1, 0],
        ...     'mi_stemi_omi_lmca_lad': [0, 0, 1]
        ... })
        >>> simplified = map_to_simplified_categories(df)
        >>> simplified.columns.tolist()
        ['control', 'nonomi', 'omi']
    """
    if mapping is None:
        mapping = CATEGORY_MAPPING

    # Find which columns from the mapping exist in the dataframe
    existing_cols = [col for col in mapping.keys() if col in outcomes.columns]

    if not existing_cols:
        logger.warning("No mapped columns found in outcomes DataFrame")
        return outcomes

    # Group columns by their simplified category
    simplified_groups: Dict[str, List[str]] = {}
    for col in existing_cols:
        simplified_name = mapping[col]
        if simplified_name not in simplified_groups:
            simplified_groups[simplified_name] = []
        simplified_groups[simplified_name].append(col)

    # Create simplified DataFrame by summing columns in each group
    # For one-hot encoded outcomes, sum gives the correct result
    simplified_data = {}
    for simplified_name, cols in simplified_groups.items():
        simplified_data[simplified_name] = outcomes[cols].sum(axis=1).clip(upper=1)

    simplified_df = pd.DataFrame(simplified_data, index=outcomes.index)

    # Preserve any columns that weren't in the mapping (e.g., binary outcomes)
    unmapped_cols = [col for col in outcomes.columns if col not in mapping]
    for col in unmapped_cols:
        simplified_df[col] = outcomes[col]

    return simplified_df


class OMIDataset:
    """
    Dataset class for ECG traces with outcomes and demographic data.
    
    Loads ECG traces from HDF5 file and corresponding outcomes, age, sex,
    and data splits from a tab-separated text file. Handles train/validation/test
    splits and age normalization.
    
    Attributes:
        traces: HDF5 dataset reference to ECG traces
        record_ids: Array of record identifiers
        outcomes: DataFrame of outcome indicators
        age: Series of normalized age values
        male: Series of sex indicators
        train: Boolean mask for training records
        valid: Boolean mask for validation records
        test: Boolean mask for test records
        
    Important Notes on Data Leakage:
        - Age normalization parameters (mean, sd) should be computed ONLY on
          training data and applied to validation/test sets
        - Pass age_mean and age_sd explicitly when loading validation/test data
        - The current implementation computes stats on ALL data if not provided,
          which can cause subtle data leakage
          
    Example:
        >>> # Training: compute normalization stats
        >>> train_dset = OMIDataset(
        ...     path_to_h5='data.hdf5',
        ...     path_to_txt='outcomes.txt',
        ...     col_outcome=['outcome1', 'outcome2']
        ... )
        >>> # Get stats from training data for use with validation/test
        >>> age_mean = train_dset.age_mean  # Should be added as attribute
        >>> age_sd = train_dset.age_sd
        >>> 
        >>> # Validation/Test: use training stats
        >>> test_dset = OMIDataset(
        ...     path_to_h5='data.hdf5',
        ...     path_to_txt='outcomes.txt',
        ...     col_outcome=['outcome1', 'outcome2'],
        ...     test=True,
        ...     age_mean=age_mean,
        ...     age_sd=age_sd
        ... )
    """
    
    def __init__(
        self,
        path_to_h5: str,
        path_to_txt: str,
        col_outcome: List[str],
        traces_dset: str = 'ecg_normalized',
        record_id_dset: str = 'id_record',
        test: bool = False,
        test_name: str = 'test',
        split_col: str = 'split',
        age_col: str = 'age',
        male_col: str = 'male',
        age_mean: Optional[float] = None,
        age_sd: Optional[float] = None,
        use_simplified_categories: bool = False,
        category_mapping: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Initialize the dataset by loading traces and outcomes.

        Args:
            path_to_h5: Path to HDF5 file containing ECG traces
            path_to_txt: Path to tab-separated file with outcomes and splits
            col_outcome: List of outcome column names
            traces_dset: Name of traces dataset in HDF5 (default: 'ecg_normalized')
            record_id_dset: Name of record ID dataset in HDF5 (default: 'id_record')
            test: Whether loading for test mode (default: False)
            test_name: Name of test split in split column (default: 'test')
            split_col: Name of split column in text file (default: 'split')
            age_col: Name of age column in text file (default: 'age')
            male_col: Name of male/sex column in text file (default: 'male')
            age_mean: Pre-computed age mean for normalization (from training data)
            age_sd: Pre-computed age standard deviation for normalization
            use_simplified_categories: If True, map detailed categories to simplified
                                       (omi, nonomi, control). Default: False
            category_mapping: Custom mapping from detailed to simplified categories.
                             Default: uses CATEGORY_MAPPING constant

        Raises:
            ValueError: If files cannot be opened or data validation fails
        """
        self.use_simplified_categories = use_simplified_categories
        self.category_mapping = category_mapping or CATEGORY_MAPPING
        # Load HDF5 file with ECG traces
        self.f = self._load_hdf5(path_to_h5, traces_dset, record_id_dset)
        self.traces = self.f[traces_dset]
        self.record_ids = np.array(self.f[record_id_dset])
        
        # Load outcomes and splits from text file
        df = self._load_outcomes_file(path_to_txt)
        
        # Validate required columns exist and have no missing values.
        # When using simplified categories, the outcome columns (omi, nonomi, control)
        # won't exist in the raw file - they get derived from detailed columns in
        # _extract_outcomes. So validate the source columns instead.
        if self.use_simplified_categories:
            detailed_cols = list(self.category_mapping.keys())
            unmapped_outcome_cols = [c for c in col_outcome
                                    if c not in set(self.category_mapping.values())]
            validate_cols = [split_col, age_col, male_col] + detailed_cols + unmapped_outcome_cols
        else:
            validate_cols = [split_col, age_col, male_col] + col_outcome
        self._validate_columns(df, validate_cols)
        
        # Align dataframe with HDF5 record order
        df = self._align_with_traces(df)
        
        # Process age with normalization
        self.age, self.age_mean, self.age_sd = self._process_age(
            df[age_col], age_mean, age_sd
        )
        self.male = df[male_col]
        
        # Set up split masks
        self._setup_splits(df, split_col, test, test_name)
        
        # Extract outcome columns only
        self.outcomes = self._extract_outcomes(df, col_outcome)
        
        logger.info(
            f"Loaded dataset with {len(self)} records: "
            f"{sum(self.train)} train, {sum(self.valid)} valid, {sum(self.test)} test"
        )
    
    def _load_hdf5(
        self, 
        path: str, 
        traces_dset: str, 
        record_id_dset: str
    ) -> h5py.File:
        """Load and validate HDF5 file."""
        try:
            f = h5py.File(path, 'r')
        except (FileNotFoundError, OSError) as e:
            raise ValueError(f"Failed to open HDF5 file '{path}': {e}") from e
        
        # Validate required datasets exist
        if traces_dset not in f:
            raise ValueError(f"Dataset '{traces_dset}' not found in HDF5 file")
        if record_id_dset not in f:
            raise ValueError(f"Dataset '{record_id_dset}' not found in HDF5 file")
        
        return f
    
    def _load_outcomes_file(self, path: str) -> pd.DataFrame:
        """Load outcomes file as DataFrame."""
        try:
            df = pd.read_csv(path, sep='\t')
        except (FileNotFoundError, OSError) as e:
            raise ValueError(f"Failed to open outcome file '{path}': {e}") from e
        
        if 'id_record' not in df.columns:
            raise ValueError("Outcome file must contain 'id_record' column")
        
        return df
    
    def _validate_columns(
        self, 
        df: pd.DataFrame, 
        required_cols: List[str]
    ) -> None:
        """Validate that required columns exist and have no missing values."""
        # Check columns exist
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for missing values
        cols_with_na = [col for col in required_cols if df[col].isna().any()]
        if cols_with_na:
            raise ValueError(f"Missing values found in columns: {', '.join(cols_with_na)}")
    
    def _align_with_traces(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align dataframe rows with HDF5 trace order."""
        self.csv_record_ids = df['id_record']
        df = df.set_index('id_record')
        
        # Validate record IDs match
        if set(self.record_ids) != set(df.index):
            hdf5_only = set(self.record_ids) - set(df.index)
            txt_only = set(df.index) - set(self.record_ids)
            raise ValueError(
                f"Record ID mismatch between HDF5 and outcome file. "
                f"HDF5-only: {len(hdf5_only)}, Text-only: {len(txt_only)}"
            )
        
        # Reorder to match HDF5 order
        df = df.reindex(self.record_ids, fill_value=0, copy=True)
        return df
    
    def _process_age(
        self,
        age_raw: pd.Series,
        age_mean: Optional[float],
        age_sd: Optional[float]
    ) -> Tuple[pd.Series, float, float]:
        """
        Normalize age values.
        
        Args:
            age_raw: Raw age values
            age_mean: Pre-computed mean (if None, computed from data)
            age_sd: Pre-computed standard deviation
            
        Returns:
            Tuple of (normalized_age, mean, sd)
        """
        if age_mean is None:
            age_mean = float(age_raw.astype(float).mean())
            age_sd = float(age_raw.astype(float).std())
            logger.warning(
                "Age normalization computed from current data. "
                "For proper validation/test, pass age_mean and age_sd from training data."
            )
        
        if age_sd is None or age_sd == 0:
            raise ValueError("Age standard deviation must not be None or 0")
        
        age_normalized = (age_raw - age_mean) / age_sd
        return age_normalized, age_mean, age_sd
    
    def _setup_splits(
        self,
        df: pd.DataFrame,
        split_col: str,
        test: bool,
        test_name: str
    ) -> None:
        """Set up boolean masks for data splits."""
        if test:
            self.test = np.array(df[split_col] == test_name)
            self.valid = np.zeros(len(df), dtype=bool)
            self.train = np.zeros(len(df), dtype=bool)
        else:
            self.test = np.zeros(len(df), dtype=bool)
            self.valid = np.array(df[split_col] == 'valid')
            self.train = np.array(df[split_col] == 'train')
    
    def _extract_outcomes(
        self,
        df: pd.DataFrame,
        col_outcome: List[str]
    ) -> pd.DataFrame:
        """Extract and validate outcome columns, optionally applying category mapping."""
        if self.use_simplified_categories:
            # When using simplified categories, we need to extract the detailed
            # columns first and then map them
            detailed_cols = list(self.category_mapping.keys())
            # Find which detailed columns exist
            available_detailed = [c for c in detailed_cols if c in df.columns]
            # Also include any columns in col_outcome that aren't in the mapping
            # (e.g., binary outcomes like lbbb)
            unmapped_cols = [c for c in col_outcome if c not in self.category_mapping
                           and c not in ['omi', 'nonomi', 'control']]
            all_needed = available_detailed + unmapped_cols

            missing_cols = set(all_needed) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Outcome columns not found: {missing_cols}")

            # Extract detailed outcomes
            detailed_outcomes = df[available_detailed].copy()

            # Map to simplified categories
            simplified = map_to_simplified_categories(detailed_outcomes, self.category_mapping)

            # Add any unmapped columns (binary outcomes)
            for col in unmapped_cols:
                simplified[col] = df[col].values

            # Reorder to match col_outcome
            final_order = [c for c in col_outcome if c in simplified.columns]
            outcomes = simplified[final_order].copy()

            logger.info(
                f"Mapped {len(available_detailed)} detailed categories to "
                f"{len([c for c in final_order if c in ['omi', 'nonomi', 'control']])} "
                f"simplified categories"
            )
        else:
            # Standard extraction
            missing_cols = set(col_outcome) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Outcome columns not found: {missing_cols}")

            # Select only outcome columns in specified order
            outcomes = df[col_outcome].copy()

        return outcomes

    def preload_to_memory(self, mask: np.ndarray) -> None:
        """
        Load selected traces into RAM for faster access during training.

        This method pre-loads the ECG traces for the selected indices into memory,
        eliminating repeated HDF5 disk reads. Indices are sorted before reading
        to ensure efficient sequential HDF5 access.

        Args:
            mask: Boolean mask indicating which records to load into memory

        Note:
            After calling this method, getbatch() will use the pre-loaded data
            instead of reading from HDF5. The HDF5 file is closed to free resources.
        """
        indices = sorted(np.nonzero(mask)[0])
        n_samples = len(indices)

        # Estimate memory usage
        sample_shape = self.traces[0].shape
        dtype = self.traces.dtype
        bytes_per_sample = np.prod(sample_shape) * np.dtype(dtype).itemsize
        total_gb = (n_samples * bytes_per_sample) / (1024**3)
        logger.info(f"Pre-loading {n_samples} samples to RAM (~{total_gb:.2f} GB)")

        # Load traces into memory with sorted indices for efficient HDF5 read
        self.traces_memory = self.traces[indices]

        # Convert outcomes/age/male to numpy arrays for faster indexing
        self.outcomes_np = self.outcomes.values
        self.age_np = self.age.values
        self.male_np = self.male.values

        # Store the index mapping: original index -> memory index
        self.memory_index_map = {orig: mem for mem, orig in enumerate(indices)}
        self.using_memory = True

        # Close HDF5 file since we no longer need it
        if self.f is not None:
            self.f.close()
            self.f = None
            self.traces = None

        logger.info(f"Pre-loaded {n_samples} samples to RAM successfully")

    def __del__(self) -> None:
        """Close HDF5 file on cleanup."""
        if hasattr(self, 'f') and self.f is not None:
            try:
                self.f.close()
            except Exception:
                pass  # Ignore errors during cleanup
    
    def __len__(self) -> int:
        """Get total number of records."""
        return len(self.outcomes.index)
    
    def getbatch(
        self,
        extract_indx: Optional[Union[List[int], np.ndarray, range]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fetch a batch of ECG traces and corresponding data.

        Args:
            extract_indx: Indices of records to extract (default: all)

        Returns:
            Tuple of (traces, outcomes, age, sex) arrays
        """
        if extract_indx is None:
            extract_indx = range(len(self))

        if getattr(self, 'using_memory', False):
            # Use pre-loaded data from RAM
            mem_indices = [self.memory_index_map[i] for i in extract_indx]
            return (
                self.traces_memory[mem_indices],
                self.outcomes_np[extract_indx],
                self.age_np[extract_indx],
                self.male_np[extract_indx]
            )
        else:
            # Read from HDF5 (original behavior)
            return (
                self.traces[extract_indx],
                self.outcomes.iloc[extract_indx].values,
                self.age.iloc[extract_indx].values,
                self.male.iloc[extract_indx].values
            )


class BatchDataloader:
    """
    Iterator for loading data in batches.
    
    Provides an iterable interface for batch-wise data loading,
    suitable for training loops with progress bars.
    
    Attributes:
        dset: Source OMIDataset
        batch_size: Number of samples per batch
        all_indx: Indices of records to iterate over
        n_batches: Total number of batches
        
    Example:
        >>> dset = OMIDataset(...)
        >>> loader = BatchDataloader(dset, batch_size=32, mask=dset.train)
        >>> for traces, outcomes, age, sex in loader:
        ...     # Process batch
        ...     pass
    """
    
    def __init__(
        self,
        dset: OMIDataset,
        batch_size: int,
        mask: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize the batch dataloader.
        
        Args:
            dset: Source dataset
            batch_size: Number of samples per batch
            mask: Boolean mask to select subset of records (default: all)
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        self.dset = dset
        self.batch_size = batch_size
        
        if mask is None:
            mask = np.ones(len(dset), dtype=bool)
        
        self.all_indx = np.nonzero(mask)[0].tolist()
        self.current_batch = 0
        
        # Ceiling division for number of batches
        self.n_batches = -(-len(self.all_indx) // self.batch_size)
    
    def __iter__(self) -> 'BatchDataloader':
        """Return iterator (self)."""
        self.current_batch = 0
        return self
    
    def __next__(self) -> List[torch.Tensor]:
        """
        Get next batch of data.
        
        Returns:
            List of tensors: [traces, outcomes, age, sex]
            
        Raises:
            StopIteration: When all batches have been processed
        """
        self.current_batch += 1
        
        if self.current_batch > self.n_batches:
            self.current_batch = 0
            raise StopIteration
        
        # Calculate batch indices
        start = (self.current_batch - 1) * self.batch_size
        end = min(self.current_batch * self.batch_size, len(self.all_indx))
        current_indx = self.all_indx[start:end]
        
        # Fetch batch data
        batch = self.dset.getbatch(current_indx)
        
        # Convert to float32 tensors
        return [torch.tensor(b, dtype=torch.float32) for b in batch]
    
    def __len__(self) -> int:
        """Get total number of batches (for progress bars)."""
        return self.n_batches


def test() -> None:
    """Test the batch dataloader functionality."""
    # Note: This test requires actual data files
    dset = OMIDataset(
        '/proj/sens2020005/omi/data/omi_analysis_data_20250217.hdf5',
        '/proj/sens2020005/omi/data/omi_analysis_data_20250217.txt',
        ['control_nomyoperi', 'control_myoperi']
    )
    
    # Create subset mask: 53 records, 10 per batch, 6 batches total
    incl = np.zeros(len(dset), dtype=bool)
    incl[:50] = True
    incl[[88, 90, 92]] = True
    
    loader = BatchDataloader(dset, batch_size=10, mask=incl)
    
    for traces, outcomes, age, sex in loader:
        print("Batch loaded:")
        print(f"  traces: {traces.shape}")
        print(f"  outcomes: {outcomes.shape}")
        print(f"  age: {age.shape}")
        print(f"  sex: {sex.shape}")


if __name__ == "__main__":
    test()

# -*- coding: utf-8 -*-
"""
Utility functions for training, logging, and reproducibility.

Based on code by Stefan Gustafsson (stefan.gustafsson@medsci.uu.se) for the OMI model.

This module provides:
    - Seed management for reproducible training
    - Output folder management
    - Configuration saving
    - Logging (TensorBoard and CSV)
"""

import datetime
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def seed_everything(
    seed: int,
    deterministic: bool = True,
    warn_only: bool = False
) -> int:
    """
    Set seeds for all random number generators for reproducible results.
    
    Sets seeds for Python's random module, NumPy, and PyTorch (CPU and CUDA).
    Optionally enables deterministic algorithms in PyTorch for full reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, enable PyTorch deterministic algorithms.
                      This may reduce performance but ensures reproducibility.
                      Default: True
        warn_only: If True, only warn (don't error) when deterministic
                  operations aren't available. Default: False
                  
    Returns:
        The seed value used
        
    Note on Reproducibility:
        Full reproducibility requires:
        1. Same seed
        2. Same hardware (GPU model, number of GPUs)
        3. Same software versions (PyTorch, CUDA, cuDNN)
        4. deterministic=True (may impact performance)
        
        Even with all these, some operations may have non-deterministic
        implementations. See PyTorch reproducibility docs for details.
        
    Example:
        >>> seed_everything(42, deterministic=True)
        42
    """
    seed = int(seed)
    
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Enable deterministic algorithms for reproducibility
    if deterministic:
        # Use deterministic algorithms where available
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        
        # cuDNN settings for reproducibility
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    logger.info(
        f"Set random seed to {seed} (deterministic={deterministic})"
    )
    
    return seed


def set_output_folder(
    main_folder: str,
    prefix: str = 'run'
) -> str:
    """
    Create a timestamped output folder for training artifacts.
    
    Creates a new folder with format: {prefix}_{YYYYMMDD_HHMMSS}
    in the specified main folder.
    
    Args:
        main_folder: Parent directory for output folders
        prefix: Prefix for the folder name (default: 'run')
        
    Returns:
        Full path to the created output folder
        
    Example:
        >>> set_output_folder('/experiments', prefix='train')
        '/experiments/train_20240115_143052'
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    folder_name = f"{prefix}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    output_folder = os.path.join(main_folder, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    logger.info(f"Created output folder: {output_folder}")
    return output_folder


def save_config(args: Any, filepath: str) -> None:
    """
    Save configuration/arguments to a JSON file.
    
    Args:
        args: Namespace or object with configuration (converted via vars())
        filepath: Full path to output JSON file
        
    Note:
        Non-serializable values will be converted to strings.
    """
    config_dict = vars(args) if hasattr(args, '__dict__') else dict(args)
    
    # Handle non-serializable types
    serializable_dict = {}
    for key, value in config_dict.items():
        try:
            json.dumps(value)
            serializable_dict[key] = value
        except (TypeError, ValueError):
            serializable_dict[key] = str(value)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_dict, f, indent=4)
    
    logger.info(f"Saved configuration to {filepath}")


def save_dset(data: pd.DataFrame, filepath: str) -> None:
    """
    Save a pandas DataFrame to CSV.
    
    Resets the index and saves without the pandas index column.
    
    Args:
        data: DataFrame to save
        filepath: Full path to output CSV file
    """
    data.reset_index().to_csv(filepath, index=False)
    logger.debug(f"Saved dataset to {filepath}")


def net_param_map(n_blocks: int) -> Tuple[List[int], List[int]]:
    """
    Get predefined ResNet architecture parameters by number of blocks.
    
    Provides tested configurations for different network depths.
    Larger networks have more capacity but require more data and compute.
    
    Args:
        n_blocks: Number of residual blocks (2, 4, 8, or 16)
        
    Returns:
        Tuple of (filter_sizes, sequence_lengths) for each layer
        
    Raises:
        ValueError: If n_blocks is not in the predefined set
        
    Architecture Details:
        - 2 blocks: Lightweight, fast training
        - 4 blocks: Balanced (recommended default)
        - 8 blocks: High capacity
        - 16 blocks: Very deep, requires large datasets
    """
    # Predefined architectures mapping depth to layer configurations
    architectures = {
        2: {
            'filter_size': [64, 196, 320],
            'seq_length': [4096, 256, 16]
        },
        4: {
            'filter_size': [64, 128, 196, 256, 320],
            'seq_length': [4096, 1024, 256, 64, 16]
        },
        8: {
            'filter_size': [64, 128, 128, 196, 256, 256, 320, 512, 512],
            'seq_length': [4096, 2048, 1024, 512, 256, 128, 64, 32, 16]
        },
        16: {
            'filter_size': [64, 64, 64, 64, 64, 128, 128, 128, 128, 
                          256, 256, 256, 256, 512, 512, 512, 512],
            'seq_length': [4096, 2048, 2048, 1024, 1024, 512, 512, 256, 256,
                          128, 128, 64, 64, 32, 32, 16, 16]
        }
    }
    
    if n_blocks not in architectures:
        raise ValueError(
            f"n_blocks must be one of {list(architectures.keys())}, got {n_blocks}"
        )
    
    config = architectures[n_blocks]
    return config['filter_size'], config['seq_length']


class Logger:
    """
    Unified logging to TensorBoard and CSV files.
    
    Handles logging of training metrics, validation metrics, and predictions
    to both TensorBoard (for visualization) and CSV (for analysis).
    
    Attributes:
        folder: Output directory for log files
        logger: TensorBoard SummaryWriter instance
        history: DataFrame accumulating logged metrics
        
    Example:
        >>> log = Logger('/experiments/run_001')
        >>> log.init_tensorboardlog()
        >>> log.init_csvlog()
        >>> log.tensorboardlog_tofile({'loss': 0.5, 'accuracy': 0.9}, epoch=1)
        >>> log.csvlog_tofile({'loss': 0.5, 'accuracy': 0.9}, 'metrics.csv')
    """
    
    def __init__(self, log_folder: str) -> None:
        """
        Initialize logger with output folder.
        
        Args:
            log_folder: Directory for log files
        """
        self.folder = log_folder
        self._tensorboard_writer: Optional[SummaryWriter] = None
        self.history: pd.DataFrame = pd.DataFrame()
        
        # Ensure folder exists
        os.makedirs(log_folder, exist_ok=True)
    
    def init_tensorboardlog(self, suffix: str = '') -> None:
        """
        Initialize TensorBoard writer.
        
        Args:
            suffix: Optional suffix for log file naming (e.g., ensemble number)
        """
        self._tensorboard_writer = SummaryWriter(
            log_dir=self.folder,
            filename_suffix=suffix
        )
        logger.debug(f"Initialized TensorBoard logging in {self.folder}")
    
    def tensorboardlog_tofile_single(
        self,
        metric_name: str,
        metric_val: float,
        step: int
    ) -> None:
        """
        Log a single metric to TensorBoard.
        
        Args:
            metric_name: Name of the metric
            metric_val: Metric value
            step: Training step (usually epoch number)
        """
        if self._tensorboard_writer is None:
            raise RuntimeError("TensorBoard writer not initialized. Call init_tensorboardlog() first.")
        
        self._tensorboard_writer.add_scalar(metric_name, metric_val, step)
    
    def tensorboardlog_tofile(
        self,
        metric_data: Dict[str, float],
        step: int
    ) -> None:
        """
        Log multiple metrics to TensorBoard.
        
        Metrics are automatically organized into groups based on naming:
        - Metrics containing 'loss' go to 'loss/' group
        - Other metrics go to 'other/' group
        
        Args:
            metric_data: Dictionary of metric names to values
            step: Training step (usually epoch number)
        """
        for metric_name, metric_val in metric_data.items():
            # Skip non-numeric values
            if not isinstance(metric_val, (int, float, np.number)):
                continue
            
            # Organize metrics into groups
            if 'loss' in metric_name.lower():
                prefixed_name = f'loss/{metric_name}'
            elif 'auroc' in metric_name.lower() or 'auc' in metric_name.lower():
                prefixed_name = f'metrics/{metric_name}'
            elif 'lr' in metric_name.lower() or 'learning' in metric_name.lower():
                prefixed_name = f'train/{metric_name}'
            else:
                prefixed_name = f'other/{metric_name}'
            
            self.tensorboardlog_tofile_single(prefixed_name, metric_val, step)
    
    def init_csvlog(self) -> None:
        """Initialize CSV logging by resetting the history DataFrame."""
        self.history = pd.DataFrame()
    
    def csvlog_tofile(
        self,
        new_log_data: Dict[str, Any],
        outfile: str
    ) -> None:
        """
        Append metrics to history and save to CSV.
        
        Args:
            new_log_data: Dictionary of metric names to values
            outfile: Output filename (relative to log folder)
        """
        self.history = pd.concat(
            [self.history, pd.DataFrame([new_log_data])],
            ignore_index=True
        )
        
        output_path = os.path.join(self.folder, outfile)
        self.history.to_csv(output_path, index=False)
    
    def predictions_tofile(
        self,
        pred: Union[pd.DataFrame, np.ndarray],
        file_prefix: str,
        epoch_nr: Optional[int] = None,
        ensemble_nr: Optional[int] = None
    ) -> None:
        """
        Save predictions to CSV file.
        
        Args:
            pred: Predictions as DataFrame or array
            file_prefix: Base filename (without extension)
            epoch_nr: Optional epoch number to include in filename
            ensemble_nr: Optional ensemble number to include in filename
        """
        # Build filename
        parts = [file_prefix]
        if ensemble_nr is not None:
            parts.append(str(ensemble_nr))
        if epoch_nr is not None:
            parts.append(str(epoch_nr))
        outfile = '_'.join(parts) + '.csv'
        
        # Convert array to DataFrame if needed
        if isinstance(pred, np.ndarray):
            pred = pd.DataFrame(pred)
        
        output_path = os.path.join(self.folder, outfile)
        save_dset(pred, output_path)
    
    def close(self) -> None:
        """Close TensorBoard writer and flush all pending logs."""
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.flush()
            self._tensorboard_writer.close()
            self._tensorboard_writer = None
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> None:
    """
    Configure logging for the training pipeline.
    
    Sets up logging to both console and optionally a file.
    
    Args:
        log_file: Optional path to log file
        level: Logging level (default: INFO)
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

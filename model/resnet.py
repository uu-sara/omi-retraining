# -*- coding: utf-8 -*-
"""
1D ResNet architecture for ECG signal classification.

This module implements a ResNet-style convolutional neural network adapted for
1D signals (ECG). The architecture follows the principles from:

References:
    [1] K. He et al., "Identity Mappings in Deep Residual Networks,"
        arXiv:1603.05027, 2016. https://arxiv.org/pdf/1603.05027.pdf
    [2] K. He et al., "Deep Residual Learning for Image Recognition,"
        CVPR 2016. https://arxiv.org/pdf/1512.03385.pdf
"""
__author__ = "Stefan Gustafsson"
__email__ = "stefan.gustafsson@medsci.uu.se"

from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import numpy as np


def _compute_padding(downsample: int, kernel_size: int) -> int:
    """
    Compute required padding for a convolutional layer.
    
    Args:
        downsample: Downsampling factor (stride)
        kernel_size: Size of the convolutional kernel
        
    Returns:
        Padding size to maintain dimension consistency
    """
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


def _compute_downsample(n_samples_in: int, n_samples_out: int) -> int:
    """
    Compute downsample rate between consecutive layers.
    
    Args:
        n_samples_in: Number of input samples
        n_samples_out: Number of output samples
        
    Returns:
        Downsample factor (stride)
        
    Raises:
        ValueError: If output samples >= input samples or not an integer factor
    """
    if n_samples_out > n_samples_in:
        raise ValueError(
            f"Number of output samples ({n_samples_out}) must be less than "
            f"input samples ({n_samples_in})"
        )
    
    if n_samples_in % n_samples_out != 0:
        raise ValueError(
            f"Input samples ({n_samples_in}) must be divisible by "
            f"output samples ({n_samples_out})"
        )
    
    downsample = n_samples_in // n_samples_out
    return downsample


def _get_activation(name: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        name: Activation function name ('ReLU', 'ELU', 'GELU')
        
    Returns:
        PyTorch activation module
    """
    activations = {
        'ReLU': nn.ReLU,
        'ELU': nn.ELU,
        'GELU': nn.GELU,
    }
    
    if name not in activations:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from: {list(activations.keys())}"
        )
    
    return activations[name]()


class ResBlock1d(nn.Module):
    """
    Residual block for 1D signals.
    
    Implements a two-layer residual block with:
        - Conv -> BatchNorm -> Activation -> Dropout -> Conv -> Add Skip -> BatchNorm -> Activation -> Dropout
    
    The skip connection handles both spatial downsampling (via MaxPool) and
    channel dimension changes (via 1x1 convolution).
    
    Attributes:
        conv1: First convolutional layer
        bn1: First batch normalization
        conv2: Second convolutional layer (with optional downsampling)
        bn2: Second batch normalization
        activation: Activation function
        dropout1, dropout2: Dropout layers
        skip_connection: Skip connection transformation (if needed)
    """
    
    def __init__(
        self,
        n_filters_in: int,
        n_filters_out: int,
        downsample: int,
        kernel_size: int,
        dropout_rate: float,
        activation: str
    ) -> None:
        """
        Initialize the residual block.
        
        Args:
            n_filters_in: Number of input channels
            n_filters_out: Number of output channels
            downsample: Downsampling factor for spatial dimension
            kernel_size: Size of convolutional kernels (must be odd)
            dropout_rate: Dropout probability (0 to 1)
            activation: Activation function name
            
        Raises:
            ValueError: If kernel_size is even
        """
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd for symmetric padding, got {kernel_size}"
            )
        
        super().__init__()
        
        # First convolution (no downsampling)
        padding1 = _compute_padding(1, kernel_size)
        self.conv1 = nn.Conv1d(
            n_filters_in, n_filters_out, kernel_size,
            padding=padding1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.activation = _get_activation(activation)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second convolution (with downsampling)
        padding2 = _compute_padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(
            n_filters_out, n_filters_out, kernel_size,
            stride=downsample, padding=padding2, bias=False
        )
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Build skip connection
        self.skip_connection = self._build_skip_connection(
            n_filters_in, n_filters_out, downsample
        )
    
    def _build_skip_connection(
        self,
        n_filters_in: int,
        n_filters_out: int,
        downsample: int
    ) -> Optional[nn.Module]:
        """
        Build skip connection layers if transformation is needed.
        
        Args:
            n_filters_in: Input channels
            n_filters_out: Output channels
            downsample: Spatial downsampling factor
            
        Returns:
            Sequential module for skip connection, or None if identity
        """
        layers = []
        
        # Handle spatial downsampling
        if downsample > 1:
            layers.append(nn.MaxPool1d(downsample, stride=downsample))
        
        # Handle channel dimension change
        if n_filters_in != n_filters_out:
            layers.append(nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False))
        
        if layers:
            return nn.Sequential(*layers)
        return None
    
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the residual block.
        
        This implementation uses a pre-activation variant where the skip
        connection is added before batch normalization of the second layer.
        
        Args:
            x: Input tensor for the main path (batch, channels, length)
            y: Input tensor for the skip path (batch, channels, length)
            
        Returns:
            Tuple of (output, skip_output) for the next block
        """
        # Transform skip connection
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        
        # First layer: Conv -> BN -> Activation -> Dropout
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        # Second layer: Conv -> Add Skip -> BN -> Activation -> Dropout
        x = self.conv2(x)
        x = x + y  # Add skip connection
        y = x  # Save for next block's skip
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        
        return x, y


class ResNet1d(nn.Module):
    """
    1D Residual Network for ECG signal classification.
    
    Architecture:
        Input -> Conv -> BN -> Activation -> [ResBlocks] -> BN -> Activation -> Dropout -> Output
    
    The network progressively reduces the spatial dimension while increasing
    the number of feature channels, learning hierarchical representations
    from local patterns to global ECG characteristics.
    
    Attributes:
        conv1: Initial convolutional layer
        bn1: Initial batch normalization
        activation: Activation function
        res_blocks: ModuleList of residual blocks
        bn2: Final batch normalization
        dropout: Final dropout layer
    """
    
    def __init__(
        self,
        input_dim: Tuple[int, int],
        blocks_dim: List[Tuple[int, int]],
        n_outcomes: int,
        kernel_size: int = 17,
        dropout_rate: float = 0.8,
        activation: str = 'ReLU'
    ) -> None:
        """
        Initialize the ResNet1d.
        
        Args:
            input_dim: Input dimensions as (n_channels, n_samples)
                       e.g., (8, 4096) for 8-lead ECG with 4096 samples
            blocks_dim: List of (n_filters, n_samples) for each residual block.
                        Defines the progressive feature extraction.
            n_outcomes: Number of output classes (unused in forward, for future use)
            kernel_size: Kernel size for convolutions (must be odd, default: 17)
            dropout_rate: Dropout probability (default: 0.8)
            activation: Activation function name (default: 'ReLU')
            
        Example:
            >>> model = ResNet1d(
            ...     input_dim=(8, 4096),
            ...     blocks_dim=[(64, 4096), (128, 1024), (196, 256), (256, 64), (320, 16)],
            ...     n_outcomes=5
            ... )
        """
        super().__init__()
        
        # Store configuration
        self._input_dim = input_dim
        self._blocks_dim = blocks_dim
        self._n_outcomes = n_outcomes
        
        # Initial convolution from input channels to first block
        n_filters_in, n_samples_in = input_dim
        n_filters_out, n_samples_out = blocks_dim[0]
        downsample = _compute_downsample(n_samples_in, n_samples_out)
        padding = _compute_padding(downsample, kernel_size)
        
        self.conv1 = nn.Conv1d(
            n_filters_in, n_filters_out, kernel_size,
            stride=downsample, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.activation = _get_activation(activation)
        
        # Build residual blocks using ModuleList for proper registration
        self.res_blocks = nn.ModuleList()
        
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _compute_downsample(n_samples_in, n_samples_out)
            
            block = ResBlock1d(
                n_filters_in=n_filters_in,
                n_filters_out=n_filters_out,
                downsample=downsample,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
                activation=activation
            )
            self.res_blocks.append(block)
        
        # Final layers
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet.
        
        Args:
            x: Input ECG tensor of shape (batch_size, n_channels, n_samples)
            
        Returns:
            Feature tensor of shape (batch_size, final_filters, final_length)
            
        Note:
            Output is not flattened - this is typically done in the parent model
            to allow for flexible feature combination (e.g., with age/sex encoding).
        """
        # Initial layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # Residual blocks with skip connections
        y = x
        for block in self.res_blocks:
            x, y = block(x, y)
        
        # Final layers
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x
    
    def get_output_dim(self) -> Tuple[int, int]:
        """
        Get the output dimensions of the network.
        
        Returns:
            Tuple of (n_filters, n_samples) for the final layer output
        """
        return self._blocks_dim[-1]

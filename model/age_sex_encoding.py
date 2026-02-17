# -*- coding: utf-8 -*-
"""
Age and sex feature encoding for the ECG model.

Based on code by Stefan Gustafsson (stefan.gustafsson@medsci.uu.se) for the OMI model.

This module provides encoding of demographic features (age, sex) to be
combined with ECG features for improved classification. The encoding
learns a non-linear transformation of the low-dimensional demographic
input into a higher-dimensional feature space.

Important Note on Input Normalization:
    Age should be normalized (mean-centered and scaled to unit variance)
    before being passed to this encoding. If raw age values (e.g., 65 years)
    are used alongside binary sex (0/1), the age gradients will dominate
    training, leading to poor learning of sex-related features.
"""

import torch
import torch.nn as nn


# Default output dimension for age/sex encoding
# This is a hyperparameter that controls the representation capacity
DEFAULT_AGESEX_DIM = 64


class AgeSexEncoding(nn.Module):
    """
    Encode age and sex into a learned feature representation.
    
    Takes 2-dimensional input (age, sex) and projects it into a higher
    dimensional space using a linear layer followed by ReLU activation.
    This allows the model to learn non-linear relationships between
    demographics and ECG patterns.
    
    Architecture:
        Input (2) -> Linear -> ReLU -> Output (output_dim)
    
    Attributes:
        output_dim: Dimension of the encoded output
        linear: Linear projection layer
        relu: ReLU activation
        
    Expected Input Format:
        - Shape: (batch_size, 2)
        - Column 0: Sex indicator (0 or 1, typically 1 for male)
        - Column 1: Normalized age (mean-centered, unit variance)
        
    Example:
        >>> encoder = AgeSexEncoding(output_dim=64)
        >>> # Normalized inputs: sex=[1, 0], age=[0.5, -0.3] (standardized)
        >>> age_sex = torch.tensor([[1, 0.5], [0, -0.3]])
        >>> features = encoder(age_sex)
        >>> features.shape
        torch.Size([2, 64])
    """
    
    def __init__(self, output_dim: int = DEFAULT_AGESEX_DIM) -> None:
        """
        Initialize the age/sex encoding layer.
        
        Args:
            output_dim: Dimension of the output encoding (default: 64)
                        Higher values provide more representation capacity
                        but increase parameter count.
        """
        super().__init__()
        
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        
        self.output_dim = output_dim
        
        # Linear projection from 2D (age, sex) to output_dim
        self.linear = nn.Linear(2, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, age_sex: torch.Tensor) -> torch.Tensor:
        """
        Encode age and sex features.
        
        Args:
            age_sex: Input tensor of shape (batch_size, 2)
                     Expected columns: [sex, age] where age is normalized
                     
        Returns:
            Encoded features of shape (batch_size, output_dim)
            
        Raises:
            ValueError: If input doesn't have shape (batch_size, 2)
        """
        if age_sex.dim() != 2 or age_sex.size(1) != 2:
            raise ValueError(
                f"age_sex must have shape (batch_size, 2), got {age_sex.shape}"
            )
        
        out = self.linear(age_sex)
        out = self.relu(out)
        return out

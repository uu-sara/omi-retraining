# -*- coding: utf-8 -*-
"""
Final prediction layer for the ECG model.

Based on code by Stefan Gustafsson (stefan.gustafsson@medsci.uu.se) for the OMI model.

This module provides the linear layer that maps from the combined
feature representation (ECG features + age/sex encoding) to the
output logits for classification.
"""

import torch
import torch.nn as nn


class LinearPredictionStage(nn.Module):
    """
    Final linear layer mapping combined features to output logits.
    
    This is a simple fully-connected layer that produces raw logits
    (linear predictions). Activation functions (sigmoid/softmax) should
    be applied downstream based on the loss function and prediction task.
    
    Attributes:
        lin_classifier: Linear transformation layer
        
    Example:
        >>> stage = LinearPredictionStage(prev_layer_dim=384, n_outcomes=5)
        >>> features = torch.randn(32, 384)  # batch of 32
        >>> logits = stage(features)
        >>> logits.shape
        torch.Size([32, 5])
    """
    
    def __init__(self, prev_layer_dim: int, n_outcomes: int) -> None:
        """
        Initialize the linear prediction layer.
        
        Args:
            prev_layer_dim: Dimension of input features (from ResNet + age/sex)
            n_outcomes: Number of output classes/outcomes
        """
        super().__init__()
        
        if prev_layer_dim <= 0:
            raise ValueError(f"prev_layer_dim must be positive, got {prev_layer_dim}")
        if n_outcomes <= 0:
            raise ValueError(f"n_outcomes must be positive, got {n_outcomes}")
        
        self.prev_layer_dim = prev_layer_dim
        self.n_outcomes = n_outcomes
        self.lin_classifier = nn.Linear(prev_layer_dim, n_outcomes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear layer.
        
        Args:
            x: Input features of shape (batch_size, prev_layer_dim)
            
        Returns:
            Logits of shape (batch_size, n_outcomes)
            
        Note:
            Returns raw logits (X @ W^T + b). Apply sigmoid or softmax
            downstream to convert to probabilities.
        """
        return self.lin_classifier(x)

# -*- coding: utf-8 -*-
"""
Code to setup an instance of ResNet1d for ECG classification.
Ensemble version also defined for combining multiple model predictions.

This module provides:
    - ECGModel: Single model for ECG classification with age/sex encoding
    - EnsembleECGModel: Ensemble of ECGModels for improved predictions
"""
__author__ = "Stefan Gustafsson"
__email__ = "stefan.gustafsson@medsci.uu.se"

import os
import logging
from typing import Tuple, List, Optional, Dict, Any

import torch
import torch.nn as nn

from resnet import ResNet1d
from prediction_stage import LinearPredictionStage
from age_sex_encoding import AgeSexEncoding

logger = logging.getLogger(__name__)


class ECGModel(nn.Module):
    """
    ECG classification model combining ResNet1d with optional age and sex features.

    This model processes ECG signals through a 1D ResNet architecture and optionally
    combines the learned representations with encoded age and sex information for
    final classification.

    Attributes:
        resnet: ResNet1d backbone for ECG feature extraction
        age_sex_emb: Encoding layer for age and sex features (None if excluded)
        lin: Final linear prediction layer
        device: Target device for computations
        include_age: Whether age is included in model input
        include_sex: Whether sex is included in model input

    Expected Input Format:
        - ECG: Tensor of shape (batch_size, n_leads, seq_length)
               Default: (batch_size, 8, 4096) for 8 leads at 500Hz for ~8 seconds
        - Age: Normalized age (mean-centered and scaled to unit variance)
               Ignored if include_age=False
        - Sex: Binary indicator (0 or 1, typically 1 for male)
               Ignored if include_sex=False
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize the ECG model.

        Args:
            config: Configuration object containing:
                - n_leads: Number of ECG leads (default: 8)
                - seq_length: Length of ECG sequence (default: 4096)
                - net_filter_size: List of filter sizes for ResNet layers
                - net_seq_length: List of sequence lengths for ResNet layers
                - n_outcomes: Number of output classes
                - kernel_size: Kernel size for convolutions
                - dropout_rate: Dropout rate for regularization
                - activation_function: Activation function name ('ReLU', 'ELU', 'GELU')
                - agesex_dim: Output dimension for age/sex encoding
                - device: Target device ('cpu' or 'cuda:X')
                - include_age: Whether to include age feature (default: True)
                - include_sex: Whether to include sex feature (default: True)
        """
        super().__init__()

        # Store configuration for reference
        self._config = config
        self.device = config.device

        # Feature inclusion flags (default to True for backward compatibility)
        self.include_age = getattr(config, 'include_age', True)
        self.include_sex = getattr(config, 'include_sex', True)
        self.use_demographics = self.include_age or self.include_sex

        # Setup ResNet backbone
        self.resnet = ResNet1d(
            input_dim=(config.n_leads, config.seq_length),
            blocks_dim=list(zip(config.net_filter_size, config.net_seq_length)),
            n_outcomes=config.n_outcomes,
            kernel_size=config.kernel_size,
            dropout_rate=config.dropout_rate,
            activation=config.activation_function
        )

        # Calculate base dimension from ResNet output
        final_resnet_filter_size = config.net_filter_size[-1]
        final_resnet_length = config.net_seq_length[-1]
        ecg_output_dim = final_resnet_filter_size * final_resnet_length

        # Setup age+sex encoding only if demographics are used
        if self.use_demographics:
            self.age_sex_emb = AgeSexEncoding(output_dim=config.agesex_dim)
            combined_output_dim = ecg_output_dim + config.agesex_dim
        else:
            self.age_sex_emb = None
            combined_output_dim = ecg_output_dim

        # Final prediction layer
        self.lin = LinearPredictionStage(
            prev_layer_dim=combined_output_dim,
            n_outcomes=config.n_outcomes
        )
    
    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through the ECG model.

        Args:
            inputs: Tuple of (age_sex, ecg) where:
                - age_sex: Tensor of shape (batch_size, 2) with [sex, age]
                           Values are ignored if include_age/include_sex are False
                - ecg: Tensor of shape (batch_size, n_leads, seq_length)

        Returns:
            Logits tensor of shape (batch_size, n_outcomes)

        Note:
            Returns raw logits. Apply sigmoid (binary) or softmax (categorical)
            downstream to get predicted probabilities.
        """
        age_sex, ecg = inputs

        # Validate input shapes
        self._validate_inputs(age_sex, ecg)

        # Process ECG through ResNet
        x_ecg = self.resnet(ecg)

        # Flatten ECG features - use flatten for safety with non-contiguous tensors
        x_ecg = torch.flatten(x_ecg, start_dim=1)

        if self.use_demographics:
            # Prepare age_sex tensor based on inclusion flags
            if self.include_age and self.include_sex:
                # Use both age and sex
                age_sex_input = age_sex
            elif self.include_sex and not self.include_age:
                # Only sex: set age to 0
                age_sex_input = age_sex.clone()
                age_sex_input[:, 1] = 0.0
            elif self.include_age and not self.include_sex:
                # Only age: set sex to 0
                age_sex_input = age_sex.clone()
                age_sex_input[:, 0] = 0.0
            else:
                # Neither (shouldn't reach here if use_demographics is True)
                age_sex_input = torch.zeros_like(age_sex)

            # Encode age and sex
            x_age_sex = self.age_sex_emb(age_sex_input)

            # Combine features and predict
            x = torch.cat([x_age_sex, x_ecg], dim=1)
        else:
            # ECG only, no demographics
            x = x_ecg

        logits = self.lin(x)

        return logits
    
    def _validate_inputs(
        self, 
        age_sex: torch.Tensor, 
        ecg: torch.Tensor
    ) -> None:
        """
        Validate input tensor shapes and types.
        
        Args:
            age_sex: Age and sex tensor
            ecg: ECG tensor
            
        Raises:
            ValueError: If input shapes are invalid
        """
        if age_sex.dim() != 2 or age_sex.size(1) != 2:
            raise ValueError(
                f"age_sex must have shape (batch_size, 2), got {age_sex.shape}"
            )
        
        if ecg.dim() != 3:
            raise ValueError(
                f"ecg must have 3 dimensions (batch, leads, length), got {ecg.dim()}"
            )
        
        if age_sex.size(0) != ecg.size(0):
            raise ValueError(
                f"Batch size mismatch: age_sex has {age_sex.size(0)}, "
                f"ecg has {ecg.size(0)}"
            )


class EnsembleECGModel(nn.Module):
    """
    Ensemble of ECG models for improved predictions.

    Combines predictions from multiple trained models using either:
    - Probability averaging (softmax/sigmoid first, then average)
    - Logit averaging (average logits, then softmax/sigmoid)

    Each ensemble member should have been trained with different random seeds
    to provide diversity.

    Attributes:
        model_list: List of loaded ECGModel instances
        device: Target device for computations
        n_ensembles: Number of ensemble members
        aggregation_method: Either "probabilities" or "logits"
        categorical_indices: Indices of categorical outcome columns
        binary_indices: Indices of binary outcome columns

    Note on Averaging Strategies:
        - Probability averaging: Standard approach, bounded [0, 1]
        - Logit averaging: Average raw logits, may provide better calibration
          for well-calibrated individual models
    """

    def __init__(
        self,
        config: Any,
        model_dir: str,
        aggregation_method: str = "probabilities",
        categorical_indices: Optional[List[int]] = None,
        binary_indices: Optional[List[int]] = None
    ) -> None:
        """
        Initialize the ensemble model by loading trained ensemble members.

        Args:
            config: Configuration object (same as ECGModel)
            model_dir: Directory containing trained model files (model_1.pth, etc.)
            aggregation_method: How to combine ensemble predictions:
                - "probabilities": Apply activation first, then average (default)
                - "logits": Average logits first, then apply activation
            categorical_indices: Indices of categorical outcome columns (for softmax)
            binary_indices: Indices of binary outcome columns (for sigmoid)

        Raises:
            FileNotFoundError: If model directory doesn't exist
            RuntimeError: If model files are missing or corrupted
            ValueError: If aggregation_method is invalid
        """
        super().__init__()

        if aggregation_method not in ["probabilities", "logits"]:
            raise ValueError(
                f"aggregation_method must be 'probabilities' or 'logits', "
                f"got '{aggregation_method}'"
            )

        self.device = config.device
        self.n_ensembles = config.n_ensembles
        self.model_dir = model_dir
        self.aggregation_method = aggregation_method
        self.categorical_indices = categorical_indices or []
        self.binary_indices = binary_indices or []
        
        # Validate model directory
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load all ensemble members and register as a ModuleList
        # This ensures proper device handling and state_dict management
        self.model_list = nn.ModuleList(
            self._load_ensembles(config)
        )
        
        # Move entire ensemble to target device
        self.to(self.device)
        
        # Set to evaluation mode by default
        self.eval()
    
    def _load_ensembles(self, config: Any) -> List[ECGModel]:
        """
        Load trained model weights for each ensemble member.
        
        Args:
            config: Model configuration object
            
        Returns:
            List of ECGModel instances with loaded weights
            
        Raises:
            FileNotFoundError: If a model file is missing
            RuntimeError: If model loading fails
        """
        loaded_models = []
        
        for ensemble_nr in range(1, config.n_ensembles + 1):
            model_path = os.path.join(
                self.model_dir, 
                f'model_{ensemble_nr}.pth'
            )
            
            # Check file exists before loading
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"Ensemble model file not found: {model_path}"
                )
            
            try:
                # Create new model instance
                ens_model = ECGModel(config)
                
                # Load trained weights with explicit device mapping
                checkpoint = torch.load(
                    model_path,
                    map_location=self.device,
                    weights_only=True  # Security: only load weights, not arbitrary objects
                )
                
                # Handle both old and new checkpoint formats
                if 'model' in checkpoint:
                    ens_model.load_state_dict(checkpoint['model'])
                else:
                    ens_model.load_state_dict(checkpoint)
                
                # Set to evaluation mode
                ens_model.eval()
                
                loaded_models.append(ens_model)
                logger.info(f"Loaded ensemble member {ensemble_nr} from {model_path}")
                
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load ensemble model {ensemble_nr} from {model_path}: {e}"
                ) from e
        
        return loaded_models
    
    def _apply_activations(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply appropriate activations to logits based on outcome type.

        Args:
            logits: Raw model output of shape (batch_size, n_outcomes)

        Returns:
            Probabilities with softmax applied to categorical and sigmoid to binary
        """
        probs = torch.zeros_like(logits)

        # Apply softmax to categorical outcomes
        if self.categorical_indices:
            cat_indices = self.categorical_indices
            probs[:, cat_indices] = torch.softmax(logits[:, cat_indices], dim=1)

        # Apply sigmoid to binary outcomes
        if self.binary_indices:
            bin_indices = self.binary_indices
            probs[:, bin_indices] = torch.sigmoid(logits[:, bin_indices])

        # If no indices specified, default to sigmoid for all (backward compatibility)
        if not self.categorical_indices and not self.binary_indices:
            probs = torch.sigmoid(logits)

        return probs

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass combining predictions across ensemble members.

        Uses either probability averaging or logit averaging based on
        the aggregation_method setting.

        Args:
            x: Tuple of (age_sex, ecg) tensors (see ECGModel.forward)

        Returns:
            Combined probabilities tensor of shape (batch_size, n_outcomes)
            Values are in [0, 1] range
        """
        # Disable gradient computation for inference efficiency
        with torch.no_grad():
            if self.aggregation_method == "probabilities":
                # Probability averaging: softmax/sigmoid first, then average
                prob_list = []
                for ens_model in self.model_list:
                    logits = ens_model(x)
                    probs = self._apply_activations(logits)
                    prob_list.append(probs)
                stacked_probs = torch.stack(prob_list, dim=0)
                return stacked_probs.mean(dim=0)

            else:  # logits averaging
                # Logit averaging: average logits first, then softmax/sigmoid
                logits_list = []
                for ens_model in self.model_list:
                    logits = ens_model(x)
                    logits_list.append(logits)
                stacked_logits = torch.stack(logits_list, dim=0)
                avg_logits = stacked_logits.mean(dim=0)
                return self._apply_activations(avg_logits)
    
    def forward_with_uncertainty(
        self,
        x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both mean prediction and uncertainty estimate.

        Uncertainty is estimated as the standard deviation of predicted
        probabilities across ensemble members. Higher std indicates greater
        disagreement between models, suggesting higher predictive uncertainty.

        For logit averaging, uncertainty is computed on probabilities derived
        from individual model logits (not from averaged logits).

        Args:
            x: Tuple of (age_sex, ecg) tensors

        Returns:
            Tuple of:
                - mean_probs: Combined probabilities (batch_size, n_outcomes)
                - std_probs: Standard deviation of probabilities (batch_size, n_outcomes)
        """
        prob_list = []
        logits_list = []

        with torch.no_grad():
            for ens_model in self.model_list:
                logits = ens_model(x)
                logits_list.append(logits)
                probs = self._apply_activations(logits)
                prob_list.append(probs)

        stacked_probs = torch.stack(prob_list, dim=0)
        std_probs = stacked_probs.std(dim=0)

        if self.aggregation_method == "probabilities":
            mean_probs = stacked_probs.mean(dim=0)
        else:  # logits averaging
            stacked_logits = torch.stack(logits_list, dim=0)
            avg_logits = stacked_logits.mean(dim=0)
            mean_probs = self._apply_activations(avg_logits)

        return mean_probs, std_probs
    

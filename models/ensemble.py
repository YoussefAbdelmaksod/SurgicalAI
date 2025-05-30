"""
Model ensemble for SurgicalAI.

This module implements ensemble techniques for combining multiple models
to improve prediction accuracy and robustness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import OrderedDict
import os

# Configure logging
logger = logging.getLogger(__name__)

class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module for combining features from different modalities.
    
    This module implements various fusion strategies for combining features
    from multiple modalities (visual, temporal, tool detection, etc.).
    """
    
    def __init__(self, 
                 input_dims: Dict[str, int], 
                 output_dim: int = 256, 
                 fusion_type: str = 'attention',
                 dropout: float = 0.3):
        """
        Initialize multi-modal fusion module.
        
        Args:
            input_dims: Dictionary mapping modality names to their feature dimensions
            output_dim: Dimension of fused output features
            fusion_type: Fusion strategy ('concat', 'attention', 'gated')
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        
        # Create projection layers for each modality
        self.projections = nn.ModuleDict()
        for modality, dim in input_dims.items():
            self.projections[modality] = nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Initialize fusion-specific components
        if fusion_type == 'concat':
            # Simple concatenation followed by projection
            self.fusion_layer = nn.Sequential(
                nn.Linear(output_dim * len(input_dims), output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        elif fusion_type == 'attention':
            # Cross-modal attention
            self.attention_queries = nn.ModuleDict()
            self.attention_keys = nn.ModuleDict()
            self.attention_values = nn.ModuleDict()
            
            for modality in input_dims.keys():
                self.attention_queries[modality] = nn.Linear(output_dim, output_dim)
                self.attention_keys[modality] = nn.Linear(output_dim, output_dim)
                self.attention_values[modality] = nn.Linear(output_dim, output_dim)
            
            self.attention_scale = nn.Parameter(torch.sqrt(torch.tensor(output_dim, dtype=torch.float32)))
            
            # Final projection after attention
            self.fusion_layer = nn.Sequential(
                nn.Linear(output_dim * len(input_dims), output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        elif fusion_type == 'gated':
            # Gated fusion
            self.gates = nn.ModuleDict()
            for modality in input_dims.keys():
                self.gates[modality] = nn.Sequential(
                    nn.Linear(output_dim * len(input_dims), output_dim),
                    nn.Sigmoid()
                )
            
            # Final projection
            self.fusion_layer = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for multi-modal fusion.
        
        Args:
            features: Dictionary mapping modality names to their feature tensors
                     Each tensor should have shape [batch_size, *feature_dims]
            
        Returns:
            Fused features of shape [batch_size, output_dim]
        """
        # Project each modality's features
        projected = {}
        for modality, feature in features.items():
            if modality in self.projections:
                projected[modality] = self.projections[modality](feature)
            else:
                logger.warning(f"Modality {modality} not found in projections")
        
        if self.fusion_type == 'concat':
            # Simple concatenation
            concat_features = torch.cat(list(projected.values()), dim=-1)
            fused = self.fusion_layer(concat_features)
        
        elif self.fusion_type == 'attention':
            # Cross-modal attention
            attended_features = []
            
            for query_modality, query_feat in projected.items():
                # Apply self-attention for each modality
                query = self.attention_queries[query_modality](query_feat)
                
                attended = torch.zeros_like(query_feat)
                attention_weights = []
                
                # Attend to all other modalities
                for key_modality, key_feat in projected.items():
                    key = self.attention_keys[key_modality](key_feat)
                    value = self.attention_values[key_modality](key_feat)
                    
                    # Calculate attention scores
                    scores = torch.matmul(query, key.transpose(-2, -1)) / self.attention_scale
                    weights = F.softmax(scores, dim=-1)
                    attention_weights.append(weights)
                    
                    # Apply attention
                    attended_value = torch.matmul(weights, value)
                    attended = attended + attended_value
                
                attended_features.append(attended)
            
            # Concatenate and fuse
            concat_features = torch.cat(attended_features, dim=-1)
            fused = self.fusion_layer(concat_features)
        
        elif self.fusion_type == 'gated':
            # Gated fusion
            concat_features = torch.cat(list(projected.values()), dim=-1)
            
            weighted_sum = None
            for modality, feat in projected.items():
                # Calculate gate value
                gate = self.gates[modality](concat_features)
                
                # Apply gate
                gated_feat = feat * gate
                
                if weighted_sum is None:
                    weighted_sum = gated_feat
                else:
                    weighted_sum = weighted_sum + gated_feat
            
            # Final projection
            fused = self.fusion_layer(weighted_sum)
        
        return fused


class ModelEnsemble(nn.Module):
    """
    Ensemble of models for improved prediction accuracy.
    
    This module implements various ensemble techniques for combining
    predictions from multiple models of the same type.
    """
    
    def __init__(self, 
                 models: List[nn.Module],
                 ensemble_type: str = 'average',
                 weights: Optional[List[float]] = None):
        """
        Initialize model ensemble.
        
        Args:
            models: List of models to ensemble
            ensemble_type: Ensemble strategy ('average', 'weighted', 'stacking')
            weights: Optional weights for each model in weighted averaging
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.ensemble_type = ensemble_type
        
        # Validate and normalize weights
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError(f"Number of weights ({len(weights)}) does not match number of models ({len(models)})")
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
        elif ensemble_type == 'weighted':
            # Default to equal weights
            self.weights = [1.0 / len(models) for _ in models]
        else:
            self.weights = None
        
        # Initialize stacking layer if needed
        if ensemble_type == 'stacking':
            # Assuming all models have the same output shape
            # This is a simple implementation that works for classification
            # For more complex outputs, a custom stacking layer would be needed
            self.stacking_layer = nn.Linear(len(models), 1)
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for model ensemble.
        
        Args:
            x: Input tensor
            *args, **kwargs: Additional arguments passed to each model
            
        Returns:
            Ensemble prediction
        """
        # Get predictions from each model
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x, *args, **kwargs)
            predictions.append(pred)
        
        # Combine predictions based on ensemble type
        if self.ensemble_type == 'average':
            # Simple averaging
            ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        
        elif self.ensemble_type == 'weighted':
            # Weighted averaging
            weighted_preds = []
            for i, pred in enumerate(predictions):
                weighted_preds.append(pred * self.weights[i])
            ensemble_pred = torch.sum(torch.stack(weighted_preds), dim=0)
        
        elif self.ensemble_type == 'stacking':
            # Stacking (meta-learning)
            # This assumes predictions have shape [batch_size, num_classes] for classification
            stacked_preds = torch.stack(predictions, dim=-1)  # [batch_size, num_classes, num_models]
            ensemble_pred = self.stacking_layer(stacked_preds).squeeze(-1)  # [batch_size, num_classes]
        
        else:
            raise ValueError(f"Unsupported ensemble type: {self.ensemble_type}")
        
        return ensemble_pred
    
    def load_models(self, checkpoint_paths: List[str]) -> None:
        """
        Load weights for all models in the ensemble.
        
        Args:
            checkpoint_paths: List of paths to model checkpoints
        """
        if len(checkpoint_paths) != len(self.models):
            raise ValueError(f"Number of checkpoints ({len(checkpoint_paths)}) does not match number of models ({len(self.models)})")
        
        for i, (model, path) in enumerate(zip(self.models, checkpoint_paths)):
            try:
                state_dict = torch.load(path, map_location='cpu')
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    # Load from training checkpoint
                    model.load_state_dict(state_dict['model_state_dict'])
                else:
                    # Load from state dict directly
                    model.load_state_dict(state_dict)
                logger.info(f"Model {i+1}/{len(self.models)} loaded from {path}")
            except Exception as e:
                logger.error(f"Failed to load model {i+1}/{len(self.models)} from {path}: {e}")
    
    def save_models(self, save_paths: List[str]) -> None:
        """
        Save weights for all models in the ensemble.
        
        Args:
            save_paths: List of paths to save model checkpoints
        """
        if len(save_paths) != len(self.models):
            raise ValueError(f"Number of save paths ({len(save_paths)}) does not match number of models ({len(self.models)})")
        
        for i, (model, path) in enumerate(zip(self.models, save_paths)):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(model.state_dict(), path)
                logger.info(f"Model {i+1}/{len(self.models)} saved to {path}")
            except Exception as e:
                logger.error(f"Failed to save model {i+1}/{len(self.models)} to {path}: {e}") 
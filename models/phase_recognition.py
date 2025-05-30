"""
Surgical phase recognition models for SurgicalAI.

This module implements models for recognizing surgical phases from laparoscopic
video feeds, using Vision Transformer (ViT) and LSTM architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import math
import os
import torchvision.models as models
from timm.models.vision_transformer import VisionTransformer
import timm
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

logger = logging.getLogger(__name__)

# Handle potential import errors with timm
try:
    import timm
    from timm.models.vision_transformer import VisionTransformer
    TIMM_AVAILABLE = True
    logger.info("timm package is available. Using Vision Transformer models.")
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm package not found. Vision Transformer models will not be available.")
    # Create a placeholder class to avoid errors
    class VisionTransformer:
        def __init__(self, *args, **kwargs):
            raise ImportError("timm package not found. Cannot create Vision Transformer models.")

# Mapping of surgical phases
PHASE_MAPPING = {
    0: 'preparation',
    1: 'calot_triangle_dissection',
    2: 'clipping_and_cutting',
    3: 'gallbladder_dissection',
    4: 'gallbladder_packaging',
    5: 'cleaning_and_coagulation',
    6: 'gallbladder_extraction'
}

class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal sequence processing.
    """
    def __init__(self, d_model, max_seq_length=200):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
        
        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class TemporalAttention(nn.Module):
    """
    Temporal attention module for emphasizing important frames in a sequence.
    """
    
    def __init__(self, hidden_size):
        """
        Initialize temporal attention module.
        
        Args:
            hidden_size: Size of the hidden state
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        # Attention layers
        self.attention_query = nn.Linear(hidden_size, hidden_size)
        self.attention_key = nn.Linear(hidden_size, hidden_size)
        self.attention_value = nn.Linear(hidden_size, hidden_size)
        self.attention_scale = nn.Parameter(torch.sqrt(torch.tensor(hidden_size, dtype=torch.float32)))
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Attended tensor of same shape
        """
        # Project to query, key, value
        query = self.attention_query(x)  # [batch_size, seq_len, hidden_size]
        key = self.attention_key(x)  # [batch_size, seq_len, hidden_size]
        value = self.attention_value(x)  # [batch_size, seq_len, hidden_size]
        
        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.attention_scale  # [batch_size, seq_len, seq_len]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, value)  # [batch_size, seq_len, hidden_size]
        
        return attended


class ViTFeatureExtractor(nn.Module):
    """
    Vision Transformer feature extractor for surgical images.
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, freeze=False):
        super().__init__()
        
        self.feature_dim = 768  # Default for base ViT models
        
        if TIMM_AVAILABLE:
            # Use timm's ViT implementation
            self.vit_model = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=0  # Remove classification head
            )
            
            # Get output dimension from the model
            if hasattr(self.vit_model, 'num_features'):
                self.feature_dim = self.vit_model.num_features
        else:
            # Use PyTorch's Vision Transformer
            from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
            
            if pretrained:
                weights = ViT_B_16_Weights.DEFAULT
            else:
                weights = None
                
            vit = vit_b_16(weights=weights)
            
            # Remove the classification head
            self.vit_model = nn.Sequential(*list(vit.children())[:-1])
            self.feature_dim = 768  # Fixed for ViT-B
        
        # Freeze ViT parameters if required
        if freeze:
            for param in self.vit_model.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Extract features from input images.
        
        Args:
            x: Input image batch [batch_size, channels, height, width]
        
        Returns:
            Image features [batch_size, feature_dim]
        """
        return self.vit_model(x)


class ViTLSTM(nn.Module):
    """
    ViT-LSTM model for surgical phase recognition.
    
    This model combines a Vision Transformer for frame-level feature extraction
    with an LSTM for temporal modeling across video frames.
    """
    
    def __init__(self, 
                 num_classes=7, 
                 vit_model='vit_base_patch16_224',
                 hidden_size=512, 
                 num_layers=2, 
                 dropout=0.3, 
                 use_temporal_attention=True, 
                 pretrained=True):
        """
        Initialize ViT-LSTM model.
        
        Args:
            num_classes: Number of phase classes
            vit_model: Name of the ViT model to use
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_temporal_attention: Whether to use temporal attention
            pretrained: Whether to use pretrained ViT weights
        """
        super().__init__()
        
        # Initialize ViT model with error handling
        try:
            if TIMM_AVAILABLE:
                self.vit = timm.create_model(vit_model, pretrained=pretrained)
                # Get ViT feature dimension
                vit_feat_dim = self.vit.head.in_features
                self.vit.head = nn.Identity()  # Remove classification head
            else:
                # Fallback to a basic CNN if timm is not available
                logger.warning("Using ResNet18 as fallback for ViT")
                resnet = models.resnet18(pretrained=False)
                # Remove classification layer
                self.vit = nn.Sequential(*list(resnet.children())[:-1])
                vit_feat_dim = 512  # ResNet18 feature dimension
        except Exception as e:
            logger.warning(f"Error loading pretrained ViT model: {str(e)}")
            logger.warning("Using ResNet18 as fallback for ViT")
            resnet = models.resnet18(pretrained=False)
            # Remove classification layer
            self.vit = nn.Sequential(*list(resnet.children())[:-1])
            vit_feat_dim = 512  # ResNet18 feature dimension
        
        # Project ViT features to LSTM input size
        self.projection = nn.Linear(vit_feat_dim, hidden_size)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Temporal attention if enabled
        self.use_temporal_attention = use_temporal_attention
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(hidden_size * 2)  # *2 for bidirectional
        
        # Classification layer
        self.classifier = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, channels, height, width]
            
        Returns:
            Tuple of (logits, features)
            - logits: Class logits of shape [batch_size, seq_len, num_classes]
            - features: Final features of shape [batch_size, seq_len, hidden_size*2]
        """
        batch_size, seq_len = x.shape[:2]
        
        # Reshape input for feature extraction
        x_reshaped = x.view(batch_size * seq_len, *x.shape[2:])  # [batch_size*seq_len, channels, height, width]
        
        # Extract features with ViT or ResNet
        if isinstance(self.vit, nn.Sequential):  # ResNet fallback
            vit_features = self.vit(x_reshaped)
            vit_features = vit_features.view(batch_size * seq_len, -1)  # Flatten features
        else:  # Original ViT
            vit_features = self.vit(x_reshaped)  # [batch_size*seq_len, vit_feat_dim]
        
        # Project features
        projected_features = self.projection(vit_features)  # [batch_size*seq_len, hidden_size]
        
        # Reshape back to sequence
        sequence_features = projected_features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_size]
        
        # Apply LSTM
        lstm_output, _ = self.lstm(sequence_features)  # [batch_size, seq_len, hidden_size*2]
        
        # Apply temporal attention if enabled
        if self.use_temporal_attention:
            lstm_output = self.temporal_attention(lstm_output)
        
        # Apply dropout
        lstm_output = self.dropout(lstm_output)
        
        # Apply classifier
        logits = self.classifier(lstm_output)  # [batch_size, seq_len, num_classes]
        
        return logits, lstm_output
    
    def extract_features(self, x):
        """
        Extract features without classification.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, channels, height, width]
            
        Returns:
            Features of shape [batch_size, seq_len, hidden_size*2]
        """
        # Forward pass without classification
        _, features = self.forward(x)
        return features


class ViTTransformerTemporal(nn.Module):
    """
    Vision Transformer + Transformer model for surgical phase recognition.
    
    Uses ViT for spatial features and a Transformer encoder for temporal modeling.
    """
    def __init__(self, num_classes=7, vit_model='vit_base_patch16_224', hidden_dim=512,
                 nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 pretrained=True, freeze_vit=False, max_seq_length=100):
        """
        Initialize ViT-Transformer model.
        
        Args:
            num_classes: Number of surgical phases to classify
            vit_model: ViT model name or configuration
            hidden_dim: Hidden dimension of the transformer
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            pretrained: Whether to use pretrained ViT weights
            freeze_vit: Whether to freeze ViT parameters
            max_seq_length: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        # Initialize ViT feature extractor
        self.feature_extractor = ViTFeatureExtractor(
            model_name=vit_model,
            pretrained=pretrained,
            freeze=freeze_vit
        )
        
        # Get feature dimension from extractor
        feature_dim = self.feature_extractor.feature_dim
        
        # Projection layer to transformer hidden dimension
        self.projection = nn.Linear(feature_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize linear layer weights."""
        for m in [self.projection, self.classifier]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x, mask=None):
        """
        Forward pass for phase recognition.
        
        Args:
            x: Input image sequence [batch_size, seq_len, channels, height, width]
            mask: Attention mask for transformer (optional)
            
        Returns:
            Phase logits for each frame [batch_size, seq_len, num_classes]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Process each frame with ViT
        frame_features = []
        for t in range(seq_len):
            features_t = self.feature_extractor(x[:, t])
            frame_features.append(features_t)
        
        # Stack features
        frame_features = torch.stack(frame_features, dim=1)  # [batch_size, seq_len, feature_dim]
        
        # Project to transformer hidden dimension
        proj_features = self.projection(frame_features)
        
        # Add positional encoding
        pos_features = self.positional_encoding(proj_features)
        
        # Process with transformer
        if mask is not None:
            transformer_out = self.transformer_encoder(pos_features, src_key_padding_mask=mask)
        else:
            transformer_out = self.transformer_encoder(pos_features)
        
        # Classify each time step
        output = self.classifier(transformer_out)
        
        return output
    
    def load(self, checkpoint_path):
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        self.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        logger.info(f"Loaded weights from {checkpoint_path}")


def get_vit_lstm_model(config):
    """
    Factory function to create a ViT-LSTM model from config.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized ViT-LSTM model
    """
    return ViTLSTM(
        num_classes=config.get('num_classes', 7),
        vit_model=config.get('vit_model', 'vit_base_patch16_224'),
        hidden_size=config.get('hidden_size', 512),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
        use_temporal_attention=config.get('use_temporal_attention', True),
        pretrained=config.get('pretrained', True)
    )

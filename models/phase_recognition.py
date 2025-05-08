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

logger = logging.getLogger(__name__)

# Try importing timm for ViT models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    logger.warning("timm package not installed. Using PyTorch's built-in Vision Transformer implementation.")
    TIMM_AVAILABLE = False


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
    Temporal attention mechanism for focusing on important frames.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        """
        Apply temporal attention.
        
        Args:
            x: Input sequence [batch_size, seq_len, hidden_dim]
        
        Returns:
            Attended sequence [batch_size, hidden_dim]
        """
        # Calculate attention weights
        attention_weights = self.attention(x)  # [batch_size, seq_len, 1]
        
        # Apply attention weights
        context = torch.sum(attention_weights * x, dim=1)  # [batch_size, hidden_dim]
        
        return context, attention_weights


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
    Vision Transformer + LSTM model for surgical phase recognition.
    
    Combines ViT for spatial feature extraction with LSTM for temporal modeling.
    """
    def __init__(self, num_classes=7, vit_model='vit_base_patch16_224', hidden_size=512, 
                 num_layers=2, dropout=0.3, pretrained=True, bidirectional=True, 
                 freeze_vit=False, use_temporal_attention=True):
        """
        Initialize ViT-LSTM model.
        
        Args:
            num_classes: Number of surgical phases to recognize
            vit_model: ViT model name or configuration
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            pretrained: Whether to use pretrained ViT weights
            bidirectional: Whether to use bidirectional LSTM
            freeze_vit: Whether to freeze ViT parameters
            use_temporal_attention: Whether to use temporal attention mechanism
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
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Define output dimension based on LSTM configuration
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # Temporal attention module
        self.use_temporal_attention = use_temporal_attention
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(lstm_output_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM and linear layer weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for name, param in self.classifier.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x, lengths=None):
        """
        Forward pass for phase recognition.
        
        Args:
            x: Input image sequence [batch_size, seq_len, channels, height, width]
            lengths: Sequence lengths for packed sequence
            
        Returns:
            Phase logits for each frame [batch_size, seq_len, num_classes]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Process each frame with ViT
        frame_features = []
        for t in range(seq_len):
            # Extract features from frame at time t
            features_t = self.feature_extractor(x[:, t])
            frame_features.append(features_t)
        
        # Stack frame features
        frame_features = torch.stack(frame_features, dim=1)  # [batch_size, seq_len, feature_dim]
        
        # Process with LSTM (with packed sequence if lengths are provided)
        if lengths is not None:
            # Pack sequence
            packed_features = nn.utils.rnn.pack_padded_sequence(
                frame_features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # Process with LSTM
            packed_outputs, _ = self.lstm(packed_features)
            
            # Unpack sequence
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            # Process with LSTM directly
            lstm_out, _ = self.lstm(frame_features)
        
        # Apply temporal attention if enabled
        if self.use_temporal_attention:
            context, _ = self.temporal_attention(lstm_out)
            # Repeat context for each time step
            output = self.classifier(context).unsqueeze(1).repeat(1, seq_len, 1)
        else:
            # Classify each time step independently
            output = self.classifier(lstm_out)
        
        return output
    
    def load(self, checkpoint_path):
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        # Load weights
        state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {checkpoint_path}")
        
        return self
        
    def predict(self, sequences, smooth=True, window_size=5):
        """
        Predict surgical phases from video sequences.
        
        Args:
            sequences: Video sequences [batch_size, seq_len, channels, height, width]
            smooth: Whether to apply temporal smoothing to predictions
            window_size: Window size for temporal smoothing
            
        Returns:
            Dict with prediction results
        """
        # Ensure model is in evaluation mode
        self.eval()
        
        with torch.no_grad():
            batch_size, seq_len = sequences.size(0), sequences.size(1)
            
            # Process each frame
            all_outputs = []
            
            for t in range(seq_len):
                # Extract features from frame at time t
                features_t = self.feature_extractor(sequences[:, t])
                all_outputs.append(features_t)
            
            # Stack frame features
            frame_features = torch.stack(all_outputs, dim=1)  # [batch_size, seq_len, feature_dim]
            
            # Create sequence lengths tensor
            sequence_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=sequences.device)
            
            # Forward pass through LSTM
            if self.bidirectional:
                # Use packed sequence if lengths are provided
                packed_features = nn.utils.rnn.pack_padded_sequence(
                    frame_features, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                
                # Process with LSTM
                packed_output, _ = self.lstm(packed_features)
                
                # Unpack sequence
                lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            else:
                # Process with LSTM directly
                lstm_output, _ = self.lstm(frame_features)
            
            # Apply temporal attention if enabled
            if self.use_temporal_attention:
                # Apply attention for each timestep in the sequence
                attended_outputs = []
                
                for t in range(seq_len):
                    # Get sequence up to time t (inclusive)
                    sequence_t = lstm_output[:, :t+1]
                    
                    # Apply attention
                    context_t, _ = self.temporal_attention(sequence_t)
                    
                    # Classify the context
                    output_t = self.classifier(context_t)
                    
                    attended_outputs.append(output_t)
                
                # Stack outputs
                logits = torch.stack(attended_outputs, dim=1)  # [batch_size, seq_len, num_classes]
            else:
                # Apply classifier to each timestep
                logits = self.classifier(lstm_output)  # [batch_size, seq_len, num_classes]
            
            # Apply temporal smoothing if enabled
            if smooth and seq_len > 1:
                smoothed_logits = []
                
                for t in range(seq_len):
                    # Define window start and end
                    start = max(0, t - window_size // 2)
                    end = min(seq_len, t + window_size // 2 + 1)
                    
                    # Average logits in window
                    window_logits = logits[:, start:end].mean(dim=1)
                    smoothed_logits.append(window_logits)
                
                # Stack smoothed logits
                logits = torch.stack(smoothed_logits, dim=1)
            
            # Get probabilities and predicted classes
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            
            # Convert to numpy arrays
            probs_np = probs.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            
            # Create mapping from indices to phase names
            phase_names = [
                'preparation',
                'calot_triangle_dissection',
                'clipping_and_cutting',
                'gallbladder_dissection',
                'gallbladder_packaging',
                'cleaning_and_coagulation',
                'gallbladder_extraction'
            ]
            
            # Create results dictionary
            results = {
                'phase_indices': predictions_np,
                'probabilities': probs_np,
                'phase_names': [[phase_names[idx] for idx in pred] for pred in predictions_np]
            }
            
            return results


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

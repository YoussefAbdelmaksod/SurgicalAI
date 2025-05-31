"""
Surgical mistake detection models for SurgicalAI.

This module implements models for detecting surgical mistakes and assessing
risk levels during laparoscopic procedures, with specific optimizations for
cholecystectomy procedures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
import json
import os
import torchvision.models as models
import timm

logger = logging.getLogger(__name__)

# Try importing transformers for GPT models
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers package not installed. GPT-based models will not be available.")
    TRANSFORMERS_AVAILABLE = False

# Cholecystectomy-specific constants
CHOLECYSTECTOMY_CRITICAL_STRUCTURES = {
    'cystic_duct': 0.85,      # Detection confidence threshold
    'cystic_artery': 0.85,
    'common_bile_duct': 0.90,
    'hepatic_artery': 0.85,
    'gallbladder': 0.75
}

CHOLECYSTECTOMY_CRITICAL_PHASES = {
    'calot_triangle_dissection': 2.0,  # Risk multiplier
    'clipping_and_cutting': 2.0,
    'gallbladder_dissection': 1.5
}

CHOLECYSTECTOMY_SPECIFIC_MISTAKES = {
    'misidentification': {
        'description': 'Potential misidentification of biliary structures',
        'associated_phases': ['calot_triangle_dissection'],
        'risk_level': 'high'
    },
    'clip_placement': {
        'description': 'Improper clip placement on cystic structures',
        'associated_phases': ['clipping_and_cutting'],
        'risk_level': 'high'
    },
    'thermal_injury': {
        'description': 'Potential thermal injury to surrounding structures',
        'associated_phases': ['gallbladder_dissection', 'calot_triangle_dissection'],
        'risk_level': 'medium'
    },
    'traction_injury': {
        'description': 'Excessive traction causing avulsion of structures',
        'associated_phases': ['gallbladder_dissection', 'calot_triangle_dissection'],
        'risk_level': 'medium'
    },
    'poor_exposure': {
        'description': 'Inadequate exposure of surgical field',
        'associated_phases': ['calot_triangle_dissection', 'gallbladder_dissection'],
        'risk_level': 'medium'
    },
    'critical_view': {
        'description': 'Failure to achieve critical view of safety',
        'associated_phases': ['calot_triangle_dissection'],
        'risk_level': 'high'
    }
}

class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module for combining visual and tool features.
    """
    def __init__(self, visual_dim=768, tool_dim=128, segmentation_dim=128, output_dim=256, dropout=0.3):
        """
        Initialize multi-modal fusion module.
        
        Args:
            visual_dim: Dimension of visual features
            tool_dim: Dimension of tool features
            segmentation_dim: Dimension of segmentation features
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Projection layers
        self.visual_projection = nn.Linear(visual_dim, output_dim)
        self.tool_projection = nn.Linear(tool_dim, output_dim)
        self.segmentation_projection = nn.Linear(segmentation_dim, output_dim)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(output_dim, output_dim, dropout)
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, visual_features, tool_features, segmentation_features):
        """
        Forward pass for multi-modal fusion.
        
        Args:
            visual_features: Visual features [batch_size, visual_dim]
            tool_features: Tool features [batch_size, tool_dim]
            segmentation_features: Segmentation features [batch_size, segmentation_dim]
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        # Project features
        visual_proj = self.visual_projection(visual_features)
        tool_proj = self.tool_projection(tool_features)
        segmentation_proj = self.segmentation_projection(segmentation_features)
        
        # Apply cross-modal attention
        visual_attended = self.cross_attention(visual_proj, tool_proj)
        tool_attended = self.cross_attention(tool_proj, segmentation_proj)
        
        # Concatenate and fuse
        concat_features = torch.cat([visual_attended, tool_attended, segmentation_proj], dim=1)
        fused_features = self.fusion_layer(concat_features)
        
        return fused_features

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention module for attending from one modality to another.
    """
    def __init__(self, query_dim, key_dim, dropout=0.1):
        """
        Initialize cross-modal attention module.
        
        Args:
            query_dim: Query dimension
            key_dim: Key dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.scale = query_dim ** 0.5
        
        # Attention layers
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(key_dim, query_dim)
        self.value_proj = nn.Linear(key_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_proj = nn.Linear(query_dim, query_dim)
    
    def forward(self, query, key_value):
        """
        Forward pass for cross-modal attention.
        
        Args:
            query: Query tensor [batch_size, query_dim]
            key_value: Key-value tensor [batch_size, key_dim]
            
        Returns:
            Attended features [batch_size, query_dim]
        """
        # Project query, key, and value
        q = self.query_proj(query).unsqueeze(1)  # [batch_size, 1, query_dim]
        k = self.key_proj(key_value).unsqueeze(1)  # [batch_size, 1, query_dim]
        v = self.value_proj(key_value).unsqueeze(1)  # [batch_size, 1, query_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, 1, 1]
        
        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        attended = torch.matmul(attn, v).squeeze(1)  # [batch_size, query_dim]
        
        # Output projection
        output = self.output_proj(attended)
        
        return output

class TemporalContextProcessor(nn.Module):
    """
    Temporal context processor for capturing temporal dynamics.
    """
    def __init__(self, input_dim=256, hidden_dim=256, num_layers=2, dropout=0.3):
        """
        Initialize temporal context processor.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = hidden_dim * 2  # For bidirectional LSTM
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = TemporalAttention(hidden_dim * 2, dropout)
    
    def forward(self, x, lengths=None):
        """
        Forward pass for temporal context processing.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            lengths: Sequence lengths (optional)
            
        Returns:
            Temporally processed features [batch_size, output_dim]
        """
        # Process with LSTM
        if lengths is not None:
            # Pack sequence
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # Process with LSTM
            packed_output, _ = self.lstm(packed_x)
            
            # Unpack sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, _ = self.lstm(x)
        
        # Apply attention
        attended_output, attention_weights = self.attention(output)
        
        return attended_output


class TemporalAttention(nn.Module):
    """
    Temporal attention module for emphasizing important frames in a sequence.
    """
    
    def __init__(self, feature_dim):
        """
        Initialize temporal attention module.
        
        Args:
            feature_dim: Feature dimension
        """
        super().__init__()
        
        # Attention projection layers
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = nn.Parameter(torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32)))
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, feature_dim]
            
        Returns:
            Attended tensor of shape [batch_size, feature_dim]
        """
        # Project inputs to queries, keys, and values
        queries = self.query(x)  # [batch_size, seq_len, feature_dim]
        keys = self.key(x)  # [batch_size, seq_len, feature_dim]
        values = self.value(x)  # [batch_size, seq_len, feature_dim]
        
        # Calculate attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale  # [batch_size, seq_len, seq_len]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, values)  # [batch_size, seq_len, feature_dim]
        
        # Pool across time dimension to get a single vector
        # This gives us a representation that considers the entire sequence
        attended = torch.mean(attended, dim=1)  # [batch_size, feature_dim]
        
        return attended


class SurgicalMistakeDetector(nn.Module):
    """
    Surgical mistake detection model.
    
    This model identifies potential mistakes and assesses risk levels
    during laparoscopic procedures.
    """
    
    def __init__(self, 
                 visual_dim=768, 
                 tool_dim=128, 
                 hidden_dim=256, 
                 num_classes=3,  # No mistake, low risk, high risk
                 use_temporal=True):
        """
        Initialize mistake detection model.
        
        Args:
            visual_dim: Dimension of visual features
            tool_dim: Dimension of tool features
            hidden_dim: Hidden dimension
            num_classes: Number of mistake classes
            use_temporal: Whether to use temporal context
        """
        super().__init__()
        
        self.visual_dim = visual_dim
        self.tool_dim = tool_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_temporal = use_temporal
        
        # Visual feature extractor (ViT backbone)
        self.visual_extractor = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=True, 
            num_classes=0
        )
        
        # Segmentation feature processor - new addition
        self.use_segmentation = True
        self.segmentation_dim = 128
        self.segmentation_processor = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, self.segmentation_dim)
        )
        
        # Multimodal fusion
        if self.use_segmentation:
            self.fusion = MultiModalFusion(
                visual_dim=visual_dim,
                tool_dim=tool_dim,
                segmentation_dim=self.segmentation_dim,
                output_dim=hidden_dim
            )
        else:
            self.fusion = MultiModalFusion(
                visual_dim=visual_dim,
                tool_dim=tool_dim,
                output_dim=hidden_dim
            )
        
        # Temporal context processor
        if use_temporal:
            self.temporal_processor = TemporalContextProcessor(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, tool_features=None, segmentation_masks=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, channels, height, width]
               or [batch_size, channels, height, width] when use_temporal=False
            tool_features: Optional tool features
            segmentation_masks: Optional segmentation masks
            
        Returns:
            Class logits
        """
        # Process input based on shape
        if self.use_temporal and len(x.shape) == 5:
            batch_size, seq_len, c, h, w = x.shape
            
            # Reshape for feature extraction
            x_reshaped = x.reshape(batch_size * seq_len, c, h, w)
            
            # Extract visual features
            visual_features = self.visual_extractor(x_reshaped)  # [batch_size*seq_len, visual_dim]
            
            # Process tool features if provided
            if tool_features is not None:
                if len(tool_features.shape) == 3:
                    # [batch_size, seq_len, tool_dim]
                    tool_features_reshaped = tool_features.reshape(batch_size * seq_len, -1)
                else:
                    # [batch_size*seq_len, tool_dim]
                    tool_features_reshaped = tool_features
            else:
                # Create zero tensor if tool features not provided
                tool_features_reshaped = torch.zeros(batch_size * seq_len, self.tool_dim, device=x.device)
            
            # Process segmentation masks if provided
            if segmentation_masks is not None and self.use_segmentation:
                if len(segmentation_masks.shape) == 5:
                    # [batch_size, seq_len, num_classes, height, width]
                    seg_batch_size, seg_seq_len = segmentation_masks.shape[:2]
                    seg_reshaped = segmentation_masks.reshape(seg_batch_size * seg_seq_len, 
                                                             segmentation_masks.shape[2],
                                                             segmentation_masks.shape[3],
                                                             segmentation_masks.shape[4])
                else:
                    # [batch_size*seq_len, num_classes, height, width]
                    seg_reshaped = segmentation_masks
                
                # Extract segmentation features
                segmentation_features = self.segmentation_processor(seg_reshaped.float())
            else:
                # Create zero tensor if segmentation masks not provided
                segmentation_features = torch.zeros(batch_size * seq_len, self.segmentation_dim, device=x.device)
            
            # Fuse features
            if self.use_segmentation and segmentation_masks is not None:
                fused_features = self.fusion(visual_features, tool_features_reshaped, segmentation_features)
            else:
                fused_features = self.fusion(visual_features, tool_features_reshaped)
            
            # Reshape back to sequence
            fused_features = fused_features.reshape(batch_size, seq_len, -1)
            
            # Process temporal context
            temporal_features = self.temporal_processor(fused_features)
            
            # Classification
            logits = self.classifier(temporal_features)
            
            return logits
        else:
            # Non-temporal processing (single frame)
            if len(x.shape) == 5:
                batch_size, seq_len, c, h, w = x.shape
                x = x.reshape(batch_size * seq_len, c, h, w)
            
            # Extract visual features
            visual_features = self.visual_extractor(x)
            
            # Process tool features if provided
            if tool_features is not None:
                if len(tool_features.shape) == 3:
                    # Reshape if needed
                    tool_features = tool_features.reshape(tool_features.shape[0] * tool_features.shape[1], -1)
            else:
                # Create zero tensor if tool features not provided
                tool_features = torch.zeros(visual_features.shape[0], self.tool_dim, device=x.device)
            
            # Process segmentation masks if provided
            if segmentation_masks is not None and self.use_segmentation:
                if len(segmentation_masks.shape) == 5:
                    # Reshape if needed
                    seg_batch_size, seg_seq_len = segmentation_masks.shape[:2]
                    segmentation_masks = segmentation_masks.reshape(
                        seg_batch_size * seg_seq_len,
                        segmentation_masks.shape[2],
                        segmentation_masks.shape[3],
                        segmentation_masks.shape[4]
                    )
                
                # Extract segmentation features
                segmentation_features = self.segmentation_processor(segmentation_masks.float())
            else:
                # Create zero tensor if segmentation masks not provided
                segmentation_features = torch.zeros(visual_features.shape[0], self.segmentation_dim, device=x.device)
            
            # Fuse features
            if self.use_segmentation and segmentation_masks is not None:
                fused_features = self.fusion(visual_features, tool_features, segmentation_features)
            else:
                fused_features = self.fusion(visual_features, tool_features)
            
            # Classification
            logits = self.classifier(fused_features)
            
            return logits

    def predict(self, x, tool_features=None, segmentation_masks=None):
        """
        Make predictions.
        
        Args:
            x: Input tensor
            tool_features: Optional tool features
            segmentation_masks: Optional segmentation masks
            
        Returns:
            Predicted class and confidence
        """
        self.eval()
        with torch.no_grad():
            logits = self(x, tool_features, segmentation_masks)
            
            # Get predictions
            if len(logits.shape) == 3:
                # Sequence output
                probs = torch.softmax(logits, dim=2)
                predictions = torch.argmax(probs, dim=2)
                confidence = torch.max(probs, dim=2)[0]
            else:
                # Single frame output
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probs, dim=1)
                confidence = torch.max(probs, dim=1)[0]
            
            return predictions, confidence


class GPTSurgicalAssistant(nn.Module):
    """
    GPT-based model for generating surgical guidance during laparoscopic cholecystectomy.
    """
    def __init__(self, model_name='gpt2', num_visual_tokens=50, num_tool_tokens=20,
                 max_sequence_length=512, device=None):
        """
        Initialize GPT-based surgical assistant.
        
        Args:
            model_name: Name of GPT model
            num_visual_tokens: Number of tokens for visual features
            num_tool_tokens: Number of tokens for tool features
            max_sequence_length: Maximum sequence length
            device: Device to use
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package is required for GPTSurgicalAssistant")
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load GPT model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add special tokens for different modalities
        special_tokens = {
            'additional_special_tokens': [
                '<visual>',
                '</visual>',
                '<tools>',
                '</tools>',
                '<phase>',
                '</phase>',
                '<guidance>',
                '</guidance>'
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Token type embeddings
        self.token_type_embeddings = nn.Embedding(4, self.model.config.n_embd)  # 4 types: text, visual, tool, phase
        
        # Visual feature projection
        self.visual_projection = nn.Linear(768, self.model.config.n_embd * num_visual_tokens)
        self.num_visual_tokens = num_visual_tokens
        
        # Tool feature projection
        self.tool_projection = nn.Linear(128, self.model.config.n_embd * num_tool_tokens)
        self.num_tool_tokens = num_tool_tokens
        
        # Max sequence length
        self.max_sequence_length = max_sequence_length
        
        # Surgical procedure knowledge
        self.procedure_steps = {}
        self.tool_descriptions = {}
    
    def load_procedure_knowledge(self, procedure_file):
        """
        Load surgical procedure knowledge from a JSON file.
        
        Args:
            procedure_file: Path to procedure JSON file
        """
        if not os.path.exists(procedure_file):
            logger.warning(f"Procedure file not found: {procedure_file}")
            return
        
        with open(procedure_file, 'r') as f:
            knowledge = json.load(f)
        
        self.procedure_steps = knowledge.get('procedure_steps', {})
        self.tool_descriptions = knowledge.get('tool_descriptions', {})
        
        logger.info(f"Loaded procedure knowledge with {len(self.procedure_steps)} steps")
    
    def prepare_inputs(self, text_prompt, visual_features=None, tool_features=None, 
                       phase_name=None, max_guidance_length=100):
        """
        Prepare inputs for the model by combining text, visual features, and tool features.
        
        Args:
            text_prompt: Text prompt describing the situation
            visual_features: Visual features from the current frame
            tool_features: Tool detection features
            phase_name: Current surgical phase name
            max_guidance_length: Maximum guidance text length
            
        Returns:
            Dict of model inputs
        """
        # Prepare prompt with special tokens
        full_prompt = text_prompt.strip()
        
        # Add phase information if available
        if phase_name:
            full_prompt += f" <phase>{phase_name}</phase>"
        
        # Add guidance token to indicate where to generate
        full_prompt += " <guidance>"
        
        # Encode prompt
        encoded_prompt = self.tokenizer.encode(full_prompt, return_tensors='pt')
        encoded_prompt = encoded_prompt.to(self.device)
        
        # Calculate token types (0 for text, 1 for visual, 2 for tools, 3 for phase)
        token_types = torch.zeros_like(encoded_prompt)
        
        # Count how many tokens we have so far
        input_ids = encoded_prompt
        current_length = input_ids.size(1)
        
        # Prepare visual features if provided
        visual_token_ids = None
        if visual_features is not None:
            # Project visual features to token embeddings
            visual_features = visual_features.to(self.device)
            visual_tokens = self.visual_projection(visual_features)
            visual_tokens = visual_tokens.view(-1, self.num_visual_tokens, self.model.config.n_embd)
            
            # Create dummy token IDs for visual tokens
            visual_token_ids = torch.full(
                (1, self.num_visual_tokens), 
                self.tokenizer.convert_tokens_to_ids('<visual>'), 
                dtype=torch.long,
                device=self.device
            )
            
            # Update token types
            visual_token_types = torch.full_like(visual_token_ids, 1)  # 1 for visual
            
            # Add visual tokens to inputs
            input_ids = torch.cat([input_ids, visual_token_ids], dim=1)
            token_types = torch.cat([token_types, visual_token_types], dim=1)
            current_length += self.num_visual_tokens
        
        # Prepare tool features if provided
        tool_token_ids = None
        if tool_features is not None:
            # Project tool features to token embeddings
            tool_features = tool_features.to(self.device)
            tool_tokens = self.tool_projection(tool_features)
            tool_tokens = tool_tokens.view(-1, self.num_tool_tokens, self.model.config.n_embd)
            
            # Create dummy token IDs for tool tokens
            tool_token_ids = torch.full(
                (1, self.num_tool_tokens), 
                self.tokenizer.convert_tokens_to_ids('<tools>'), 
                dtype=torch.long,
                device=self.device
            )
            
            # Update token types
            tool_token_types = torch.full_like(tool_token_ids, 2)  # 2 for tools
            
            # Add tool tokens to inputs
            input_ids = torch.cat([input_ids, tool_token_ids], dim=1)
            token_types = torch.cat([token_types, tool_token_types], dim=1)
            current_length += self.num_tool_tokens
        
        # Ensure we don't exceed max sequence length
        if current_length > self.max_sequence_length:
            # Trim from the beginning but keep the guidance token
            guidance_pos = (input_ids == self.tokenizer.convert_tokens_to_ids('<guidance>')).nonzero()
            if guidance_pos.numel() > 0:
                guidance_pos = guidance_pos[0, 1].item()
                # Keep some context before guidance token
                start_pos = max(0, guidance_pos - 50)
                input_ids = input_ids[:, start_pos:]
                token_types = token_types[:, start_pos:]
                current_length = input_ids.size(1)
        
        # Prepare attention mask
        attention_mask = torch.ones((1, current_length), device=self.device)
        
        return {
            'input_ids': input_ids,
            'token_type_ids': token_types,
            'attention_mask': attention_mask,
            'visual_token_ids': visual_token_ids,
            'visual_tokens': visual_tokens if visual_features is not None else None,
            'tool_token_ids': tool_token_ids,
            'tool_tokens': tool_tokens if tool_features is not None else None,
            'max_guidance_length': max_guidance_length
        }
    
    def embed_tokens(self, inputs):
        """
        Embed input tokens, replacing visual and tool token embeddings with the projected features.
        
        Args:
            inputs: Dict with input components
            
        Returns:
            Token embeddings
        """
        # Get standard token embeddings
        token_embeddings = self.model.transformer.wte(inputs['input_ids'])
        
        # Add token type embeddings
        token_embeddings = token_embeddings + self.token_type_embeddings(inputs['token_type_ids'])
        
        # Replace visual token embeddings if present
        if inputs['visual_token_ids'] is not None and inputs['visual_tokens'] is not None:
            # Find positions of visual tokens
            visual_positions = (inputs['input_ids'] == inputs['visual_token_ids'][0, 0]).nonzero()
            if visual_positions.numel() > 0:
                start_pos = visual_positions[0, 1].item()
                end_pos = start_pos + inputs['visual_tokens'].size(1)
                if end_pos <= token_embeddings.size(1):
                    token_embeddings[:, start_pos:end_pos] = inputs['visual_tokens']
        
        # Replace tool token embeddings if present
        if inputs['tool_token_ids'] is not None and inputs['tool_tokens'] is not None:
            # Find positions of tool tokens
            tool_positions = (inputs['input_ids'] == inputs['tool_token_ids'][0, 0]).nonzero()
            if tool_positions.numel() > 0:
                start_pos = tool_positions[0, 1].item()
                end_pos = start_pos + inputs['tool_tokens'].size(1)
                if end_pos <= token_embeddings.size(1):
                    token_embeddings[:, start_pos:end_pos] = inputs['tool_tokens']
        
        return token_embeddings
    
    def forward(self, inputs):
        """
        Forward pass for the model.
        
        Args:
            inputs: Dict with input components
            
        Returns:
            Dict with output logits
        """
        # Embed tokens
        inputs_embeds = self.embed_tokens(inputs)
        
        # Forward pass through the model
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs['attention_mask'],
            return_dict=True
        )
        
        return outputs
    
    def generate_guidance(self, text_prompt, visual_features=None, tool_features=None, 
                          phase_name=None, max_length=100, temperature=0.7, num_return_sequences=1):
        """
        Generate guidance for the surgical procedure.
        
        Args:
            text_prompt: Text prompt describing the situation
            visual_features: Visual features from the current frame
            tool_features: Tool detection features
            phase_name: Current surgical phase name
            max_length: Maximum guidance text length
            temperature: Sampling temperature
            num_return_sequences: Number of guidance options to generate
            
        Returns:
            List of generated guidance texts
        """
        # Prepare model inputs
        inputs = self.prepare_inputs(
            text_prompt=text_prompt,
            visual_features=visual_features,
            tool_features=tool_features,
            phase_name=phase_name,
            max_guidance_length=max_length
        )
        
        # Create inputs for generation
        gen_inputs = {
            'inputs_embeds': self.embed_tokens(inputs),
            'attention_mask': inputs['attention_mask'],
            'max_length': inputs['input_ids'].size(1) + max_length,
            'temperature': temperature,
            'num_return_sequences': num_return_sequences,
            'do_sample': True,
            'top_k': 50,
            'top_p': 0.95,
            'pad_token_id': self.tokenizer.eos_token_id
        }
        
        # Generate guidance
        with torch.no_grad():
            outputs = self.model.generate(**gen_inputs)
        
        # Decode outputs
        guidance_texts = []
        for output in outputs:
            # Get only the newly generated tokens
            new_tokens = output[inputs['input_ids'].size(1):]
            # Decode the tokens
            guidance_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            # Stop at the end guidance tag if present
            if '</guidance>' in guidance_text:
                guidance_text = guidance_text.split('</guidance>')[0]
            guidance_texts.append(guidance_text.strip())
        
        return guidance_texts
    
    def get_step_guidance(self, current_phase, detected_tools, current_actions, patient_status=None):
        """
        Get procedural guidance based on current surgical phase and detected tools.
        
        Args:
            current_phase: Current surgical phase
            detected_tools: List of detected tools
            current_actions: Description of current actions
            patient_status: Optional patient status information
            
        Returns:
            Dict with guidance information
        """
        # Create prompt
        prompt = f"Current phase: {current_phase}. "
        
        if detected_tools:
            prompt += f"Detected tools: {', '.join(detected_tools)}. "
        
        prompt += f"Actions: {current_actions}. "
        
        if patient_status:
            prompt += f"Patient status: {patient_status}. "
        
        if current_phase in self.procedure_steps:
            # Add procedure knowledge to the prompt
            step_info = self.procedure_steps[current_phase]
            prompt += f"Standard procedure for this phase: {step_info['description']}. "
            
            if 'key_points' in step_info:
                prompt += f"Key points: {'; '.join(step_info['key_points'])}. "
        
        prompt += "Provide guidance for the surgeon: "
        
        # Generate guidance
        guidance_texts = self.generate_guidance(
            text_prompt=prompt,
            phase_name=current_phase,
            max_length=150
        )
        
        # Create response
        response = {
            'phase': current_phase,
            'guidance': guidance_texts[0],
            'detected_tools': detected_tools,
            'procedure_reference': self.procedure_steps.get(current_phase, {}),
            'alternatives': guidance_texts[1:] if len(guidance_texts) > 1 else []
        }
        
        return response


def get_mistake_detector_model(config):
    """
    Factory function to create a mistake detection model from config.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized mistake detection model
    """
    return SurgicalMistakeDetector(
        visual_dim=config.get('visual_dim', 768),
        tool_dim=config.get('tool_dim', 128),
        hidden_dim=config.get('hidden_dim', 256),
        num_classes=config.get('num_classes', 3),
        use_temporal=config.get('use_temporal', True)
    )
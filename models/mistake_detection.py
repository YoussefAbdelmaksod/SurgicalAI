"""
Surgical mistake detection models for SurgicalAI.

This module implements models for detecting mistakes and providing guidance
during surgical procedures, using multi-modal inputs and contextual awareness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
import json
import os

logger = logging.getLogger(__name__)

# Try importing transformers for GPT models
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers package not installed. GPT-based models will not be available.")
    TRANSFORMERS_AVAILABLE = False


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module for combining visual features with tool detection results.
    """
    def __init__(self, visual_dim, tool_dim, output_dim, dropout=0.2):
        """
        Initialize fusion module.
        
        Args:
            visual_dim: Dimension of visual features
            tool_dim: Dimension of tool detection features
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.visual_projection = nn.Linear(visual_dim, output_dim)
        self.tool_projection = nn.Linear(tool_dim, output_dim)
        
        # Attention for modality fusion
        self.attention = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, visual_features, tool_features):
        """
        Forward pass for multi-modal fusion.
        
        Args:
            visual_features: Visual features [batch_size, visual_dim]
            tool_features: Tool detection features [batch_size, tool_dim]
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        # Project features to common space
        visual_proj = self.visual_projection(visual_features)
        tool_proj = self.tool_projection(tool_features)
        
        # Calculate attention weights
        concat_features = torch.cat([visual_proj, tool_proj], dim=1)
        attention_weights = self.attention(concat_features)
        
        # Apply attention weights
        fused = attention_weights[:, 0:1] * visual_proj + attention_weights[:, 1:2] * tool_proj
        
        # Apply fusion layer
        output = self.fusion(fused)
        
        return output


class TemporalContextProcessor(nn.Module):
    """
    Temporal context processor using GRU for sequential information.
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.2):
        """
        Initialize temporal context processor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.output_dim = hidden_dim * 2  # bidirectional
    
    def forward(self, x, lengths=None):
        """
        Process sequence with temporal context.
        
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            lengths: Sequence lengths for packed sequence
            
        Returns:
            Processed sequence with temporal context [batch_size, seq_len, output_dim]
        """
        if lengths is not None:
            # Pack sequence
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # Process with GRU
            packed_output, _ = self.gru(packed_x)
            
            # Unpack sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            # Process with GRU directly
            output, _ = self.gru(x)
        
        return output


class SurgicalMistakeDetector(nn.Module):
    """
    Multi-modal model for detecting surgical mistakes with risk assessment.
    """
    def __init__(self, visual_dim=768, tool_dim=128, num_tools=10, hidden_dim=256, 
                 num_classes=3, use_temporal=True, dropout=0.3):
        """
        Initialize surgical mistake detector.
        
        Args:
            visual_dim: Dimension of visual features
            tool_dim: Dimension of tool features
            num_tools: Number of possible surgical tools
            hidden_dim: Hidden dimension
            num_classes: Number of mistake/risk classes
            use_temporal: Whether to use temporal context
            dropout: Dropout rate
        """
        super().__init__()
        
        self.use_temporal = use_temporal
        
        # Tool embedding layer (one-hot tool vectors to dense embeddings)
        self.tool_embedding = nn.Embedding(num_tools, tool_dim)
        
        # Multi-modal fusion
        self.fusion = MultiModalFusion(
            visual_dim=visual_dim,
            tool_dim=tool_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # Temporal context processor (if enabled)
        if use_temporal:
            self.temporal_processor = TemporalContextProcessor(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=dropout
            )
            temporal_dim = self.temporal_processor.output_dim
        else:
            temporal_dim = hidden_dim
        
        # Mistake detection head
        self.mistake_classifier = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Risk assessment head
        self.risk_regressor = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Risk score between 0 and 1
        )
    
    def forward(self, visual_features, tool_ids, lengths=None):
        """
        Forward pass for mistake detection.
        
        Args:
            visual_features: Visual features [batch_size, seq_len, visual_dim]
            tool_ids: Tool IDs [batch_size, seq_len]
            lengths: Sequence lengths (optional)
            
        Returns:
            Dict containing mistake logits and risk scores
        """
        batch_size, seq_len = visual_features.size(0), visual_features.size(1)
        
        # Process each timestep
        fused_features = []
        for t in range(seq_len):
            # Get features for current timestep
            visual_t = visual_features[:, t]
            tool_ids_t = tool_ids[:, t]
            
            # Get tool embeddings
            tool_embeddings = self.tool_embedding(tool_ids_t)
            
            # Fuse features
            fused_t = self.fusion(visual_t, tool_embeddings)
            fused_features.append(fused_t)
        
        # Stack features
        fused_features = torch.stack(fused_features, dim=1)  # [batch_size, seq_len, hidden_dim]
        
        # Apply temporal processing if enabled
        if self.use_temporal:
            features = self.temporal_processor(fused_features, lengths)
        else:
            features = fused_features
        
        # Detect mistakes and assess risk
        mistake_logits = self.mistake_classifier(features)
        risk_scores = self.risk_regressor(features)
        
        return {
            'mistake_logits': mistake_logits,
            'risk_scores': risk_scores
        }
    
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
    
    def predict(self, visual_features, tool_detections, threshold=0.5):
        """
        Make predictions with the model.
        
        Args:
            visual_features: Visual features [batch_size, seq_len, channels, height, width]
            tool_detections: Tool detection results
            threshold: Confidence threshold for mistake detection
            
        Returns:
            Dict with prediction results
        """
        # Ensure model is in evaluation mode
        self.eval()
        
        with torch.no_grad():
            batch_size, seq_len = visual_features.size(0), visual_features.size(1)
            
            # Extract visual features (assuming they're already processed)
            visual_features_flat = visual_features.reshape(batch_size, seq_len, -1)
            
            # Convert tool detections to tool IDs
            tool_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=visual_features.device)
            
            # Fill in detected tools (assuming highest confidence tool)
            for b in range(batch_size):
                for t in range(seq_len):
                    if 'labels' in tool_detections and len(tool_detections['labels'][b]) > 0:
                        # Get highest confidence tool
                        max_idx = torch.argmax(tool_detections['scores'][b])
                        tool_ids[b, t] = tool_detections['labels'][b][max_idx]
            
            # Forward pass
            outputs = self.forward(visual_features_flat, tool_ids)
            
            # Get predictions
            mistake_logits = outputs['mistake_logits']
            risk_scores = outputs['risk_scores']
            
            # Get mistake indices and probabilities
            mistake_probs = F.softmax(mistake_logits, dim=-1)
            mistake_indices = torch.argmax(mistake_probs, dim=-1)
            
            # Convert to numpy arrays
            mistake_indices_np = mistake_indices.cpu().numpy()
            mistake_probs_np = mistake_probs.cpu().numpy()
            risk_scores_np = risk_scores.cpu().numpy()
            
            # Create mapping from indices to mistake names
            mistake_names = ['no_mistake', 'minor_mistake', 'critical_mistake']
            
            # Create mapping from risk score to description
            def risk_to_description(risk):
                if risk < 0.3:
                    return 'Low Risk'
                elif risk < 0.7:
                    return 'Medium Risk'
                else:
                    return 'High Risk'
            
            # Create results dictionary
            results = {
                'mistake_indices': mistake_indices_np,
                'mistake_probabilities': mistake_probs_np,
                'mistake_names': [mistake_names[idx] for idx in mistake_indices_np.flatten()],
                'risk_levels': risk_scores_np.flatten(),
                'risk_descriptions': [risk_to_description(risk) for risk in risk_scores_np.flatten()]
            }
            
            return results
    
    def explain_prediction(self, mistake_idx, risk_level):
        """
        Generate explanation for a prediction.
        
        Args:
            mistake_idx: Mistake index
            risk_level: Risk level score
            
        Returns:
            Explanation string
        """
        # Mapping from mistake index to explanation template
        explanation_templates = {
            0: "No mistakes detected in the current action.",
            1: "Minor mistake detected: {0}. This is generally recoverable.",
            2: "Critical mistake detected: {0}. Immediate correction is recommended."
        }
        
        # Specific details based on risk level
        risk_details = {
            0: [  # No mistake
                "Procedure is following standard protocol.",
                "All actions appear correct based on the current phase.",
                "Tool usage is appropriate for this stage."
            ],
            1: [  # Minor mistake
                "Instrument positioning could be improved.",
                "Excessive force may be applied.",
                "Slight deviation from optimal tissue handling.",
                "Inefficient tool movement detected."
            ],
            2: [  # Critical mistake
                "Possible anatomical structure at risk.",
                "Inappropriate tool selection for this step.",
                "Potential risk of tissue damage.",
                "Critical structure is being approached unsafely."
            ]
        }
        
        # Select specific detail based on risk level
        import random
        detail_idx = int(risk_level * len(risk_details[mistake_idx]))
        detail_idx = min(detail_idx, len(risk_details[mistake_idx]) - 1)
        detail = risk_details[mistake_idx][detail_idx]
        
        # Generate explanation
        explanation = explanation_templates[mistake_idx].format(detail)
        
        return explanation


class GPTSurgicalAssistant(nn.Module):
    """
    GPT-based model for context-aware guidance and surgical procedure assistance.
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
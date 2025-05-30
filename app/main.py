"""
Main application module for SurgicalAI system.

This module provides integration of all SurgicalAI components, including:
- Surgical phase recognition with ViT-LSTM
- Surgical tool detection with Faster R-CNN
- Surgical mistake detection and risk assessment
- Guidance generation with GPT-based models

Optimized for real-time laparoscopic cholecystectomy procedure analysis.
"""

import os
import sys
import logging
import cv2
import torch
import numpy as np
import yaml
from flask import Flask, render_template, request, jsonify, Response
import threading
import time
import queue
import torch.cuda.amp as amp

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from models.phase_recognition import ViTLSTM, ViTTransformerTemporal
from models.tool_detection import AdvancedToolDetectionModel, ToolDetectionEnsemble
from models.mistake_detection import SurgicalMistakeDetector, GPTSurgicalAssistant
from utils.helpers import load_config, setup_logging, resize_image, normalize_image, get_device

# Implement load_video function since we removed dataloader.py
def load_video(video_path, frame_rate=1):
    """
    Load video frames from a file path.
    
    Args:
        video_path: Path to video file
        frame_rate: Frame rate for sampling (frames per second)
        
    Returns:
        List of frames (BGR format)
    """
    if not os.path.exists(video_path):
        return None
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval for desired frame rate
    frame_interval = max(1, int(fps / frame_rate))
    
    # Extract frames
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Only keep frames at the specified interval
        if frame_idx % frame_interval == 0:
            frames.append(frame)
        
        frame_idx += 1
    
    # Release video capture
    cap.release()
    
    return frames

# Constants specific to laparoscopic cholecystectomy
CHOLECYSTECTOMY_PHASES = [
    'preparation',
    'calot_triangle_dissection',
    'clipping_and_cutting',
    'gallbladder_dissection',
    'gallbladder_packaging',
    'cleaning_and_coagulation',
    'gallbladder_extraction'
]

# Critical structures in laparoscopic cholecystectomy
CRITICAL_STRUCTURES = {
    'cystic_duct': {'risk_multiplier': 2.0, 'associated_phase': 'calot_triangle_dissection'},
    'cystic_artery': {'risk_multiplier': 2.0, 'associated_phase': 'calot_triangle_dissection'},
    'common_bile_duct': {'risk_multiplier': 3.0, 'associated_phase': 'calot_triangle_dissection'},
    'hepatic_artery': {'risk_multiplier': 2.5, 'associated_phase': 'calot_triangle_dissection'},
    'liver_bed': {'risk_multiplier': 1.5, 'associated_phase': 'gallbladder_dissection'}
}

# Critical phase to tool mapping
PHASE_TOOL_MAPPING = {
    'calot_triangle_dissection': ['Grasper', 'Hook', 'Bipolar'],
    'clipping_and_cutting': ['Clipper', 'Scissors'],
    'gallbladder_dissection': ['Grasper', 'Hook'],
    'gallbladder_packaging': ['Grasper', 'Specimen Bag']
}


class SurgicalAISystem:
    """
    Integrated SurgicalAI system for real-time surgical analysis.
    Combines multiple advanced models for comprehensive surgical assistance.
    Optimized for laparoscopic cholecystectomy procedures.
    """
    
    def __init__(self, config_path='config/default_config.yaml', use_ensemble=True, use_gpt=True):
        """
        Initialize the SurgicalAI system.
        
        Args:
            config_path: Path to the configuration file
            use_ensemble: Whether to use ensemble models for tool detection
            use_gpt: Whether to use GPT for guidance generation
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Set up logging
        self.logger = setup_logging(self.config['app']['log_level'])
        self.logger.info("Initializing SurgicalAI system")
        
        # Set device
        self.device = get_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Flag to track model initialization
        self.models_initialized = False
        
        # Initialize models
        self._init_models(use_ensemble, use_gpt)
        
        # Frame buffer for temporal context
        self.frame_buffer = []
        self.max_buffer_size = 15  # Increased buffer size for better temporal context
        
        # Inference lock to prevent concurrent access
        self.inference_lock = threading.Lock()
        
        # Initialize mixed precision scaler for faster inference
        self.scaler = amp.GradScaler()
        
        # Initialize phase tracking
        self.current_phase = None
        self.phase_confidence = 0.0
        self.phase_start_time = None
        self.phase_durations = {phase: 0 for phase in CHOLECYSTECTOMY_PHASES}
        
        # Initialize critical structure detection
        self.critical_structures_detected = {}
        
        # Processing rate for different components (for real-time optimization)
        self.process_rates = {
            'phase_recognition': 5,    # Process every 5 frames
            'tool_detection': 1,       # Process every frame (critical for safety)
            'mistake_detection': 3,    # Process every 3 frames
        }
        
        # Frame counters
        self.frame_counters = {
            'phase_recognition': 0,
            'tool_detection': 0,
            'mistake_detection': 0,
        }
        
        self.logger.info("SurgicalAI system initialized successfully")
    
    def _init_models(self, use_ensemble, use_gpt):
        """Initialize all AI models with optimizations for cholecystectomy."""
        self.logger.info("Loading AI models...")
        
        # Set up model paths
        models_base_path = os.path.join('models', 'weights')
        phase_model_path = os.path.join(models_base_path, 'vit_lstm', 'phase_recognition.pth')
        tool_model_path = os.path.join(models_base_path, 'tool_detection', 'tool_detection.pth')
        mistake_model_path = os.path.join(models_base_path, 'mistake_detector', 'mistake_detection.pth')
        guidance_model_path = os.path.join(models_base_path, 'guidance', 'guidance.pth')
        
        # Check if model directories exist
        os.makedirs(os.path.dirname(phase_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(tool_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(mistake_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(guidance_model_path), exist_ok=True)
        
        # Load phase recognition model
        self.logger.info("Loading phase recognition model...")
        self.phase_model = None
        try:
            # Try to load pretrained weights
            if os.path.exists(phase_model_path):
                self.logger.info(f"Loading pretrained weights from {phase_model_path}")
        self.phase_model = ViTLSTM(
                    num_classes=len(CHOLECYSTECTOMY_PHASES),
                    pretrained=True
        ).to(self.device)
                self.phase_model.load_state_dict(torch.load(phase_model_path, map_location=self.device))
            else:
                # Fall back to base model if no weights found
                self.logger.warning(f"No pretrained weights found at {phase_model_path}, using base model")
                self.phase_model = ViTLSTM(
                    num_classes=len(CHOLECYSTECTOMY_PHASES),
                    pretrained=True
                ).to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to load phase recognition model: {str(e)}")
        
        # Load tool detection model
        self.logger.info("Loading tool detection model...")
        self.tool_model = None
        try:
            # Use ensemble if requested
        if use_ensemble:
                self.logger.info("Using ensemble model for tool detection")
            self.tool_model = ToolDetectionEnsemble(
                    config=self.config['model']['tool_detection'],
                    device=self.device
                )
        else:
            self.tool_model = AdvancedToolDetectionModel(
                    config=self.config['model']['tool_detection'],
                    device=self.device
                )
                
            # Load pretrained weights if available
            if os.path.exists(tool_model_path):
                self.logger.info(f"Loading pretrained weights from {tool_model_path}")
                self.tool_model.load_state_dict(torch.load(tool_model_path, map_location=self.device))
        except Exception as e:
            self.logger.error(f"Failed to load tool detection model: {str(e)}")
        
        # Load mistake detection model
        self.logger.info("Loading mistake detection model...")
        self.mistake_model = None
        try:
        self.mistake_model = SurgicalMistakeDetector(
                config=self.config['model']['mistake_detection'],
                device=self.device
            )
            
            # Load pretrained weights if available
            if os.path.exists(mistake_model_path):
                self.logger.info(f"Loading pretrained weights from {mistake_model_path}")
                self.mistake_model.load_state_dict(torch.load(mistake_model_path, map_location=self.device))
        except Exception as e:
            self.logger.error(f"Failed to load mistake detection model: {str(e)}")
            
        # Load GPT guidance model if requested
        self.guidance_model = None
        if use_gpt:
            self.logger.info("Loading GPT guidance model...")
            try:
                from models.gpt_guidance import SurgicalGPTGuidance
                
                procedure_knowledge_path = self.config['paths'].get('procedure_knowledge', 'data/procedure_knowledge.json')
                
                self.guidance_model = SurgicalGPTGuidance(
                    model_name="gpt2",
                    weights_path=guidance_model_path if os.path.exists(guidance_model_path) else None,
                    procedure_knowledge_path=procedure_knowledge_path,
                    device=self.device
                )
            except Exception as e:
                self.logger.error(f"Failed to load GPT guidance model: {str(e)}")
                
        # Initialize voice assistant
        self.logger.info("Initializing voice assistant...")
        try:
            from models.voice_assistant import VoiceAssistant
            
            voice_config = self.config.get('voice_guidance', {})
            feedback_level = voice_config.get('feedback_level', 'standard')
            critical_warnings_only = voice_config.get('critical_warnings_only', False)
            
            self.voice_assistant = VoiceAssistant(
                feedback_level=feedback_level,
                critical_warnings_only=critical_warnings_only,
                enable_voice_commands=voice_config.get('enable_voice_commands', False)
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize voice assistant: {str(e)}")
            self.voice_assistant = None
            
        # Initialize user profile manager
        self.logger.info("Initializing user profile manager...")
        try:
            from utils.user_profiles import ProfileManager
            
            profiles_dir = os.path.join('data', 'profiles')
            self.profile_manager = ProfileManager(profiles_dir=profiles_dir)
            
            # Set default user profile
            default_user_id = self.config.get('app', {}).get('default_user_id', 'default')
            
            # Get or create default profile
            self.current_user_profile = self.profile_manager.get_profile(default_user_id)
            if not self.current_user_profile:
                self.current_user_profile = self.profile_manager.create_profile(
                    user_id=default_user_id,
                    name="Default User",
                    experience_level="intermediate"
                )
                
            # Update voice assistant with user profile
            if self.voice_assistant:
                self.voice_assistant.user_profile = self.current_user_profile
        except Exception as e:
            self.logger.error(f"Failed to initialize user profile manager: {str(e)}")
            self.profile_manager = None
            self.current_user_profile = None
            
        # Set models to evaluation mode
        if self.phase_model:
            self.phase_model.eval()
        if self.tool_model:
            self.tool_model.eval()
        if self.mistake_model:
            self.mistake_model.eval()
        if self.guidance_model:
            self.guidance_model.eval()
            
        # Track if models are initialized
        self.models_initialized = (self.phase_model is not None and 
                                 self.tool_model is not None and 
                                 self.mistake_model is not None)
                                 
        if self.models_initialized:
            self.logger.info("All models initialized successfully")
        else:
            self.logger.warning("Some models failed to initialize")
            
    def set_user_profile(self, user_id=None, user_profile=None):
        """
        Set the current user profile for personalized guidance.
        
        Args:
            user_id: User ID to load profile for (ignored if user_profile is provided)
            user_profile: User profile object
            
        Returns:
            bool: True if profile was set successfully
        """
        if not self.profile_manager:
            self.logger.warning("Profile manager not initialized, cannot set user profile")
            return False
            
        if user_profile:
            self.current_user_profile = user_profile
            
            # Update voice assistant with new profile
            if self.voice_assistant:
                self.voice_assistant.user_profile = self.current_user_profile
                
            self.logger.info(f"User profile set to provided profile: {user_profile.user_id}")
            return True
            
        if user_id:
            # Get profile from manager
            profile = self.profile_manager.get_profile(user_id)
            if profile:
                self.current_user_profile = profile
                
                # Update voice assistant with new profile
                if self.voice_assistant:
                    self.voice_assistant.user_profile = self.current_user_profile
                    
                self.logger.info(f"User profile set to: {user_id}")
                return True
            else:
                self.logger.warning(f"User profile not found for ID: {user_id}")
                return False
                
        return False
        
    def provide_guidance(self, context):
        """
        Provide personalized guidance based on the current surgical context.
        
        Args:
            context: Dictionary with surgical context
            
        Returns:
            Dict with guidance information
        """
        if not self.guidance_model:
            return {"primary_guidance": "Guidance model not available."}
            
        # Add current user profile to context
        return self.guidance_model.get_personalized_guidance(
            context=context,
            user_profile=self.current_user_profile
        )
        
    def speak_guidance(self, message, priority_level="information", context=None):
        """
        Speak a guidance message with the voice assistant.
        
        Args:
            message: Message to speak
            priority_level: Priority level of the message
            context: Additional context information
            
        Returns:
            bool: True if message was spoken
        """
        if not self.voice_assistant:
            self.logger.info(f"Voice guidance (not spoken): {message}")
            return False
            
        if self.current_user_profile:
            # Use personalized guidance
            self.voice_assistant.provide_personalized_guidance(
                message=message,
                priority_level=priority_level,
                context=context
            )
        else:
            # Use standard guidance
            self.voice_assistant.speak(
                message=message,
                priority_level=priority_level,
                context=context
            )
            
        return True
        
    def handle_phase_transition(self, new_phase, confidence):
        """
        Handle transition to a new surgical phase.
        
        Args:
            new_phase: Name of the new phase
            confidence: Confidence level of the phase prediction
            
        Returns:
            bool: True if transition was handled successfully
        """
        if new_phase == self.current_phase:
            return False
            
        old_phase = self.current_phase
        self.current_phase = new_phase
        
        # Get phase description from config
        phase_info = self.config.get('phases', {}).get(new_phase, {})
        phase_description = phase_info.get('description', '')
        
        # Log phase transition
        self.logger.info(f"Phase transition: {old_phase} -> {new_phase} (confidence: {confidence:.2f})")
        
        # Create context for guidance
        context = {
            "phase": new_phase,
            "previous_phase": old_phase,
            "confidence": confidence,
            "description": phase_description
        }
        
        # Provide voice guidance for the new phase
        if self.voice_assistant:
            self.voice_assistant.provide_phase_guidance(new_phase, is_transition=True)
            
        # Generate guidance using GPT model if available
        if self.guidance_model:
            guidance = self.guidance_model.get_phase_guidance(
                phase_name=new_phase,
                is_transition=True,
                user_profile=self.current_user_profile
            )
            
            # Speak guidance if there's additional information beyond the phase announcement
            if guidance and len(guidance) > 20:  # Arbitrary threshold to avoid speaking just the phase name
                self.speak_guidance(guidance, priority_level="instruction", context=context)
                
        return True
        
    def handle_mistake_detection(self, mistake_info, frame_context):
        """
        Handle detected surgical mistake.
        
        Args:
            mistake_info: Dictionary with mistake information
            frame_context: Current frame context
            
        Returns:
            bool: True if mistake was handled successfully
        """
        if not mistake_info:
            return False
            
        # Log mistake
        mistake_type = mistake_info.get("type", "unknown")
        description = mistake_info.get("description", "")
        risk_level = mistake_info.get("risk_level", 0.0)
        
        self.logger.info(f"Mistake detected: {mistake_type} - {description} (risk: {risk_level:.2f})")
        
        # Provide voice warning
        if self.voice_assistant:
            self.voice_assistant.warn_about_mistake(mistake_info)
            
        # Generate guidance using GPT model if available
        if self.guidance_model and risk_level >= 0.5:  # Only for significant mistakes
            context = {
                "phase": self.current_phase,
                "detected_tools": frame_context.get("detected_tools", []),
                "anatomical_structures": frame_context.get("anatomical_structures", []),
                "cvs_achieved": frame_context.get("cvs_achieved", False),
                "previous_actions": []
            }
            
            guidance = self.guidance_model.get_mistake_guidance(
                mistake_info=mistake_info,
                context=context,
                user_profile=self.current_user_profile
            )
            
            # Speak guidance if not empty
            if guidance:
                self.speak_guidance(guidance, priority_level="warning", context=context)
                
        return True
        
    def update_user_performance(self, session_data):
        """
        Update user performance metrics in their profile.
        
        Args:
            session_data: Dictionary with session performance data
            
        Returns:
            bool: True if update was successful
        """
        if not self.current_user_profile or not self.profile_manager:
            return False
            
        try:
            # Add performance record to user profile
            self.current_user_profile.add_performance_record(
                procedure_type="laparoscopic_cholecystectomy",
                performance_metrics=session_data.get("metrics", {}),
                mistakes=session_data.get("mistakes", []),
                phase_durations=session_data.get("phase_durations", {}),
                tool_usage=session_data.get("tool_usage", {})
            )
            
            # Save updated profile
            self.profile_manager.update_profile(self.current_user_profile)
            
            self.logger.info(f"Updated performance metrics for user: {self.current_user_profile.user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update user performance: {str(e)}")
            return False
    
    def preprocess_frame(self, frame):
        """
        Preprocess a frame for model input.
        
        Args:
            frame: Raw input frame (BGR format)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        frame_resized = resize_image(
            frame_rgb, 
            target_size=tuple(self.config['data']['image_size'])
        )
        
        # Normalize pixel values
        frame_normalized = normalize_image(frame_resized)
        
        # Convert to tensor
        frame_tensor = torch.FloatTensor(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        
        # Move to device
        frame_tensor = frame_tensor.to(self.device)
        
        return frame_tensor
    
    def update_frame_buffer(self, frame_tensor):
        """
        Update frame buffer with new frame tensor.
        
        Args:
            frame_tensor: New frame tensor to add to buffer
        """
        # Add new frame to buffer
        self.frame_buffer.append(frame_tensor)
        
        # Remove oldest frame if buffer is full
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
    
    def get_sequence_tensor(self):
        """
        Get sequence tensor from frame buffer.
        
        Returns:
            Sequence tensor [1, seq_len, channels, height, width]
        """
        if not self.frame_buffer:
            return None
        
        # Stack frames along sequence dimension
        sequence = torch.cat(self.frame_buffer, dim=0).unsqueeze(0)
        
        return sequence
    
    def analyze_frame(self, frame):
        """
        Analyze a single frame to detect phase, tools, and potential mistakes.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dict with analysis results
        """
        if not self.models_initialized:
            self.logger.error("Models not initialized")
            return {"error": "Models not initialized"}
        
        # Acquire lock to prevent concurrent inference
            with self.inference_lock:
                try:
                # Preprocess frame
                    frame_tensor = self.preprocess_frame(frame)
                
                # Update frame buffer
                self.update_frame_buffer(frame_tensor)
                
                # Initialize results
                results = {
                    "timestamp": time.time(),
                    "frame_size": frame.shape
                }
                
                # Get sequence tensor
                sequence = self.get_sequence_tensor()
                
                if sequence is None:
                    return {"error": "Not enough frames in buffer"}
                
                # Increment frame counters
                for key in self.frame_counters:
                    self.frame_counters[key] += 1
                
                # Use mixed precision for faster inference
                with torch.cuda.amp.autocast(enabled=True):
                    # Phase recognition (process every N frames)
                    if self.frame_counters['phase_recognition'] >= self.process_rates['phase_recognition']:
                        self.frame_counters['phase_recognition'] = 0
                        
                        # Set model to evaluation mode
                        self.phase_model.eval()
                        
                        # Perform inference
                    with torch.no_grad():
                            phase_results = self.phase_model.predict(sequence, smooth=True)
                        
                        # Get phase index and name
                        phase_index = phase_results['phase_indices'][0][-1]  # Get most recent phase
                        phase_name = phase_results['phase_names'][0][-1]
                        phase_probs = phase_results['probabilities'][0][-1]
                        
                        # Get phase confidence
                        phase_confidence = float(phase_probs[phase_index])
                        
                        # Phase change tracking
                        if self.current_phase != phase_name and phase_confidence > 0.7:
                            if self.current_phase is not None:
                                # Record duration of previous phase
                                if self.phase_start_time is not None:
                                    duration = time.time() - self.phase_start_time
                                    self.phase_durations[self.current_phase] = duration
                            
                            # Update current phase
                            self.current_phase = phase_name
                            self.phase_confidence = phase_confidence
                            self.phase_start_time = time.time()
                            
                            # Log phase change
                            self.logger.info(f"Phase changed to {phase_name} with confidence {phase_confidence:.2f}")
                        
                        # Add results
                                results['phase'] = {
                            'name': phase_name,
                            'confidence': phase_confidence,
                            'index': int(phase_index),
                            'durations': self.phase_durations
                        }
                        
                        # Add phase-specific guidance based on current phase
                        results['phase']['guidance'] = self._get_phase_guidance(phase_name)
                    
                    # Tool detection (process every frame for critical phases, otherwise every N frames)
                    is_critical_phase = self.current_phase in ['calot_triangle_dissection', 'clipping_and_cutting']
                    tool_process_rate = 1 if is_critical_phase else self.process_rates['tool_detection']
                    
                    if self.frame_counters['tool_detection'] >= tool_process_rate:
                        self.frame_counters['tool_detection'] = 0
                        
                        # Set model to evaluation mode
                        self.tool_model.eval()
                        
                        # Perform inference
                        with torch.no_grad():
                            # Use only the last frame for tool detection
                            tool_detections = self.tool_model(frame_tensor)
                            
                            # Get tool detection results
                            if len(tool_detections) > 0:
                                detections = tool_detections[0]  # First image in batch
                                
                                # Filter detections by confidence
                                conf_mask = detections['scores'] > 0.6
                                boxes = detections['boxes'][conf_mask].cpu().numpy()
                                scores = detections['scores'][conf_mask].cpu().numpy()
                                labels = detections['labels'][conf_mask].cpu().numpy()
                                
                                # Convert labels to tool names
                                tool_names = [self._get_tool_name(label) for label in labels]
                                
                                # Add results
                                results['tools'] = {
                                    'boxes': boxes.tolist(),
                                    'scores': scores.tolist(),
                                    'names': tool_names
                                }
                                
                                # Add tool-specific guidance based on current phase
                                if self.current_phase in PHASE_TOOL_MAPPING:
                                    recommended_tools = PHASE_TOOL_MAPPING[self.current_phase]
                                    detected_tools = set(tool_names)
                                    missing_tools = [t for t in recommended_tools if t not in detected_tools]
                                    
                                    results['tools']['recommended'] = recommended_tools
                                    results['tools']['missing'] = missing_tools
                                    
                                    if missing_tools:
                                        results['tools']['guidance'] = f"Consider using {', '.join(missing_tools)} for this phase."
                    
                    # Mistake detection (process every N frames)
                    if self.frame_counters['mistake_detection'] >= self.process_rates['mistake_detection']:
                        self.frame_counters['mistake_detection'] = 0
                        
                        try:
                            # Set model to evaluation mode
                            self.mistake_model.eval()
                            
                            # Prepare inputs for mistake detection
                            visual_features = sequence
                            tool_detections = results.get('tools', {'names': []})
                            
                            # Perform inference
                            with torch.no_grad():
                                mistake_results = self.mistake_model.predict(visual_features, tool_detections)
                                
                                # Only report mistakes if confidence is high enough
                                if mistake_results['mistake_indices'][0] > 0:  # Not 'no_mistake'
                                    mistake_confidence = mistake_results['mistake_probabilities'][0][mistake_results['mistake_indices'][0]]
                                    
                                # Apply risk multiplier for critical phases
                                risk_multiplier = 1.0
                                if self.current_phase in ['calot_triangle_dissection', 'clipping_and_cutting']:
                                    risk_multiplier = 1.5
                                
                                adjusted_confidence = min(mistake_confidence * risk_multiplier, 1.0)
                                
                                if adjusted_confidence > 0.5:  # Confidence threshold
                                        results['mistake'] = {
                                            'name': mistake_results['mistake_names'][0],
                                        'confidence': float(adjusted_confidence),
                                            'risk_description': mistake_results['risk_descriptions'][0]
                                        }
                                        
                                        # Generate explanation
                                        explanation = self.mistake_model.explain_prediction(
                                            mistake_results['mistake_indices'][0],
                                            mistake_results['risk_levels'][0]
                                        )
                                        
                                        results['mistake']['explanation'] = explanation
                                    
                                    # Add specific guidance for cholecystectomy mistakes
                                    if self.current_phase == 'calot_triangle_dissection':
                                        results['mistake']['guidance'] = "Ensure critical view of safety before proceeding."
                                    elif self.current_phase == 'clipping_and_cutting':
                                        results['mistake']['guidance'] = "Confirm proper clip placement before cutting."
                            except Exception as e:
                                self.logger.error(f"Mistake detection failed: {str(e)}")
                        
                return results
            
                            except Exception as e:
                self.logger.error(f"Frame analysis failed: {str(e)}")
                return {"error": str(e)}
    
    def _get_tool_name(self, label_id):
        """Get tool name from label ID."""
        tool_names = [
            'Background',
            'Bipolar',
            'Clipper',
            'Grasper',
            'Hook',
            'Irrigator',
            'Scissors',
            'Specimen Bag'
        ]
        
        if 0 <= label_id < len(tool_names):
            return tool_names[label_id]
        else:
            return f"Unknown ({label_id})"
    
    def _get_phase_guidance(self, phase_name):
        """Get phase-specific guidance for cholecystectomy."""
        guidance = {
            'preparation': "Ensure proper port placement and initial exploration.",
            'calot_triangle_dissection': "Dissect carefully to achieve critical view of safety. Identify cystic duct and artery.",
            'clipping_and_cutting': "Place three clips on cystic duct (two proximal, one distal) and cut between proximal clips.",
            'gallbladder_dissection': "Dissect gallbladder from liver bed using electrocautery. Stay close to gallbladder wall.",
            'gallbladder_packaging': "Place gallbladder in specimen bag for extraction.",
            'cleaning_and_coagulation': "Check liver bed for bleeding and ensure hemostasis.",
            'gallbladder_extraction': "Extract specimen bag with gallbladder through umbilical port."
        }
        
        return guidance.get(phase_name, "")
    
    def process_video(self, video_path, output_path=None, frame_rate=1):
        """
        Process a surgical video file.
        
        Args:
            video_path: Path to input video file
            output_path: Path to output video file (optional)
            frame_rate: Frame rate for processing (frames per second)
            
        Returns:
            Dict with processing results
        """
        if not self.models_initialized:
            self.logger.error("Models not initialized")
            return {"error": "Models not initialized"}
        
        self.logger.info(f"Processing video: {video_path}")
        
        # Load video
        video_frames = load_video(video_path, frame_rate=frame_rate)
        
        if not video_frames:
            self.logger.error("Failed to load video")
            return {"error": "Failed to load video"}
        
        self.logger.info(f"Loaded {len(video_frames)} frames from video")
        
        # Process frames
        results = []
        
        for i, frame in enumerate(video_frames):
            self.logger.info(f"Processing frame {i+1}/{len(video_frames)}")
            
            # Analyze frame
            frame_results = self.analyze_frame(frame)
            results.append(frame_results)
            
            # Visualize results
            if output_path:
                visualization = self.visualize_results(frame.copy(), frame_results)
                
                # Write frame to output video
                # (Implementation would go here)
        
        return {
            "video_path": video_path,
            "frames_processed": len(video_frames),
            "results": results
        }
    
    def visualize_results(self, frame, results):
        """
        Visualize analysis results on frame.
        
        Args:
            frame: Input frame
            results: Analysis results
            
        Returns:
            Frame with visualizations
        """
        # Skip if error in results
        if "error" in results:
            cv2.putText(frame, f"Error: {results['error']}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # Visualize phase
        if "phase" in results:
            phase_name = results["phase"]["name"]
            confidence = results["phase"]["confidence"]
            
            # Different colors for different phases
            phase_colors = {
                'preparation': (255, 255, 0),
                'calot_triangle_dissection': (0, 0, 255),  # Red for critical phase
                'clipping_and_cutting': (0, 0, 255),       # Red for critical phase
                'gallbladder_dissection': (255, 0, 0),
                'gallbladder_packaging': (0, 255, 0),
                'cleaning_and_coagulation': (0, 255, 255),
                'gallbladder_extraction': (255, 0, 255)
            }
            
            color = phase_colors.get(phase_name, (255, 255, 255))
            
            # Draw phase name and confidence
            cv2.putText(frame, f"Phase: {phase_name}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Highlight border for critical phases
            if phase_name in ['calot_triangle_dissection', 'clipping_and_cutting']:
                border_thickness = 10
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), border_thickness)
        
        # Visualize tool detections
        if "tools" in results:
            boxes = results["tools"].get("boxes", [])
            scores = results["tools"].get("scores", [])
            names = results["tools"].get("names", [])
            
            for box, score, name in zip(boxes, scores, names):
                x1, y1, x2, y2 = map(int, box)
                
                # Determine color based on tool type
                if name == 'Clipper' or name == 'Scissors':
                    # Highlight critical tools in red
                    color = (0, 0, 255)  # Red
                elif name == 'Grasper' or name == 'Hook':
                    color = (255, 0, 0)  # Blue
                else:
                    color = (0, 255, 0)  # Green
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw tool name and confidence
                label = f"{name}: {score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Visualize mistake detection
        if "mistake" in results:
            mistake = results["mistake"]
            name = mistake["name"]
            risk = mistake["risk_description"]
            explanation = mistake.get("explanation", "")
            
            # Determine color based on risk level
            if risk == 'High Risk':
                color = (0, 0, 255)  # Red
            elif risk == 'Medium Risk':
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 255)  # Yellow
            
            # Draw mistake information
            cv2.putText(frame, f"Mistake: {name}", (10, frame.shape[0] - 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Risk: {risk}", (10, frame.shape[0] - 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Wrap explanation text to fit screen
            max_width = 80
            words = explanation.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line + " " + word) <= max_width:
                    current_line += " " + word if current_line else word
                else:
                    lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Draw explanation
            for i, line in enumerate(lines):
                y_pos = frame.shape[0] - 30 + i * 25
                if y_pos < frame.shape[0]:
                    cv2.putText(frame, line, (10, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame


class VideoProcessor:
    """
    Real-time video processor for SurgicalAI.
    Handles video capture and frame processing.
    """
    
    def __init__(self, surgical_ai_system, source=0, process_every_n_frames=5):
        """
        Initialize the video processor.
        
        Args:
            surgical_ai_system: SurgicalAI system instance
            source: Video source (0 for webcam, or path to video file)
            process_every_n_frames: Process every n frames to reduce computational load
        """
        self.surgical_ai = surgical_ai_system
        self.source = source
        self.process_every_n_frames = process_every_n_frames
        
        # Frame processing queue and thread
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        self.processing_thread = None
        self.running = False
        
        # Latest frame and result
        self.latest_frame = None
        self.latest_result = None
        self.latest_visualization = None
        
        # Frame counter
        self.frame_count = 0
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def start(self):
        """Start video processing."""
        if self.running:
            return
        
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Open video capture
        self.cap = cv2.VideoCapture(self.source)
        
        self.surgical_ai.logger.info("Video processor started")
    
    def stop(self):
        """Stop video processing."""
        self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
        
        # Release video capture
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        self.surgical_ai.logger.info("Video processor stopped")
    
    def _process_frames(self):
        """Frame processing thread."""
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.1)
                
                # Process frame
                result = self.surgical_ai.analyze_frame(frame)
                
                # Create visualization
                visualization = self.surgical_ai.visualize_results(frame, result)
                
                # Update latest results
                with self.lock:
                    self.latest_result = result
                    self.latest_visualization = visualization
                
                # Put result in queue
                try:
                    self.result_queue.put((result, visualization), block=False)
                except queue.Full:
                    pass
                
                # Mark task as done
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.surgical_ai.logger.error(f"Error processing frame: {e}")
    
    def get_frame(self):
        """
        Get the latest frame and analysis results.
        
        Returns:
            tuple: (frame, results, visualization)
        """
        if not self.running:
            return None, None, None
        
        # Read frame from video capture
        ret, frame = self.cap.read()
        
        if not ret:
            # End of video or error
            self.stop()
            return None, None, None
        
        # Increment frame counter
        self.frame_count += 1
        
        # Only process every n frames
        if self.frame_count % self.process_every_n_frames == 0:
            try:
                # Add frame to processing queue
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass
        
        # Get latest results
        with self.lock:
            latest_result = self.latest_result
            latest_visualization = self.latest_visualization
        
        return frame, latest_result, latest_visualization
    
    def get_jpeg_frame(self):
        """
        Get JPEG-encoded visualization frame for web streaming.
        
        Returns:
            JPEG-encoded frame bytes
        """
        _, _, visualization = self.get_frame()
        
        if visualization is None:
            # Return an empty frame if no visualization is available
            empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', empty_frame)
        else:
            _, buffer = cv2.imencode('.jpg', visualization)
        
        return buffer.tobytes()


class WebApp:
    """
    Web application for SurgicalAI interface.
    """
    
    def __init__(self, surgical_ai_system=None):
        """
        Initialize the web application.
        
        Args:
            surgical_ai_system: SurgicalAI system instance
        """
        # Load configuration
        self.config = load_config('config/default_config.yaml')
        
        # Create Flask app
        self.app = Flask(__name__)
        
        # Set up logging
        self.logger = setup_logging(self.config['app']['log_level'])
        
        # Create SurgicalAI system if not provided
        if surgical_ai_system is None:
            self.surgical_ai = SurgicalAISystem(
                config_path='config/default_config.yaml',
                use_ensemble=True,
                use_gpt=True
            )
        else:
            self.surgical_ai = surgical_ai_system
        
        # Video processor
        self.video_processor = None
        
        # Set up routes
        self._setup_routes()
        
        self.logger.info("Web application initialized")
    
    def _setup_routes(self):
        """Set up web app routes."""
        # Main page
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        # API status
        @self.app.route('/api/status')
        def status():
            return jsonify({
                'status': 'online',
                'version': '0.1.0'
            })
        
        # Start video processing
        @self.app.route('/api/start_video', methods=['POST'])
        def start_video():
            try:
                data = request.get_json()
                source = data.get('source', 0)
                
                if isinstance(source, str) and source.isdigit():
                    source = int(source)
                
                if self.video_processor:
                    self.video_processor.stop()
                
                self.video_processor = VideoProcessor(
                    self.surgical_ai,
                    source=source,
                    process_every_n_frames=5
                )
                self.video_processor.start()
                
                return jsonify({'status': 'success'})
                
            except Exception as e:
                self.logger.error(f"Error starting video: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        # Stop video processing
        @self.app.route('/api/stop_video', methods=['POST'])
        def stop_video():
            try:
                if self.video_processor:
                    self.video_processor.stop()
                    self.video_processor = None
                
                return jsonify({'status': 'success'})
                
            except Exception as e:
                self.logger.error(f"Error stopping video: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        # Video stream
        @self.app.route('/video_feed')
        def video_feed():
            def generate():
                while self.video_processor and self.video_processor.running:
                    frame = self.video_processor.get_jpeg_frame()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(0.033)  # ~30 FPS
            
            return Response(generate(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        
        # Analyze a single image
        @self.app.route('/api/analyze', methods=['POST'])
        def analyze():
            try:
                if 'image' not in request.files:
                    return jsonify({'error': 'No image provided'}), 400
                
                file = request.files['image']
                
                # Read and convert image
                img_bytes = file.read()
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Analyze frame
                results = self.surgical_ai.analyze_frame(frame)
                
                return jsonify(results)
                
            except Exception as e:
                self.logger.error(f"Error analyzing image: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def run(self, host=None, port=None, debug=None):
        """
        Run the web application.
        
        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        # Use config values if not provided
        host = host or self.config['app']['host']
        port = port or self.config['app']['port']
        debug = debug if debug is not None else self.config['app']['debug']
        
        self.logger.info(f"Starting web application on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)


def main():
    """Main entry point for the SurgicalAI application."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='SurgicalAI Application')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['web', 'video', 'image'], default='web',
                       help='Application mode (web, video, image)')
    parser.add_argument('--input', type=str, default=None,
                       help='Input file for video or image mode')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for video or image mode')
    parser.add_argument('--host', type=str, default=None,
                       help='Host address for web mode')
    parser.add_argument('--port', type=int, default=None,
                       help='Port number for web mode')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode for web mode')
    parser.add_argument('--no-ensemble', action='store_true',
                       help='Disable model ensemble')
    parser.add_argument('--no-gpt', action='store_true',
                       help='Disable GPT guidance')
    
    args = parser.parse_args()
    
    # Create SurgicalAI system
    system = SurgicalAISystem(
        config_path=args.config,
        use_ensemble=not args.no_ensemble,
        use_gpt=not args.no_gpt
    )
    
    # Run in specified mode
    if args.mode == 'web':
        # Create and run web application
        app = WebApp(system)
        app.run(host=args.host, port=args.port, debug=args.debug)
        
    elif args.mode == 'video':
        # Process video file
        if args.input is None:
            print("Error: Input video file is required for video mode")
            return
        
        system.process_video(args.input, args.output, frame_rate=1)
        
    elif args.mode == 'image':
        # Process single image
        if args.input is None:
            print("Error: Input image file is required for image mode")
            return
        
        # Read image
        frame = cv2.imread(args.input)
        
        # Analyze frame
        results = system.analyze_frame(frame)
        
        # Print results
        import json
        print(json.dumps(results, indent=2))
        
        # Save visualization if output path is provided
        if args.output:
            visualization = system.visualize_results(frame, results)
            cv2.imwrite(args.output, visualization)


if __name__ == '__main__':
    main() 
#!/usr/bin/env python
"""
SurgicalAI Inference Script

This script runs inference on surgical videos using the SurgicalAI system.
It provides a command-line interface for processing videos and visualizing results.
"""

import os
import sys
import argparse
import logging
import time
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from utils.helpers import setup_logging, load_config, get_device
from models import check_model_weights
from models.phase_recognition import ViTLSTM
from models.tool_detection import AdvancedToolDetectionModel
from models.mistake_detection import SurgicalMistakeDetector

# Set up logging
logger = logging.getLogger(__name__)
setup_logging()

class SurgicalAIInference:
    """
    SurgicalAI inference system for processing surgical videos.
    """
    
    def __init__(self, config_path="config/default_config.yaml", device=None):
        """
        Initialize the SurgicalAI inference system.
        
        Args:
            config_path: Path to configuration file
            device: Device to run inference on ('cuda', 'cpu')
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Set device
        self.device = device or get_device()
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._init_models()
        
        # Frame buffer for temporal context
        self.frame_buffer = []
        self.max_buffer_size = self.config.get('inference', {}).get('buffer_size', 15)
        
        # Initialize phase tracking
        self.phase_history = []
        self.phase_names = self.config['model']['phase_recognition'].get('class_names', [
            'preparation', 'calot_triangle_dissection', 'clipping_cutting', 
            'gallbladder_dissection', 'gallbladder_extraction', 
            'cleaning_coagulation', 'closing'
        ])
        
        # Processing rates for different components
        self.process_rates = {
            'phase_recognition': self.config.get('inference', {}).get('phase_recognition_interval', 5),
            'tool_detection': self.config.get('inference', {}).get('tool_detection_interval', 1),
            'mistake_detection': self.config.get('inference', {}).get('mistake_detection_interval', 3),
        }
        
        # Frame counters
        self.frame_counters = {k: 0 for k in self.process_rates.keys()}
        
        logger.info("SurgicalAI inference system initialized")
    
    def _init_models(self):
        """Initialize inference models."""
        logger.info("Loading models...")
        
        # Check model weights
        weight_status = check_model_weights()
        
        # Load phase recognition model if weights exist
        if weight_status['phase_recognition']:
            logger.info("Loading phase recognition model...")
            self.phase_model = ViTLSTM(
                num_classes=len(self.phase_names),
                vit_model=self.config['model']['phase_recognition'].get('vit_model', 'vit_base_patch16_224'),
                hidden_size=self.config['model']['phase_recognition'].get('hidden_size', 512),
                num_layers=self.config['model']['phase_recognition'].get('num_layers', 2),
                dropout=self.config['model']['phase_recognition'].get('dropout', 0.3),
                use_temporal_attention=self.config['model']['phase_recognition'].get('use_temporal_attention', True)
            ).to(self.device)
            
            # Load weights
            weights_path = os.path.join("models", "weights", "phase", "phase_recognition.pth")
            self.phase_model.load(weights_path)
            logger.info(f"Loaded phase recognition model from {weights_path}")
        else:
            logger.warning("Phase recognition weights not found. Using random weights.")
            self.phase_model = ViTLSTM(
                num_classes=len(self.phase_names),
                pretrained=True  # At least use pretrained ViT backbone
            ).to(self.device)
        
        # Load tool detection model if weights exist
        if weight_status['tool_detection']:
            logger.info("Loading tool detection model...")
            self.tool_model = AdvancedToolDetectionModel(
                num_classes=self.config['model']['tool_detection'].get('num_classes', 8),
                architecture=self.config['model']['tool_detection'].get('architecture', 'faster_rcnn'),
                backbone_name=self.config['model']['tool_detection'].get('backbone', 'resnet50'),
                use_fpn=self.config['model']['tool_detection'].get('use_fpn', True),
                score_threshold=self.config['model']['tool_detection'].get('score_threshold', 0.5)
            ).to(self.device)
            
            # Load weights
            weights_path = os.path.join("models", "weights", "tool", "tool_detection.pth")
            self.tool_model.load(weights_path)
            logger.info(f"Loaded tool detection model from {weights_path}")
        else:
            logger.warning("Tool detection weights not found. Using pretrained COCO weights.")
            self.tool_model = AdvancedToolDetectionModel(
                num_classes=8,
                pretrained=True  # Use COCO pretrained weights
            ).to(self.device)
        
        # Load mistake detection model if weights exist
        if weight_status['mistake_detection']:
            logger.info("Loading mistake detection model...")
            self.mistake_model = SurgicalMistakeDetector(
                visual_dim=self.config['model']['mistake_detection'].get('visual_dim', 768),
                tool_dim=self.config['model']['mistake_detection'].get('tool_dim', 128),
                num_tools=self.config['model']['mistake_detection'].get('num_tools', 10),
                hidden_dim=self.config['model']['mistake_detection'].get('hidden_dim', 256),
                num_classes=self.config['model']['mistake_detection'].get('num_classes', 3),
                use_temporal=self.config['model']['mistake_detection'].get('use_temporal', True)
            ).to(self.device)
            
            # Load weights
            weights_path = os.path.join("models", "weights", "mistake", "mistake_detection.pth")
            self.mistake_model.load(weights_path)
            logger.info(f"Loaded mistake detection model from {weights_path}")
            self.has_mistake_model = True
        else:
            logger.warning("Mistake detection weights not found. Mistake detection will be disabled.")
            self.has_mistake_model = False
    
    def preprocess_frame(self, frame):
        """
        Preprocess a frame for inference.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Tensor: Preprocessed frame tensor
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 (standard input size for ViT)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        
        # Convert to float and normalize
        frame_float = frame_resized.astype(np.float32) / 255.0
        frame_normalized = (frame_float - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to tensor and add batch dimension
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return frame_tensor
    
    def update_frame_buffer(self, frame_tensor):
        """
        Update frame buffer with a new frame.
        
        Args:
            frame_tensor: Input frame tensor
        """
        self.frame_buffer.append(frame_tensor.clone())
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
    
    def get_sequence_tensor(self):
        """
        Get sequence tensor from frame buffer.
        
        Returns:
            Tensor: Sequence tensor
        """
        if not self.frame_buffer:
            return None
        
        return torch.cat(self.frame_buffer, dim=0)
    
    def process_frame(self, frame, frame_idx):
        """
        Process a single frame.
        
        Args:
            frame: Input frame (numpy array)
            frame_idx: Frame index
            
        Returns:
            dict: Processing results
        """
        results = {
            'frame_idx': frame_idx,
            'phase': None,
            'phase_confidence': 0.0,
            'tools': [],
            'mistakes': []
        }
        
        # Preprocess frame
        frame_tensor = self.preprocess_frame(frame)
        frame_tensor = frame_tensor.to(self.device)
        
        # Update frame buffer
        self.update_frame_buffer(frame_tensor)
        
        # Process phase recognition
        if frame_idx % self.process_rates['phase_recognition'] == 0:
            self.frame_counters['phase_recognition'] += 1
            
            if len(self.frame_buffer) >= 5:  # Need minimum sequence length
                sequence = self.get_sequence_tensor()
                
                # Run phase recognition
                with torch.no_grad():
                    phase_logits, attention = self.phase_model(sequence.unsqueeze(0))
                    phase_probs = torch.softmax(phase_logits, dim=1)[0]
                    
                    # Get predicted phase
                    phase_idx = torch.argmax(phase_probs).item()
                    phase_conf = phase_probs[phase_idx].item()
                    
                    # Update results
                    results['phase'] = self.phase_names[phase_idx]
                    results['phase_confidence'] = phase_conf
                    
                    # Update phase history for temporal consistency
                    self.phase_history.append((phase_idx, phase_conf))
                    if len(self.phase_history) > 10:
                        self.phase_history.pop(0)
        
        # Process tool detection
        if frame_idx % self.process_rates['tool_detection'] == 0:
            self.frame_counters['tool_detection'] += 1
            
            # Prepare input for tool detection (needs original frame)
            height, width = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor_orig = torch.from_numpy(frame_rgb.transpose(2, 0, 1)).float() / 255.0
            frame_tensor_orig = frame_tensor_orig.to(self.device).unsqueeze(0)
            
            # Run tool detection
            with torch.no_grad():
                detections = self.tool_model(frame_tensor_orig)
                
                # Process detections
                if isinstance(detections, list) and len(detections) > 0:
                    # Get detections for first image in batch
                    detection = detections[0]
                    
                    # Extract boxes, scores, and labels
                    if 'boxes' in detection and len(detection['boxes']) > 0:
                        boxes = detection['boxes'].cpu().numpy()
                        scores = detection['scores'].cpu().numpy()
                        labels = detection['labels'].cpu().numpy()
                        
                        # Filter by confidence
                        threshold = self.config['model']['tool_detection'].get('score_threshold', 0.5)
                        valid_idx = scores >= threshold
                        
                        boxes = boxes[valid_idx]
                        scores = scores[valid_idx]
                        labels = labels[valid_idx]
                        
                        # Convert to original image coordinates
                        boxes[:, 0] *= width / frame_tensor_orig.shape[3]
                        boxes[:, 2] *= width / frame_tensor_orig.shape[3]
                        boxes[:, 1] *= height / frame_tensor_orig.shape[2]
                        boxes[:, 3] *= height / frame_tensor_orig.shape[2]
                        
                        # Add to results
                        for box, score, label in zip(boxes, scores, labels):
                            tool_name = self._get_tool_name(label)
                            results['tools'].append({
                                'tool': tool_name,
                                'bbox': box.tolist(),
                                'score': float(score)
                            })
        
        # Process mistake detection if available
        if self.has_mistake_model and frame_idx % self.process_rates['mistake_detection'] == 0:
            self.frame_counters['mistake_detection'] += 1
            
            # Only run mistake detection if we have phase and tool information
            if results['phase'] is not None and len(results['tools']) > 0:
                # Get current phase index
                current_phase_idx = self.phase_names.index(results['phase'])
                
                # Extract visual features from the last frame in buffer
                visual_features = self.frame_buffer[-1].to(self.device)
                
                # Prepare tool IDs
                tool_ids = [self._get_tool_id(tool['tool']) for tool in results['tools']]
                tool_ids_tensor = torch.tensor(tool_ids, device=self.device)
                
                # Run mistake detection
                with torch.no_grad():
                    mistake_output = self.mistake_model.predict(
                        visual_features, 
                        tool_ids_tensor, 
                        current_phase=current_phase_idx
                    )
                    
                    # Add mistakes to results
                    for mistake in mistake_output:
                        results['mistakes'].append({
                            'type': mistake['type'],
                            'risk_level': mistake['risk_level'],
                            'description': mistake['description'],
                            'confidence': mistake['confidence']
                        })
        
        return results
    
    def _get_tool_name(self, label_id):
        """Get tool name from label ID."""
        tool_categories = self.config['model']['tool_detection'].get('categories', [
            'Background', 'Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'Specimen bag'
        ])
        
        if 0 <= label_id < len(tool_categories):
            return tool_categories[label_id]
        return f"Unknown-{label_id}"
    
    def _get_tool_id(self, tool_name):
        """Get tool ID from tool name."""
        tool_categories = self.config['model']['tool_detection'].get('categories', [
            'Background', 'Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'Specimen bag'
        ])
        
        try:
            return tool_categories.index(tool_name)
        except ValueError:
            return 0  # Return background class if not found
    
    def visualize_results(self, frame, results):
        """
        Visualize processing results on a frame.
        
        Args:
            frame: Input frame
            results: Processing results
            
        Returns:
            Frame with visualizations
        """
        # Create a copy of the frame
        vis_frame = frame.copy()
        
        # Draw phase information
        if results['phase'] is not None:
            phase_text = f"Phase: {results['phase']} ({results['phase_confidence']:.2f})"
            cv2.putText(vis_frame, phase_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw tool detections
        for tool in results['tools']:
            # Get bounding box
            x1, y1, x2, y2 = map(int, tool['bbox'])
            
            # Draw rectangle
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            label_text = f"{tool['tool']} ({tool['score']:.2f})"
            cv2.putText(vis_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw mistake detections
        if results['mistakes']:
            mistake_y = 60
            for mistake in results['mistakes']:
                # Color based on risk level
                if mistake['risk_level'] == 'high':
                    color = (0, 0, 255)  # Red for high risk
                elif mistake['risk_level'] == 'medium':
                    color = (0, 165, 255)  # Orange for medium risk
                else:
                    color = (0, 255, 255)  # Yellow for low risk
                
                # Draw mistake information
                mistake_text = f"Risk: {mistake['type']} ({mistake['risk_level']})"
                cv2.putText(vis_frame, mistake_text, (10, mistake_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                mistake_y += 25
        
        return vis_frame
    
    def process_video(self, video_path, output_path=None, show_output=True, save_output=True):
        """
        Process a surgical video file and analyze each frame.
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the output video (None to auto-generate)
            show_output: Whether to show the output video in a window
            save_output: Whether to save the output video
            
        Returns:
            Dict with analysis results and metrics
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Setup output video writer if needed
        if save_output:
            if output_path is None:
                # Auto-generate output path
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(os.path.dirname(video_path), f"{base_name}_analysis.mp4")
                
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Saving output to: {output_path}")
        
        # Setup voice guidance if available
        voice_guidance = None
        try:
            from models.voice_assistant import VoiceAssistant
            from utils.user_profiles import ProfileManager
            
            # Initialize voice assistant
            voice_guidance = VoiceAssistant(
                feedback_level="standard",
                critical_warnings_only=False,
                enable_voice_commands=False
            )
            
            # Try to load a user profile
            try:
                profile_manager = ProfileManager(profiles_dir="data/profiles")
                user_profile = profile_manager.get_profile("default")
                if user_profile:
                    voice_guidance.user_profile = user_profile
                    logger.info("Loaded user profile for personalized guidance")
            except Exception as e:
                logger.warning(f"Could not load user profile: {e}")
                
            logger.info("Voice guidance enabled")
        except Exception as e:
            logger.warning(f"Voice guidance not available: {e}")
        
        # Initialize stats
        frame_count = 0
        processed_count = 0
        skipped_count = 0
        start_time = time.time()
        
        # Initialize results storage
        phase_changes = []
        detected_tools = []
        detected_mistakes = []
        
        # Current phase tracking
        current_phase = None
        phase_start_time = None
        phase_durations = {}
        
        # Progress bar
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        # Process frames
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames based on process rates for efficiency
            should_process = {}
            for component, rate in self.process_rates.items():
                self.frame_counters[component] += 1
                if self.frame_counters[component] >= rate:
                    should_process[component] = True
                    self.frame_counters[component] = 0
                else:
                    should_process[component] = False
            
            # Process frame if any component needs processing
            if any(should_process.values()):
                # Process frame with SurgicalAI
                results = self.process_frame(frame, frame_count)
                processed_count += 1
                
                # Voice guidance for phase transitions
                if voice_guidance and results.get('phase') != current_phase:
                    # Track phase change
                    new_phase = results.get('phase')
                    if new_phase:
                        # Record phase change
                        phase_changes.append({
                            'frame': frame_count,
                            'time': frame_count / fps,
                            'previous_phase': current_phase,
                            'new_phase': new_phase,
                            'confidence': results.get('phase_confidence', 0.0)
                        })
                        
                        # Calculate phase duration if not first phase
                        if current_phase and phase_start_time:
                            phase_duration = frame_count / fps - phase_start_time
                            phase_durations[current_phase] = phase_duration
                            
                        # Update current phase and start time
                        current_phase = new_phase
                        phase_start_time = frame_count / fps
                        
                        # Provide voice guidance for phase transition
                        voice_guidance.provide_phase_guidance(
                            phase_name=new_phase, 
                            is_transition=True
                        )
                
                # Voice guidance for detected mistakes
                if voice_guidance and results.get('mistakes'):
                    for mistake in results.get('mistakes', []):
                        # Record mistake
                        detected_mistakes.append({
                            'frame': frame_count,
                            'time': frame_count / fps,
                            'phase': current_phase,
                            'type': mistake.get('type', 'unknown'),
                            'description': mistake.get('description', ''),
                            'risk_level': mistake.get('risk_level', 0.0)
                        })
                        
                        # Provide voice warning about mistake
                        voice_guidance.warn_about_mistake(mistake)
                
                # Record detected tools
                if results.get('tools'):
                    detected_tools.append({
                        'frame': frame_count,
                        'time': frame_count / fps,
                        'phase': current_phase,
                        'tools': results.get('tools', [])
                    })
                    
                    # Get recommended tools for current phase
                    recommended_tools = []
                    if current_phase and current_phase in self.config.get('phase_tool_mapping', {}):
                        recommended_tools = self.config['phase_tool_mapping'][current_phase]
                        
                    # Check if right tools are being used
                    if voice_guidance and recommended_tools:
                        voice_guidance.provide_tool_guidance(
                            current_tools=results.get('tools', []),
                            recommended_tools=recommended_tools
                        )
                
                # Draw visualization
                vis_frame = self.visualize_results(frame.copy(), results)
                
                # Show output if requested
                if show_output:
                    cv2.imshow('SurgicalAI Analysis', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Save output if requested
                if save_output:
                    out.write(vis_frame)
            else:
                skipped_count += 1
                
                # For skipped frames, just copy visualization if showing/saving
                if show_output or save_output:
                    # Use last visualization or original frame
                    vis_frame = frame.copy()
                    
                    # Show output if requested
                    if show_output:
                        cv2.imshow('SurgicalAI Analysis', vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    # Save output if requested
                    if save_output:
                        out.write(vis_frame)
            
            # Update progress bar
            pbar.update(1)
        
        # Close resources
        pbar.close()
        cap.release()
        if save_output:
            out.release()
        if show_output:
            cv2.destroyAllWindows()
            
        # Calculate final phase duration for last phase
        if current_phase and phase_start_time:
            phase_duration = frame_count / fps - phase_start_time
            phase_durations[current_phase] = phase_duration
            
        # Calculate processing stats
        end_time = time.time()
        processing_time = end_time - start_time
        fps_processed = frame_count / processing_time
        
        logger.info(f"Processed {processed_count}/{frame_count} frames ({skipped_count} skipped)")
        logger.info(f"Processing time: {processing_time:.2f}s ({fps_processed:.2f} fps)")
        
        # Clean up voice guidance
        if voice_guidance:
            voice_guidance.cleanup()
        
        # Compile results
        results = {
            'video_path': video_path,
            'output_path': output_path if save_output else None,
            'frames_processed': processed_count,
            'frames_total': frame_count,
            'processing_time': processing_time,
            'fps_processed': fps_processed,
            'phase_changes': phase_changes,
            'phase_durations': phase_durations,
            'detected_tools': detected_tools,
            'detected_mistakes': detected_mistakes
        }
        
        return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='SurgicalAI Inference')
    
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output video')
    parser.add_argument('--results', type=str, default=None,
                        help='Path to save results as JSON')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--process-rate', type=int, default=1,
                        help='Process every Nth frame')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on (cuda, cpu)')
    
    args = parser.parse_args()
    
    # Create inference system
    inference = SurgicalAIInference(args.config, args.device)
    
    # Process video
    results = inference.process_video(
        args.video,
        args.output,
        not args.no_visualize,
        args.process_rate
    )
    
    # Save results
    if args.results:
        import json
        with open(args.results, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {args.results}")
    
    logger.info("Inference complete!")

if __name__ == '__main__':
    main() 
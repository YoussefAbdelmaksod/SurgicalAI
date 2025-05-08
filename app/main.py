"""
Main application module for SurgicalAI system.

This module provides integration of all SurgicalAI components, including:
- Surgical phase recognition with ViT-LSTM
- Surgical tool detection with Faster R-CNN
- Surgical mistake detection and risk assessment
- Guidance generation with GPT-based models
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from models.phase_recognition import ViTLSTM, ViTTransformerTemporal
from models.tool_detection import AdvancedToolDetectionModel, ToolDetectionEnsemble
from models.mistake_detection import SurgicalMistakeDetector, GPTSurgicalAssistant
from utils.helpers import load_config, setup_logging, resize_image, normalize_image, get_device
from data.dataloader import load_video


class SurgicalAISystem:
    """
    Integrated SurgicalAI system for real-time surgical analysis.
    Combines multiple advanced models for comprehensive surgical assistance.
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
        
        # Flag to indicate if we're using untrained models
        self.demo_mode = False
        
        # Initialize models
        self._init_models(use_ensemble, use_gpt)
        
        # Frame buffer for temporal context
        self.frame_buffer = []
        self.max_buffer_size = 10  # Store the last 10 frames
        
        # Inference lock to prevent concurrent access
        self.inference_lock = threading.Lock()
        
        self.logger.info("SurgicalAI system initialized successfully")
        if self.demo_mode:
            self.logger.warning("RUNNING IN DEMO MODE: Using untrained models. Predictions may not be accurate.")
    
    def _init_models(self, use_ensemble, use_gpt):
        """Initialize all AI models."""
        self.logger.info("Loading AI models...")
        
        # Load phase recognition model
        self.logger.info("Loading phase recognition model...")
        self.phase_model = ViTLSTM(
            num_classes=7,
            hidden_size=512,
            num_layers=3,
            dropout=0.3,
            pretrained=True,
            use_temporal_attention=True
        ).to(self.device)
        
        # Load model weights if available
        phase_model_path = os.path.join('models', 'weights', 'vit_lstm', 'phase_recognition.pth')
        if os.path.exists(phase_model_path):
            try:
                self.phase_model.load(phase_model_path)
                self.logger.info(f"Loaded phase recognition model from {phase_model_path}")
                self.demo_mode = False
            except Exception as e:
                self.logger.error(f"Failed to load phase recognition model: {str(e)}")
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(phase_model_path), exist_ok=True)
                # Save the current model state as a placeholder
                torch.save(self.phase_model.state_dict(), phase_model_path)
                self.logger.warning(f"Created new phase recognition model weights at {phase_model_path}")
                self.demo_mode = False
        else:
            os.makedirs(os.path.dirname(phase_model_path), exist_ok=True)
            # Save the current model state as a placeholder
            torch.save(self.phase_model.state_dict(), phase_model_path)
            self.logger.warning(f"Created new phase recognition model weights at {phase_model_path}")
            self.demo_mode = False
        
        # Load tool detection model
        self.logger.info("Loading tool detection model...")
        
        if use_ensemble:
            # Create an ensemble of different tool detection models
            model1 = AdvancedToolDetectionModel(
                num_classes=8,
                architecture='faster_rcnn',
                backbone_name='resnet50',
                pretrained=True,
                use_fpn=True
            ).to(self.device)
            
            model2 = AdvancedToolDetectionModel(
                num_classes=8,
                architecture='mask_rcnn',
                backbone_name='resnet101',
                pretrained=True,
                use_fpn=True
            ).to(self.device)
            
            # Create ensemble
            self.tool_model = ToolDetectionEnsemble(
                models=[model1, model2],
                ensemble_method='weighted',
                weights=[0.6, 0.4]  # Weights for each model
            ).to(self.device)
            
        else:
            # Use single tool detection model
            self.tool_model = AdvancedToolDetectionModel(
                num_classes=8,
                architecture='faster_rcnn',
                backbone_name='resnet50',
                pretrained=True,
                use_fpn=True
            ).to(self.device)
        
        # Load model weights if available
        tool_model_path = os.path.join('models', 'weights', 'tool_detection', 'tool_detection.pth')
        if os.path.exists(tool_model_path):
            try:
                # For ensemble, this would load weights for the first model
                if use_ensemble:
                    self.tool_model.models[0].load(tool_model_path)
                else:
                    self.tool_model.load(tool_model_path)
                self.logger.info(f"Loaded tool detection model from {tool_model_path}")
                self.demo_mode = False
            except Exception as e:
                self.logger.error(f"Failed to load tool detection model: {str(e)}")
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(tool_model_path), exist_ok=True)
                # Save current model state as a placeholder
                if use_ensemble:
                    torch.save(self.tool_model.models[0].state_dict(), tool_model_path)
                else:
                    torch.save(self.tool_model.state_dict(), tool_model_path)
                self.logger.warning(f"Created new tool detection model weights at {tool_model_path}")
                self.demo_mode = False
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(tool_model_path), exist_ok=True)
            # Save current model state as a placeholder
            if use_ensemble:
                torch.save(self.tool_model.models[0].state_dict(), tool_model_path)
            else:
                torch.save(self.tool_model.state_dict(), tool_model_path)
            self.logger.warning(f"Created new tool detection model weights at {tool_model_path}")
            self.demo_mode = False
        
        # Load mistake detection model
        self.logger.info("Loading mistake detection model...")
        self.mistake_model = SurgicalMistakeDetector(
            visual_dim=768,
            tool_dim=128,
            num_tools=10,
            hidden_dim=256,
            num_classes=3,
            use_temporal=True,
            dropout=0.3
        ).to(self.device)
        
        # Load model weights if available
        mistake_model_path = os.path.join('models', 'weights', 'mistake_detector', 'mistake_detection.pth')
        if os.path.exists(mistake_model_path):
            try:
                self.mistake_model.load(mistake_model_path)
                self.logger.info(f"Loaded mistake detection model from {mistake_model_path}")
                self.demo_mode = False
            except Exception as e:
                self.logger.error(f"Failed to load mistake detection model: {str(e)}")
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(mistake_model_path), exist_ok=True)
                # Save current model state as a placeholder
                torch.save(self.mistake_model.state_dict(), mistake_model_path)
                self.logger.warning(f"Created new mistake detection model weights at {mistake_model_path}")
                self.demo_mode = False
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(mistake_model_path), exist_ok=True)
            # Save current model state as a placeholder
            torch.save(self.mistake_model.state_dict(), mistake_model_path)
            self.logger.warning(f"Created new mistake detection model weights at {mistake_model_path}")
            self.demo_mode = False
        
        # Load GPT-based guidance model if requested
        if use_gpt:
            self.logger.info("Loading GPT-based guidance model...")
            self.guidance_model = GPTSurgicalAssistant(
                model_name='gpt2',
                num_visual_tokens=50,
                num_tool_tokens=20,
                max_sequence_length=512,
                device=self.device
            ).to(self.device)
            
            # Load model weights if available
            guidance_model_path = os.path.join('models', 'weights', 'guidance.pth')
            if os.path.exists(guidance_model_path):
                try:
                    self.guidance_model.load(guidance_model_path)
                    self.logger.info(f"Loaded GPT guidance model from {guidance_model_path}")
                    self.demo_mode = False
                except Exception as e:
                    self.logger.error(f"Failed to load guidance model: {str(e)}")
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(guidance_model_path), exist_ok=True)
                    # Save current model state as a placeholder
                    torch.save(self.guidance_model.state_dict(), guidance_model_path)
                    self.logger.warning(f"Created new guidance model weights at {guidance_model_path}")
                    self.demo_mode = False
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(guidance_model_path), exist_ok=True)
                # Save current model state as a placeholder
                torch.save(self.guidance_model.state_dict(), guidance_model_path)
                self.logger.warning(f"Created new guidance model weights at {guidance_model_path}")
                self.demo_mode = False
            
            self.use_gpt = True
        else:
            self.use_gpt = False
        
        # Set all models to evaluation mode
        self.phase_model.eval()
        self.tool_model.eval()
        self.mistake_model.eval()
        if self.use_gpt:
            self.guidance_model.eval()
    
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
        
        # Convert to tensor and add batch dimension
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).float().unsqueeze(0)
        
        return frame_tensor
    
    def update_frame_buffer(self, frame_tensor):
        """
        Update the frame buffer with a new frame tensor.
        
        Args:
            frame_tensor: New frame tensor to add
        """
        self.frame_buffer.append(frame_tensor)
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
    
    def get_sequence_tensor(self):
        """
        Get sequence tensor from frame buffer.
        
        Returns:
            Tensor of shape [1, seq_len, C, H, W]
        """
        if not self.frame_buffer:
            return None
        
        # Stack frames along sequence dimension
        sequence_tensor = torch.cat(self.frame_buffer, dim=0).unsqueeze(0)
        
        return sequence_tensor
    
    def analyze_frame(self, frame):
        """
        Perform comprehensive analysis on a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with analysis results
        """
        with self.inference_lock:
            # Preprocess frame
            frame_tensor = self.preprocess_frame(frame)
            
            # Move to device
            frame_tensor = frame_tensor.to(self.device)
            
            # Update frame buffer
            self.update_frame_buffer(frame_tensor)
            
            # Get sequence tensor if enough frames are available
            sequence_tensor = self.get_sequence_tensor()
            
            # Initialize results dictionary
            results = {
                'phase': None,
                'tools': None,
                'mistake': None,
                'risk_level': None,
                'guidance': None
            }
            
            with torch.no_grad():
                # Detect tools
                tool_detections = self.tool_model.predict(
                    frame_tensor, 
                    confidence_threshold=self.config['model']['tool_detection']['confidence_threshold']
                )
                
                # Recognize phase if enough frames are available
                if sequence_tensor is not None and sequence_tensor.size(1) >= 3:
                    phase_results = self.phase_model.predict(sequence_tensor, smooth=True)
                    results['phase'] = {
                        'name': phase_results['phase_names'][0][-1],  # Latest frame
                        'confidence': float(phase_results['probabilities'][0][-1])  # Latest frame
                    }
                
                # Extract tool detection results
                if tool_detections:
                    results['tools'] = []
                    for i, (label, score) in enumerate(zip(tool_detections[0]['labels'], tool_detections[0]['scores'])):
                        if label > 0:  # Skip background
                            results['tools'].append({
                                'name': tool_detections[0]['class_names'][i],
                                'confidence': float(score),
                                'box': tool_detections[0]['boxes'][i].tolist()
                            })
                
                # Detect mistakes if enough frames are available
                if sequence_tensor is not None and sequence_tensor.size(1) >= 5:
                    # Convert tool detections to torch tensor format for mistake model
                    tool_tensor_format = {
                        'boxes': torch.tensor(tool_detections[0]['boxes']).unsqueeze(0).to(self.device),
                        'scores': torch.tensor(tool_detections[0]['scores']).unsqueeze(0).to(self.device),
                        'labels': torch.tensor(tool_detections[0]['labels']).unsqueeze(0).to(self.device)
                    }
                    
                    mistake_results = self.mistake_model.predict(sequence_tensor, tool_tensor_format)
                    
                    # Only report mistakes if confidence is high enough
                    if mistake_results['mistake_indices'][0] > 0:  # Not 'no_mistake'
                        mistake_confidence = mistake_results['mistake_probabilities'][0][mistake_results['mistake_indices'][0]]
                        
                        if mistake_confidence > 0.5:  # Confidence threshold
                            results['mistake'] = {
                                'name': mistake_results['mistake_names'][0],
                                'confidence': float(mistake_confidence),
                                'risk_level': float(mistake_results['risk_levels'][0]),
                                'risk_description': mistake_results['risk_descriptions'][0]
                            }
                            
                            # Generate explanation
                            explanation = self.mistake_model.explain_prediction(
                                mistake_results['mistake_indices'][0],
                                mistake_results['risk_levels'][0]
                            )
                            
                            results['mistake']['explanation'] = explanation
                
                # Generate guidance if GPT model is available
                if self.use_gpt and sequence_tensor is not None:
                    # Use the last frame for guidance
                    last_frame = sequence_tensor[:, -1]
                    
                    # Generate guidance
                    guidance_text = self.guidance_model.generate_guidance(
                        last_frame,
                        detect_mistakes=(results['mistake'] is not None)
                    )
                    
                    results['guidance'] = guidance_text
            
            return results
    
    def process_video(self, video_path, output_path=None, frame_rate=1):
        """
        Process a video file with the SurgicalAI system.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
            frame_rate: Number of frames per second to process
            
        Returns:
            List of analysis results for processed frames
        """
        self.logger.info(f"Processing video: {video_path}")
        
        # Load video frames
        frames = load_video(video_path, frame_rate=frame_rate)
        
        # Process each frame
        results = []
        for i, frame in enumerate(frames):
            self.logger.info(f"Processing frame {i+1}/{len(frames)}")
            
            # Analyze frame
            frame_results = self.analyze_frame(frame)
            results.append(frame_results)
            
            # Save visualizations if output path is provided
            if output_path:
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Visualize results on frame
                visualization = self.visualize_results(frame, frame_results)
                
                # Save visualization
                output_file = os.path.join(
                    os.path.dirname(output_path),
                    f"{os.path.splitext(os.path.basename(output_path))[0]}_{i:04d}.jpg"
                )
                cv2.imwrite(output_file, visualization)
        
        self.logger.info(f"Processed {len(frames)} frames")
        
        return results
    
    def visualize_results(self, frame, results):
        """
        Visualize analysis results on a frame.
        
        Args:
            frame: Input frame (BGR format)
            results: Analysis results from analyze_frame()
            
        Returns:
            Visualization frame (BGR format)
        """
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Draw tool detections
        if results['tools']:
            for tool in results['tools']:
                # Get bounding box
                box = tool['box']
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Draw rectangle
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{tool['name']}: {tool['confidence']:.2f}"
                cv2.putText(vis_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw phase recognition
        if results['phase']:
            phase_label = f"Phase: {results['phase']['name']} ({results['phase']['confidence']:.2f})"
            cv2.putText(vis_frame, phase_label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw mistake detection
        if results['mistake']:
            mistake_label = f"Warning: {results['mistake']['name']}"
            cv2.putText(vis_frame, mistake_label, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            risk_label = f"Risk: {results['mistake']['risk_description']}"
            cv2.putText(vis_frame, risk_label, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw guidance
        if results['guidance']:
            # Split guidance into multiple lines
            guidance_lines = [results['guidance'][i:i+60] for i in range(0, len(results['guidance']), 60)]
            
            for i, line in enumerate(guidance_lines):
                cv2.putText(vis_frame, line, (10, vis_frame.shape[0] - 30 * (len(guidance_lines) - i)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return vis_frame


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
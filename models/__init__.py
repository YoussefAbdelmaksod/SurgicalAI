"""
SurgicalAI models module.

This module contains all the model implementations for the SurgicalAI system.
"""

import os
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Import core models
from .phase_recognition import ViTLSTM, ViTTransformerTemporal, get_vit_lstm_model
from .tool_detection import AdvancedToolDetectionModel, ToolDetectionEnsemble, get_tool_detection_model
from .mistake_detection import SurgicalMistakeDetector, GPTSurgicalAssistant, get_mistake_detector_model
from .ensemble import ModelEnsemble, MultiModalFusion
from .base_model import BaseModel

# Try to import optional modules
try:
    from .voice_assistant import VoiceAssistant
except ImportError:
    logger.warning("pyttsx3 not installed. Voice assistant will not be available.")
    VoiceAssistant = None

# Try to import GPT guidance
try:
    from .gpt_guidance import GPTGuidance
except ImportError:
    logger.warning("transformers package not installed. GPT-based models will not be available.")
    GPTGuidance = None

# Define the paths to model weights
WEIGHTS_DIR = Path(__file__).parent / "weights"
PHASE_WEIGHTS_DIR = WEIGHTS_DIR / "phase"
TOOL_WEIGHTS_DIR = WEIGHTS_DIR / "tool"
MISTAKE_WEIGHTS_DIR = WEIGHTS_DIR / "mistake"

def check_model_weights(required_only=True):
    """
    Check if model weights exist in the expected locations.
    
    Args:
        required_only: Only check for essential model weights
        
    Returns:
        dict: Status of model weights (exists/missing)
    """
    weight_status = {
        "phase_recognition": False,
        "tool_detection": False,
        "mistake_detection": False
    }
    
    # Check for phase recognition weights
    phase_weight_path = PHASE_WEIGHTS_DIR / "phase_recognition.pth"
    if phase_weight_path.exists():
        weight_status["phase_recognition"] = True
    
    # Check for tool detection weights
    tool_weight_path = TOOL_WEIGHTS_DIR / "tool_detection.pth"
    if tool_weight_path.exists():
        weight_status["tool_detection"] = True
    
    # Check for mistake detection weights
    mistake_weight_path = MISTAKE_WEIGHTS_DIR / "mistake_detection.pth"
    if mistake_weight_path.exists():
        weight_status["mistake_detection"] = True
    
    # Log status
    for model, exists in weight_status.items():
        if exists:
            logger.info(f"{model} weights found")
        else:
            logger.warning(f"{model} weights not found")
    
    return weight_status

# Create weight directories if they don't exist
os.makedirs(PHASE_WEIGHTS_DIR, exist_ok=True)
os.makedirs(TOOL_WEIGHTS_DIR, exist_ok=True)
os.makedirs(MISTAKE_WEIGHTS_DIR, exist_ok=True)

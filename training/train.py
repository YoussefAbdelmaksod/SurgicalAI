"""
Main training script for SurgicalAI models.

This script coordinates the training of all models in the SurgicalAI system.
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
import torch
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.phase_recognition_trainer import PhaseRecognitionTrainer
from training.tool_detection_trainer import ToolDetectionTrainer
from training.mistake_detection_trainer import MistakeDetectionTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training/logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SurgicalAI models')
    
    parser.add_argument('--config', type=str, default='training/configs/training_config.yaml',
                        help='Path to training configuration file')
    
    parser.add_argument('--models', type=str, nargs='+', 
                        choices=['phase', 'tool', 'mistake', 'all'],
                        default=['all'],
                        help='Models to train: phase, tool, mistake, or all')
    
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    
    parser.add_argument('--phase-ckpt', type=str, default=None,
                        help='Path to phase recognition checkpoint for resuming training')
    
    parser.add_argument('--tool-ckpt', type=str, default=None,
                        help='Path to tool detection checkpoint for resuming training')
    
    parser.add_argument('--mistake-ckpt', type=str, default=None,
                        help='Path to mistake detection checkpoint for resuming training')
    
    return parser.parse_args()


def train_phase_recognition(config_path, resume_checkpoint=None):
    """
    Train phase recognition model.
    
    Args:
        config_path: Path to configuration file
        resume_checkpoint: Optional path to checkpoint for resuming training
        
    Returns:
        Path to best model checkpoint
    """
    logger.info("Starting phase recognition training...")
    
    trainer = PhaseRecognitionTrainer(config_path)
    
    if resume_checkpoint:
        logger.info(f"Resuming phase recognition training from {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint)
    
    best_model_path = trainer.train()
    
    logger.info(f"Phase recognition training completed. Best model: {best_model_path}")
    
    return best_model_path


def train_tool_detection(config_path, resume_checkpoint=None):
    """
    Train tool detection model.
    
    Args:
        config_path: Path to configuration file
        resume_checkpoint: Optional path to checkpoint for resuming training
        
    Returns:
        Path to best model checkpoint
    """
    logger.info("Starting tool detection training...")
    
    trainer = ToolDetectionTrainer(config_path)
    
    if resume_checkpoint:
        logger.info(f"Resuming tool detection training from {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint)
    
    best_model_path = trainer.train()
    
    logger.info(f"Tool detection training completed. Best model: {best_model_path}")
    
    return best_model_path


def train_mistake_detection(config_path, resume_checkpoint=None):
    """
    Train mistake detection model.
    
    Args:
        config_path: Path to configuration file
        resume_checkpoint: Optional path to checkpoint for resuming training
        
    Returns:
        Path to best model checkpoint
    """
    logger.info("Starting mistake detection training...")
    
    trainer = MistakeDetectionTrainer(config_path)
    
    if resume_checkpoint:
        logger.info(f"Resuming mistake detection training from {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint)
    
    best_model_path = trainer.train()
    
    logger.info(f"Mistake detection training completed. Best model: {best_model_path}")
    
    return best_model_path


def main():
    """Main function to run training."""
    args = parse_args()
    
    # Check if CUDA is available
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Create necessary directories
    os.makedirs('training/logs', exist_ok=True)
    os.makedirs('training/checkpoints', exist_ok=True)
    
    # Determine which models to train
    models_to_train = args.models
    if 'all' in models_to_train:
        models_to_train = ['phase', 'tool', 'mistake']
    
    results = {}
    
    # Train phase recognition model
    if 'phase' in models_to_train:
        phase_ckpt = args.phase_ckpt if args.resume else None
        results['phase'] = train_phase_recognition(args.config, phase_ckpt)
    
    # Train tool detection model
    if 'tool' in models_to_train:
        tool_ckpt = args.tool_ckpt if args.resume else None
        results['tool'] = train_tool_detection(args.config, tool_ckpt)
    
    # Train mistake detection model
    if 'mistake' in models_to_train:
        mistake_ckpt = args.mistake_ckpt if args.resume else None
        results['mistake'] = train_mistake_detection(args.config, mistake_ckpt)
    
    # Print summary of trained models
    logger.info("Training completed. Summary of trained models:")
    for model_name, model_path in results.items():
        logger.info(f"  {model_name}: {model_path}")
    
    # Create a model config file with paths to best models
    model_config = {
        'phase_recognition': str(results.get('phase', '')),
        'tool_detection': str(results.get('tool', '')),
        'mistake_detection': str(results.get('mistake', ''))
    }
    
    model_config_path = Path('training/checkpoints/model_paths.yaml')
    with open(model_config_path, 'w') as f:
        yaml.dump(model_config, f)
    
    logger.info(f"Model paths saved to {model_config_path}")


if __name__ == '__main__':
    main() 
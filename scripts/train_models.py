#!/usr/bin/env python
"""
Script for training SurgicalAI models.

This script provides a command-line interface for training the models
in the SurgicalAI system, either individually or all together.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import torch
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import (
    train_phase_recognition,
    train_tool_detection,
    train_mistake_detection
)
from utils.helpers import setup_logging, load_config

def parse_args():
    """Parse command-line arguments."""
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
    
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Root directory for datasets')
    
    parser.add_argument('--output-dir', type=str, default='training/checkpoints',
                      help='Directory to save model checkpoints')
    
    parser.add_argument('--log-dir', type=str, default='training/logs',
                      help='Directory to save training logs')
    
    parser.add_argument('--no-cuda', action='store_true',
                      help='Disable CUDA even if available')
    
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of data loading workers')
    
    parser.add_argument('--epochs', type=int, default=None,
                      help='Override number of epochs in config')
    
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Override batch size in config')
    
    parser.add_argument('--lr', type=float, default=None,
                      help='Override learning rate in config')
    
    return parser.parse_args()

def create_directories(args):
    """Create necessary directories for training."""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create model-specific directories
    os.makedirs(os.path.join(args.output_dir, 'phase_recognition'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'tool_detection'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'mistake_detection'), exist_ok=True)
    
    # Create timestamp directory for logs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'train_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    return log_dir

def override_config(config, args):
    """Override configuration with command-line arguments."""
    # Override common parameters
    if args.epochs:
        for section in ['phase_recognition', 'tool_detection', 'mistake_detection']:
            if section in config and 'training' in config[section]:
                config[section]['training']['epochs'] = args.epochs
    
    if args.batch_size:
        for section in ['phase_recognition', 'tool_detection', 'mistake_detection']:
            if section in config and 'training' in config[section]:
                config[section]['training']['batch_size'] = args.batch_size
    
    if args.lr:
        for section in ['phase_recognition', 'tool_detection', 'mistake_detection']:
            if section in config and 'training' in config[section]:
                config[section]['training']['learning_rate'] = args.lr
    
    # Override general parameters
    config['general']['cuda'] = torch.cuda.is_available() and not args.no_cuda
    config['general']['seed'] = args.seed
    config['general']['num_workers'] = args.workers
    
    return config

def main():
    """Main function to run training."""
    # Parse command-line arguments
    args = parse_args()
    
    # Create necessary directories
    log_dir = create_directories(args)
    
    # Set up logging
    setup_logging(log_dir=log_dir)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command-line arguments
    config = override_config(config, args)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available() and not args.no_cuda
    logger.info(f"CUDA available: {cuda_available}")
    if cuda_available:
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Save modified configuration
    modified_config_path = os.path.join(log_dir, 'training_config.yaml')
    with open(modified_config_path, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Modified configuration saved to {modified_config_path}")
    
    # Determine which models to train
    models_to_train = args.models
    if 'all' in models_to_train:
        models_to_train = ['phase', 'tool', 'mistake']
    
    # Training results
    results = {}
    
    # Train phase recognition model
    if 'phase' in models_to_train:
        logger.info("Starting phase recognition training...")
        phase_ckpt = args.phase_ckpt if args.resume else None
        results['phase'] = train_phase_recognition(args.config, phase_ckpt)
        logger.info(f"Phase recognition training completed. Best model: {results['phase']}")
    
    # Train tool detection model
    if 'tool' in models_to_train:
        logger.info("Starting tool detection training...")
        tool_ckpt = args.tool_ckpt if args.resume else None
        results['tool'] = train_tool_detection(args.config, tool_ckpt)
        logger.info(f"Tool detection training completed. Best model: {results['tool']}")
    
    # Train mistake detection model
    if 'mistake' in models_to_train:
        logger.info("Starting mistake detection training...")
        mistake_ckpt = args.mistake_ckpt if args.resume else None
        results['mistake'] = train_mistake_detection(args.config, mistake_ckpt)
        logger.info(f"Mistake detection training completed. Best model: {results['mistake']}")
    
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
    
    # Save model paths
    model_config_path = os.path.join(args.output_dir, 'model_paths.yaml')
    with open(model_config_path, 'w') as f:
        yaml.dump(model_config, f)
    
    logger.info(f"Model paths saved to {model_config_path}")

if __name__ == '__main__':
    main() 
#!/usr/bin/env python
"""
Comprehensive training script for all SurgicalAI models.

This script provides a unified training pipeline for all models in the SurgicalAI system:
1. Tool Detection (Faster R-CNN)
2. Phase Recognition (ViT-LSTM)
3. Mistake Detection
4. GPT-based Surgical Assistant

This is the main training entrypoint that should be used for all model training.

Usage:
    python training/train_all_models.py --config config/training_config.yaml --data_dir data
"""

import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from datetime import datetime
from tqdm import tqdm
import shutil
import torch.nn as nn
from sklearn.metrics import f1_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from models.tool_detection import AdvancedToolDetectionModel
from models.phase_recognition import ViTLSTM, ViTTransformerTemporal
from models.mistake_detection import SurgicalMistakeDetector, GPTSurgicalAssistant

# Import datasets and utilities
from data.coco_dataset import COCOSurgicalToolDataset
from data.augmentation import create_transforms
from utils.helpers import load_config, setup_logging, get_device
from utils.lr_scheduler import get_scheduler
from utils.performance_evaluation import SurgicalPerformanceEvaluator

# Import training utilities
from training.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train all SurgicalAI models')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='models/checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--train_subset', nargs='+', default=['all'],
                        choices=['all', 'tool_detection', 'phase_recognition', 'mistake_detection', 'gpt_assistant'],
                        help='Which model subset to train')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with limited data')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--no_validation', action='store_true',
                        help='Skip validation (useful for full dataset training)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_config(config, output_dir, filename='training_config.yaml'):
    """Save configuration to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, filename)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_timestamp_dir(base_dir):
    """Create a timestamped directory for this training run."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_custom_collate_fn(model_type):
    """Get collate function specific to model type."""
    if model_type == 'tool_detection':
        # For object detection models
        return lambda batch: tuple(zip(*batch))
    else:
        # Default collate function for other models
        return None


def train_tool_detection_model(config, data_dir, output_dir, device, logger, debug=False, resume=False):
    """Train the tool detection model."""
    logger.info("=== Training Tool Detection Model ===")
    
    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, 'tool_detection')
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Define image size and batch size
    image_size = config['model']['tool_detection'].get('image_size', 800)
    batch_size = config['training']['tool_detection'].get('batch_size', 8)
    if debug:
        batch_size = min(batch_size, 2)
    
    # Create transformations
    augmentation_level = config['training']['tool_detection'].get('augmentation_level', 'medium')
    train_transforms = create_transforms(augmentation_level, (image_size, image_size))
    val_transforms = create_transforms('none', (image_size, image_size))
    
    # Create datasets and dataloaders
    train_dataset = COCOSurgicalToolDataset(
        os.path.join(data_dir, 'train'), 
        transform=train_transforms,
        training=True
    )
    
    # Create smaller dataset for debug mode
    if debug:
        indices = torch.randperm(len(train_dataset))[:min(len(train_dataset), 32)]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=get_custom_collate_fn('tool_detection'),
        num_workers=config['training'].get('num_workers', 2),
        pin_memory=True
    )
    
    # Validation dataset and dataloader
    val_dataset = None
    val_loader = None
    
    if not config['training'].get('no_validation', False):
        val_dataset = COCOSurgicalToolDataset(
            os.path.join(data_dir, 'valid'), 
            transform=val_transforms,
            training=False
        )
        
        if debug:
            indices = torch.randperm(len(val_dataset))[:min(len(val_dataset), 16)]
            val_dataset = torch.utils.data.Subset(val_dataset, indices)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=get_custom_collate_fn('tool_detection'),
            num_workers=config['training'].get('num_workers', 2),
            pin_memory=True
        )
    
    # Create model
    backbone = config['model']['tool_detection'].get('backbone', 'resnet50')
    model = AdvancedToolDetectionModel(
        num_classes=8,  # Background + 7 tool classes
        architecture='faster_rcnn',
        backbone_name=backbone,
        pretrained=True,
        use_fpn=True
    )
    model.to(device)
    
    # Create optimizer
    learning_rate = config['training']['tool_detection'].get('learning_rate', 0.001)
    weight_decay = config['training']['tool_detection'].get('weight_decay', 0.0001)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler
    scheduler_config = config['training']['tool_detection'].get('lr_scheduler', {})
    scheduler = None
    
    if scheduler_config.get('use_scheduler', True):
        scheduler_type = scheduler_config.get('scheduler_type', 'cosine')
        num_epochs = config['training']['tool_detection'].get('epochs', 30)
        
        scheduler_kwargs = {
            'T_max': num_epochs,
            'eta_min': scheduler_config.get('eta_min', 0.00001)
        }
        
        if scheduler_type == 'warmup_cosine':
            scheduler_kwargs['warmup_epochs'] = scheduler_config.get('warmup_epochs', 3)
            scheduler_kwargs['max_epochs'] = num_epochs
            scheduler_kwargs['warmup_start_lr'] = scheduler_config.get('warmup_start_lr', 0.000001)
        
        scheduler = get_scheduler(scheduler_type, optimizer, **scheduler_kwargs)
    
    # Create custom trainer class
    class ToolDetectionTrainer(Trainer):
        def train_epoch(self, train_loader):
            """Train the model for one epoch."""
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} (Training)")
            self.model.train()
            
            for images, targets in progress_bar:
                # Move data to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                          for k, v in t.items()} for t in targets]
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass and optimize
                losses.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Update metrics
                total_loss += losses.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': losses.item(),
                    'avg_loss': total_loss / num_batches
                })
            
            # Calculate average loss
            avg_loss = total_loss / max(1, num_batches)
            
            return {'loss': avg_loss}
        
        def validate(self, val_loader):
            """Validate the model."""
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(val_loader, desc=f"Epoch {self.current_epoch} (Validation)")
            self.model.eval()
            
            with torch.no_grad():
                for images, targets in progress_bar:
                    # Move data to device
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                              for k, v in t.items()} for t in targets]
                    
                    # Forward pass
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    # Update metrics
                    total_loss += losses.item()
                    num_batches += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'val_loss': losses.item(),
                        'avg_val_loss': total_loss / num_batches
                    })
            
            # Calculate average loss
            avg_loss = total_loss / max(1, num_batches)
            
            return {'loss': avg_loss}
    
    # Create trainer
    num_epochs = config['training']['tool_detection'].get('epochs', 30)
    trainer = ToolDetectionTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        checkpoint_dir=model_output_dir,
        early_stopping_patience=config['training'].get('early_stopping_patience', 10)
    )
    
    # Resume training if requested
    if resume:
        checkpoint_path = os.path.join(model_output_dir, 'best_model.pth')
        if os.path.exists(checkpoint_path):
            logger.info(f"Resuming training from {checkpoint_path}")
            trainer.load_checkpoint(checkpoint_path)
    
    # Train the model
    trainer.train(
        num_epochs=num_epochs,
        val_frequency=config['training'].get('val_frequency', 1),
        primary_val_metric='loss',
        val_metric_higher_is_better=False,
        save_best_only=config['training'].get('save_best_only', True)
    )
    
    logger.info("Tool Detection Model training complete!")
    return model, trainer


def train_phase_recognition_model(config, data_dir, output_dir, device, logger, debug=False, resume=False):
    """Train the phase recognition model."""
    logger.info("=== Training Phase Recognition Model ===")
    
    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, 'phase_recognition')
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Implementation of phase recognition training
    # This would be similar to tool detection but with different datasets and models
    # For now, let's create a placeholder that returns None
    
    # TODO: Implement full phase recognition training
    logger.info("Phase Recognition training will be implemented in a future update")
    return None, None


def train_mistake_detection_model(config, data_dir, output_dir, device, logger, debug=False, resume=False):
    """Train the mistake detection model."""
    logger.info("=== Training Mistake Detection Model ===")
    
    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, 'mistake_detection')
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Create datasets
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    
    # Create mistake detection model
    model_config = config['model']['mistake_detection']
    model = SurgicalMistakeDetector(
        visual_dim=model_config.get('visual_dim', 768),
        tool_dim=model_config.get('tool_dim', 128),
        num_tools=model_config.get('num_tools', 10),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_classes=model_config.get('num_classes', 3),
        use_temporal=model_config.get('use_temporal', True),
        dropout=model_config.get('dropout', 0.3)
    ).to(device)
    
    logger.info(f"Created mistake detection model with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Create datasets
    from data.dataset import MistakeDetectionDataset
    
    train_dataset = MistakeDetectionDataset(
        root=train_dir,
        sequence_length=config['data'].get('sequence_length', 16),
        temporal_stride=config['data'].get('temporal_stride', 2),
        transforms=True,
        use_augmentation=config['data'].get('use_augmentation', True),
        image_size=config['data'].get('image_size', 224)
    )
    
    val_dataset = MistakeDetectionDataset(
        root=val_dir,
        sequence_length=config['data'].get('sequence_length', 16),
        temporal_stride=config['data'].get('temporal_stride', 2),
        transforms=True,
        use_augmentation=False,
        image_size=config['data'].get('image_size', 224)
    )
    
    # For debugging, limit dataset size
    if debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(32, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(16, len(val_dataset))))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['mistake_detection'].get('batch_size', 8),
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['mistake_detection'].get('batch_size', 8),
        shuffle=False,
        num_workers=config['data'].get('num_workers', 2),
        pin_memory=True
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['mistake_detection'].get('learning_rate', 0.001),
        weight_decay=config['training']['mistake_detection'].get('weight_decay', 0.01)
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create loss function
    # For mistake classification
    mistake_criterion = nn.CrossEntropyLoss()
    # For risk regression
    risk_criterion = nn.MSELoss()
    
    # Define combined loss function
    def combined_loss_fn(outputs, targets):
        mistake_loss = mistake_criterion(outputs['mistake_logits'], targets['mistake_labels'])
        risk_loss = risk_criterion(outputs['risk_scores'], targets['risk_scores'])
        return mistake_loss + risk_loss
    
    # Create training class
    class MistakeDetectionTrainer(Trainer):
        def train_epoch(self, train_loader):
            """Train one epoch."""
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            # Create progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Extract inputs and targets
                visual_features = batch['visual_features'].to(self.device)
                tool_ids = batch['tool_ids'].to(self.device)
                mistake_labels = batch['mistake_labels'].to(self.device)
                risk_scores = batch['risk_scores'].to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(visual_features, tool_ids)
                
                # Calculate loss
                loss = combined_loss_fn(
                    outputs, 
                    {'mistake_labels': mistake_labels, 'risk_scores': risk_scores}
                )
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                
                # Calculate accuracy for mistake classification
                _, predicted = outputs['mistake_logits'].max(1)
                total += mistake_labels.size(0)
                correct += predicted.eq(mistake_labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{total_loss / (batch_idx + 1):.4f}",
                    'acc': f"{100. * correct / total:.2f}%"
                })
            
            # Return metrics
            metrics = {
                'loss': total_loss / len(train_loader),
                'accuracy': correct / total
            }
            
            return metrics
        
        def validate(self, val_loader):
            """Validate the model."""
            self.model.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc=f"Epoch {self.current_epoch} [Val]")
                
                for batch_idx, batch in enumerate(progress_bar):
                    # Extract inputs and targets
                    visual_features = batch['visual_features'].to(self.device)
                    tool_ids = batch['tool_ids'].to(self.device)
                    mistake_labels = batch['mistake_labels'].to(self.device)
                    risk_scores = batch['risk_scores'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(visual_features, tool_ids)
                    
                    # Calculate loss
                    loss = combined_loss_fn(
                        outputs, 
                        {'mistake_labels': mistake_labels, 'risk_scores': risk_scores}
                    )
                    
                    # Update statistics
                    total_loss += loss.item()
                    
                    # Calculate accuracy for mistake classification
                    _, predicted = outputs['mistake_logits'].max(1)
                    total += mistake_labels.size(0)
                    correct += predicted.eq(mistake_labels).sum().item()
                    
                    # Save predictions for F1 score calculation
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(mistake_labels.cpu().numpy())
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{total_loss / (batch_idx + 1):.4f}",
                        'acc': f"{100. * correct / total:.2f}%"
                    })
            
            # Calculate F1 score
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            
            # Return metrics
            metrics = {
                'loss': total_loss / len(val_loader),
                'accuracy': correct / total,
                'f1_score': f1
            }
            
            return metrics
    
    # Create trainer
    trainer = MistakeDetectionTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        checkpoint_dir=model_output_dir,
        early_stopping_patience=config['training'].get('early_stopping_patience', 10)
    )
    
    # Load checkpoint if resuming
    if resume:
        checkpoint_path = os.path.join(model_output_dir, 'best_model.pth')
        if os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
            logger.info(f"Resumed from checkpoint: {checkpoint_path}")
    
    # Train model
    best_model, best_metrics = trainer.train(
        num_epochs=config['training']['mistake_detection'].get('epochs', 30)
    )
    
    # Save final model
    final_path = os.path.join(model_output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final model to {final_path}")
    
    # Create best model path for system to use
    best_model_path = os.path.join(model_output_dir, 'mistake_detection.pth')
    shutil.copyfile(
        os.path.join(model_output_dir, 'best_model.pth'),
        best_model_path
    )
    logger.info(f"Copied best model to {best_model_path}")
    
    return best_model, best_metrics


def train_gpt_assistant_model(config, data_dir, output_dir, device, logger, debug=False, resume=False):
    """Train the GPT-based surgical assistant model."""
    logger.info("=== Training GPT-based Surgical Assistant Model ===")
    
    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, 'gpt_assistant')
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Implementation of GPT assistant training
    # This would be similar to tool detection but with different datasets and models
    # For now, let's create a placeholder that returns None
    
    # TODO: Implement full GPT assistant training
    logger.info("GPT Assistant training will be implemented in a future update")
    return None, None


def main():
    """Main function to train all models."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting comprehensive SurgicalAI training pipeline")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory with timestamp
    timestamp_dir = create_timestamp_dir(args.output_dir)
    logger.info(f"Saving outputs to {timestamp_dir}")
    
    # Save configuration
    save_config(config, timestamp_dir)
    
    # Set device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Update config with command line arguments
    config['training']['debug'] = args.debug
    config['training']['no_validation'] = args.no_validation
    
    # Determine which models to train
    train_models = args.train_subset
    if 'all' in train_models:
        train_models = ['tool_detection', 'phase_recognition', 'mistake_detection', 'gpt_assistant']
    
    # Train selected models
    trained_models = {}
    
    if 'tool_detection' in train_models:
        model, trainer = train_tool_detection_model(
            config=config,
            data_dir=args.data_dir,
            output_dir=timestamp_dir,
            device=device,
            logger=logger,
            debug=args.debug,
            resume=args.resume
        )
        trained_models['tool_detection'] = {'model': model, 'trainer': trainer}
    
    if 'phase_recognition' in train_models:
        model, trainer = train_phase_recognition_model(
            config=config,
            data_dir=args.data_dir,
            output_dir=timestamp_dir,
            device=device,
            logger=logger,
            debug=args.debug,
            resume=args.resume
        )
        trained_models['phase_recognition'] = {'model': model, 'trainer': trainer}
    
    if 'mistake_detection' in train_models:
        model, trainer = train_mistake_detection_model(
            config=config,
            data_dir=args.data_dir,
            output_dir=timestamp_dir,
            device=device,
            logger=logger,
            debug=args.debug,
            resume=args.resume
        )
        trained_models['mistake_detection'] = {'model': model, 'trainer': trainer}
    
    if 'gpt_assistant' in train_models:
        model, trainer = train_gpt_assistant_model(
            config=config,
            data_dir=args.data_dir,
            output_dir=timestamp_dir,
            device=device,
            logger=logger,
            debug=args.debug,
            resume=args.resume
        )
        trained_models['gpt_assistant'] = {'model': model, 'trainer': trainer}
    
    logger.info("Training complete! All selected models have been trained.")
    logger.info(f"Model checkpoints are saved in {timestamp_dir}")
    
    return trained_models


if __name__ == "__main__":
    main() 
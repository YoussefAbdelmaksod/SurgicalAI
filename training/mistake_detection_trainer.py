"""
Training module for surgical mistake detection.

This module implements the training pipeline for the mistake detection model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import logging
import yaml
from pathlib import Path
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import torch.cuda.amp as amp

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mistake_detection import SurgicalMistakeDetector
from training.surgical_datasets import get_dataloader

logger = logging.getLogger(__name__)

class MistakeDetectionTrainer:
    """
    Trainer for surgical mistake detection model.
    """
    
    def __init__(self, config_path):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to training configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up device
        if self.config['general']['cuda'] and torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU for training")
        
        # Set random seed for reproducibility
        self.seed = self.config['general']['seed']
        self._set_seed(self.seed)
        
        # Create checkpoint directory
        self.checkpoint_dir = Path('training/checkpoints/mistake_detection')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Set up logging directory
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path(f'training/logs/mistake_detection_{current_time}')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Initialize model
        self._init_model()
        
        # Initialize optimizer and scheduler
        self._init_optimizer()
        
        # Initialize dataloaders
        self._init_dataloaders()
        
        # Initialize mixed precision training
        self.scaler = amp.GradScaler(enabled=self.config['general']['mixed_precision'])
        
        # Initialize metrics tracking
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        
        logger.info("Mistake detection trainer initialized")
    
    def _set_seed(self, seed):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _init_model(self):
        """Initialize the mistake detection model."""
        model_config = self.config['mistake_detection']['model']
        
        self.model = SurgicalMistakeDetector(
            visual_dim=model_config['visual_dim'],
            tool_dim=model_config['tool_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_classes=model_config['num_classes'],
            use_temporal=model_config['use_temporal']
        ).to(self.device)
        
        # Initialize criteria
        self.classification_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss()
        
        logger.info(f"Initialized mistake detection model with {model_config['num_classes']} classes")
    
    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        train_config = self.config['mistake_detection']['training']
        
        # Initialize optimizer
        if train_config['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay']
            )
        elif train_config['optimizer'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay']
            )
        elif train_config['optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=train_config['learning_rate'],
                momentum=0.9,
                weight_decay=train_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")
        
        # Initialize scheduler
        if train_config['lr_scheduler'].lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=train_config['epochs'],
                eta_min=train_config['learning_rate'] * 0.01
            )
        elif train_config['lr_scheduler'].lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=10, 
                gamma=0.1
            )
        elif train_config['lr_scheduler'].lower() == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=0.1, 
                patience=5
            )
        else:
            self.scheduler = None
        
        logger.info(f"Initialized {train_config['optimizer']} optimizer with "
                   f"learning rate {train_config['learning_rate']}")
    
    def _init_dataloaders(self):
        """Initialize training and validation dataloaders."""
        data_config = self.config['mistake_detection']['data']
        train_config = self.config['mistake_detection']['training']
        
        # Determine dataset path
        data_dir = Path(f"data/{data_config['dataset']}")
        
        # Create dataloaders
        self.dataloaders = get_dataloader(
            dataset_name=data_config['dataset'],
            data_dir=str(data_dir),
            batch_size=train_config['batch_size'],
            num_workers=self.config['general']['num_workers'],
            sequence_length=5,  # Fixed sequence length for mistake detection
            use_synthetic=data_config.get('use_synthetic_data', False),
            synthetic_ratio=data_config.get('synthetic_ratio', 0.3)
        )
        
        logger.info(f"Initialized dataloaders for {data_config['dataset']} dataset")
    
    def train(self):
        """
        Train the mistake detection model.
        
        Returns:
            Path to the best model checkpoint
        """
        train_config = self.config['mistake_detection']['training']
        num_epochs = train_config['epochs']
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc, train_f1 = self._train_epoch(epoch)
            
            # Validation phase
            if (epoch + 1) % self.config['general']['val_interval'] == 0:
                val_loss, val_acc, val_f1, conf_matrix = self._validate_epoch(epoch)
                
                # Log validation results
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('F1/val', val_f1, epoch)
                
                # Plot and log confusion matrix
                if epoch % 5 == 0:
                    cm_fig = self._plot_confusion_matrix(conf_matrix)
                    self.writer.add_figure('Confusion Matrix', cm_fig, epoch)
                
                # Update learning rate if using ReduceLROnPlateau
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_f1)
                
                # Save model if it's the best so far
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    checkpoint_path = self.checkpoint_dir / f'best_model_epoch_{epoch}.pth'
                    self._save_checkpoint(checkpoint_path, epoch, val_f1)
                    logger.info(f"New best model saved with F1: {val_f1:.4f}")
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                
                # Early stopping
                if self.epochs_without_improvement >= train_config['early_stopping_patience']:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Update learning rate for epoch-based schedulers
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
                
            # Save checkpoint periodically
            if (epoch + 1) % self.config['general']['save_interval'] == 0:
                checkpoint_path = self.checkpoint_dir / f'model_epoch_{epoch}.pth'
                self._save_checkpoint(checkpoint_path, epoch, val_f1 if 'val_f1' in locals() else 0.0)
        
        # Save final model
        final_checkpoint_path = self.checkpoint_dir / 'final_model.pth'
        self._save_checkpoint(final_checkpoint_path, num_epochs-1, self.best_val_f1)
        
        logger.info(f"Training completed. Best validation F1: {self.best_val_f1:.4f}")
        
        # Close tensorboard writer
        self.writer.close()
        
        # Return path to best model
        best_model_path = self.checkpoint_dir / f'best_model.pth'
        return best_model_path
    
    def _train_epoch(self, epoch):
        """
        Train the model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average loss and metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Use tqdm to show progress bar
        train_loader = self.dataloaders['train']
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Get data and move to device
            frames = batch['frames'].to(self.device)  # [batch_size, seq_len, C, H, W]
            labels = batch['mistakes'].to(self.device)  # [batch_size, seq_len]
            
            # Check if masks are available (from EndoSurgical dataset)
            masks = None
            if 'masks' in batch:
                masks = batch['masks'].to(self.device)  # [batch_size, seq_len, num_classes, H, W]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with amp.autocast(enabled=self.config['general']['mixed_precision']):
                # If we have masks, use them as additional input
                if masks is not None:
                    # Reshape for batch processing
                    b, s, c, h, w = frames.shape
                    bm, sm, cm, hm, wm = masks.shape
                    
                    # Flatten batch and sequence dimensions for processing
                    frames_flat = frames.view(b * s, c, h, w)
                    masks_flat = masks.view(bm * sm, cm, hm, wm)
                    
                    # Create dummy tool features (since we don't have tool detections for supplementary data)
                    # This depends on your model's architecture
                    tool_features_flat = None
                    
                    # Get predictions
                    outputs_flat = self.model(frames_flat, tool_features=tool_features_flat, segmentation_masks=masks_flat)
                    
                    # Reshape back to batch and sequence dimensions
                    outputs = outputs_flat.view(b, s, -1)
                else:
                    # Standard forward pass without masks
                    outputs = self.model(frames)  # [batch_size, seq_len, num_classes]
                
                # Reshape for loss calculation
                b, s, c = outputs.shape
                outputs_flat = outputs.reshape(-1, c)
                labels_flat = labels.reshape(-1)
                
                # Calculate loss
                loss = self.classification_criterion(outputs_flat, labels_flat)
            
            # Backward pass with scaled gradients
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if 'grad_clip' in self.config['mistake_detection']['training']:
                clip_value = self.config['mistake_detection']['training']['grad_clip']
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            
            # Update weights with scaled gradients
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update total loss
            total_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(outputs_flat, dim=1).detach().cpu().numpy()
            
            # Update metrics
            all_preds.extend(preds)
            all_labels.extend(labels_flat.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard at intervals
            log_interval = self.config['general']['log_interval']
            if batch_idx % log_interval == 0:
                iteration = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), iteration)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        avg_loss = total_loss / len(train_loader)
        
        # Log metrics
        self.writer.add_scalar('train/accuracy', accuracy, epoch)
        self.writer.add_scalar('train/f1_score', f1, epoch)
        self.writer.add_scalar('train/loss', avg_loss, epoch)
        
        # Log learning rate
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/lr_group_{i}', param_group['lr'], epoch)
        
        logger.info(f"Epoch {epoch} [Train] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Update learning rate scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # For ReduceLROnPlateau, we update based on validation metrics
                pass
            else:
                self.scheduler.step()
        
        return avg_loss, accuracy, f1
    
    def _validate_epoch(self, epoch):
        """
        Validate the model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average validation loss and metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Use tqdm to show progress bar
        val_loader = self.dataloaders['val']
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                # Get data and move to device
                frames = batch['frames'].to(self.device)  # [batch_size, seq_len, C, H, W]
                labels = batch['mistakes'].to(self.device)  # [batch_size, seq_len]
                
                # Check if masks are available (from EndoSurgical dataset)
                masks = None
                if 'masks' in batch:
                    masks = batch['masks'].to(self.device)  # [batch_size, seq_len, num_classes, H, W]
                
                # Forward pass with mixed precision
                with amp.autocast(enabled=self.config['general']['mixed_precision']):
                    # If we have masks, use them as additional input
                    if masks is not None:
                        # Reshape for batch processing
                        b, s, c, h, w = frames.shape
                        bm, sm, cm, hm, wm = masks.shape
                        
                        # Flatten batch and sequence dimensions for processing
                        frames_flat = frames.view(b * s, c, h, w)
                        masks_flat = masks.view(bm * sm, cm, hm, wm)
                        
                        # Create dummy tool features (since we don't have tool detections for supplementary data)
                        # This depends on your model's architecture
                        tool_features_flat = None
                        
                        # Get predictions
                        outputs_flat = self.model(frames_flat, tool_features=tool_features_flat, segmentation_masks=masks_flat)
                        
                        # Reshape back to batch and sequence dimensions
                        outputs = outputs_flat.view(b, s, -1)
                    else:
                        # Standard forward pass without masks
                        outputs = self.model(frames)  # [batch_size, seq_len, num_classes]
                    
                    # Reshape for loss calculation
                    b, s, c = outputs.shape
                    outputs_flat = outputs.reshape(-1, c)
                    labels_flat = labels.reshape(-1)
                    
                    # Calculate loss
                    loss = self.classification_criterion(outputs_flat, labels_flat)
                
                # Update total loss
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(outputs_flat, dim=1).detach().cpu().numpy()
                
                # Update metrics
                all_preds.extend(preds)
                all_labels.extend(labels_flat.detach().cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        avg_loss = total_loss / len(val_loader)
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Log metrics
        self.writer.add_scalar('val/accuracy', accuracy, epoch)
        self.writer.add_scalar('val/f1_score', f1, epoch)
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        
        # Plot and log confusion matrix
        cm_fig = self._plot_confusion_matrix(conf_matrix)
        self.writer.add_figure('val/confusion_matrix', cm_fig, epoch)
        
        logger.info(f"Epoch {epoch} [Val] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Update scheduler if it's ReduceLROnPlateau
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(f1)
        
        return avg_loss, accuracy, f1
    
    def _plot_confusion_matrix(self, conf_matrix):
        """
        Plot confusion matrix.
        
        Args:
            conf_matrix: Confusion matrix from sklearn
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        # Set labels
        class_names = ['No Mistake', 'Low Risk', 'High Risk']
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        return fig
    
    def _plot_risk_correlation(self, risk_preds, risk_levels):
        """
        Plot correlation between predicted and actual risk levels.
        
        Args:
            risk_preds: Predicted risk levels
            risk_levels: Actual risk levels
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Convert to numpy arrays
        risk_preds = np.array(risk_preds)
        risk_levels = np.array(risk_levels)
        
        # Plot scatter plot
        ax.scatter(risk_levels, risk_preds, alpha=0.5)
        
        # Plot diagonal line (perfect prediction)
        ax.plot([0, 1], [0, 1], 'r--')
        
        # Set labels
        ax.set_xlabel('Actual Risk Level')
        ax.set_ylabel('Predicted Risk Level')
        ax.set_title('Risk Level Correlation')
        
        # Add regression line
        z = np.polyfit(risk_levels, risk_preds, 1)
        p = np.poly1d(z)
        ax.plot(np.sort(risk_levels), p(np.sort(risk_levels)), "b-")
        
        # Add correlation coefficient
        correlation = np.corrcoef(risk_levels, risk_preds)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes)
        
        return fig
    
    def _save_checkpoint(self, path, epoch, val_f1):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            epoch: Current epoch number
            val_f1: Validation F1 score
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_f1': val_f1,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        
        # Also save as best model if it's the best so far
        if val_f1 >= self.best_val_f1:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Epoch number of the loaded checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_f1 = checkpoint.get('val_f1', 0.0)
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with val F1: {self.best_val_f1:.4f}")
        
        return checkpoint['epoch'] 
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
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (loss, accuracy, f1_score)
        """
        self.model.train()
        
        running_loss = 0.0
        running_loss_cls = 0.0
        running_loss_reg = 0.0
        all_preds = []
        all_labels = []
        
        # Use tqdm for progress bar
        dataloader = self.dataloaders['train']
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} (Train)")
        
        for i, batch in enumerate(pbar):
            # Get data
            frames = batch['frames'].to(self.device)  # [batch_size, seq_len, 3, H, W]
            mistake_type = batch['mistake_type'].to(self.device)  # [batch_size]
            risk_level = batch['risk_level'].to(self.device)  # [batch_size]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with amp.autocast(enabled=self.config['general']['mixed_precision']):
                # Forward pass through model
                logits, risk_pred = self.model(frames)
                
                # Compute loss
                loss_cls = self.classification_criterion(logits, mistake_type)
                loss_reg = self.regression_criterion(risk_pred, risk_level)
                
                # Combined loss
                loss = loss_cls + 0.5 * loss_reg
            
            # Backward pass with mixed precision
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            running_loss += loss.item()
            running_loss_cls += loss_cls.item()
            running_loss_reg += loss_reg.item()
            
            # Calculate accuracy and F1 score
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            labels = mistake_type.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss / (i+1):.4f}"
            })
            
            # Log training loss
            if i % self.config['general']['log_interval'] == 0:
                step = epoch * len(dataloader) + i
                self.writer.add_scalar('Loss/train_step', loss.item(), step)
                self.writer.add_scalar('Loss/cls_step', loss_cls.item(), step)
                self.writer.add_scalar('Loss/reg_step', loss_reg.item(), step)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], step)
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(dataloader)
        avg_loss_cls = running_loss_cls / len(dataloader)
        avg_loss_reg = running_loss_reg / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Log epoch metrics
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('Loss/cls', avg_loss_cls, epoch)
        self.writer.add_scalar('Loss/reg', avg_loss_reg, epoch)
        self.writer.add_scalar('Accuracy/train', accuracy, epoch)
        self.writer.add_scalar('F1/train', f1, epoch)
        
        logger.info(f"Epoch {epoch+1} (Train): Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")
        
        return avg_loss, accuracy, f1
    
    def _validate_epoch(self, epoch):
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (loss, accuracy, f1_score, confusion_matrix)
        """
        self.model.eval()
        
        running_loss = 0.0
        running_loss_cls = 0.0
        running_loss_reg = 0.0
        all_preds = []
        all_labels = []
        all_risk_preds = []
        all_risk_levels = []
        
        # Use tqdm for progress bar
        dataloader = self.dataloaders['val']
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} (Val)")
        
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                # Get data
                frames = batch['frames'].to(self.device)  # [batch_size, seq_len, 3, H, W]
                mistake_type = batch['mistake_type'].to(self.device)  # [batch_size]
                risk_level = batch['risk_level'].to(self.device)  # [batch_size]
                
                # Forward pass with mixed precision
                with amp.autocast(enabled=self.config['general']['mixed_precision']):
                    # Forward pass through model
                    logits, risk_pred = self.model(frames)
                    
                    # Compute loss
                    loss_cls = self.classification_criterion(logits, mistake_type)
                    loss_reg = self.regression_criterion(risk_pred, risk_level)
                    
                    # Combined loss
                    loss = loss_cls + 0.5 * loss_reg
                
                # Update metrics
                running_loss += loss.item()
                running_loss_cls += loss_cls.item()
                running_loss_reg += loss_reg.item()
                
                # Calculate accuracy and F1 score
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                labels = mistake_type.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                # Save risk predictions
                all_risk_preds.extend(risk_pred.cpu().numpy())
                all_risk_levels.extend(risk_level.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{running_loss / (i+1):.4f}"
                })
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(dataloader)
        avg_loss_cls = running_loss_cls / len(dataloader)
        avg_loss_reg = running_loss_reg / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Log epoch metrics
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        self.writer.add_scalar('Loss/cls_val', avg_loss_cls, epoch)
        self.writer.add_scalar('Loss/reg_val', avg_loss_reg, epoch)
        self.writer.add_scalar('Accuracy/val', accuracy, epoch)
        self.writer.add_scalar('F1/val', f1, epoch)
        
        # Plot risk level correlation
        if epoch % 5 == 0:
            risk_fig = self._plot_risk_correlation(all_risk_preds, all_risk_levels)
            self.writer.add_figure('Risk Correlation', risk_fig, epoch)
        
        logger.info(f"Epoch {epoch+1} (Val): Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")
        
        return avg_loss, accuracy, f1, conf_matrix
    
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
"""
Training module for surgical phase recognition.

This module implements the training pipeline for the ViTLSTM phase recognition model.
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch.cuda.amp as amp

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase_recognition import ViTLSTM
from training.surgical_datasets import get_dataloader

logger = logging.getLogger(__name__)

class PhaseRecognitionTrainer:
    """
    Trainer for surgical phase recognition model.
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
        self.checkpoint_dir = Path('training/checkpoints/phase_recognition')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Set up logging directory
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path(f'training/logs/phase_recognition_{current_time}')
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
        
        logger.info("Phase recognition trainer initialized")
    
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
        """Initialize the phase recognition model."""
        model_config = self.config['phase_recognition']['model']
        
        self.model = ViTLSTM(
            num_classes=model_config['num_classes'],
            vit_model=model_config['vit_model'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            use_temporal_attention=model_config['use_temporal_attention'],
            pretrained=model_config['pretrained']
        ).to(self.device)
        
        # Initialize criterion with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config['phase_recognition']['training']['label_smoothing']
        )
        
        logger.info(f"Initialized ViTLSTM model with {model_config['num_classes']} classes")
    
    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        train_config = self.config['phase_recognition']['training']
        
        # Separate parameters for ViT and LSTM to apply different learning rates
        vit_params = []
        lstm_params = []
        
        for name, param in self.model.named_parameters():
            if 'vit' in name:
                vit_params.append(param)
            else:
                lstm_params.append(param)
        
        # Apply lower learning rate to pretrained ViT parameters
        param_groups = [
            {'params': vit_params, 'lr': train_config['learning_rate'] * 0.1},
            {'params': lstm_params, 'lr': train_config['learning_rate']}
        ]
        
        # Initialize optimizer
        if train_config['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(
                param_groups, 
                weight_decay=train_config['weight_decay']
            )
        elif train_config['optimizer'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                param_groups, 
                weight_decay=train_config['weight_decay']
            )
        elif train_config['optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                param_groups, 
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
        data_config = self.config['phase_recognition']['data']
        train_config = self.config['phase_recognition']['training']
        
        # Determine dataset path
        data_dir = Path(f"data/{data_config['dataset']}")
        
        # Create dataloaders
        self.dataloaders = get_dataloader(
            dataset_name=data_config['dataset'],
            data_dir=str(data_dir),
            batch_size=train_config['batch_size'],
            num_workers=self.config['general']['num_workers'],
            sequence_length=train_config['sequence_length'],
            overlap=train_config['sequence_length'] // 2  # 50% overlap
        )
        
        logger.info(f"Initialized dataloaders for {data_config['dataset']} dataset")
    
    def train(self):
        """
        Train the phase recognition model.
        
        Returns:
            Path to the best model checkpoint
        """
        train_config = self.config['phase_recognition']['training']
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
        train_config = self.config['phase_recognition']['training']
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Use tqdm for progress bar
        dataloader = self.dataloaders['train']
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} (Train)")
        
        for i, batch in enumerate(pbar):
            # Get data
            frames = batch['frames'].to(self.device)  # [batch_size, seq_len, 3, H, W]
            phases = batch['phases'].to(self.device)  # [batch_size, seq_len]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with amp.autocast(enabled=self.config['general']['mixed_precision']):
                # Reshape frames to [batch_size * seq_len, 3, H, W]
                batch_size, seq_len = frames.shape[:2]
                
                # Forward pass through model
                logits, _ = self.model(frames)
                
                # Compute loss for each frame in the sequence
                loss = 0
                for t in range(seq_len):
                    loss += self.criterion(logits[:, t], phases[:, t])
                loss = loss / seq_len
            
            # Backward pass with mixed precision
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if train_config['grad_clip'] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_config['grad_clip'])
            
            # Update weights with mixed precision
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            running_loss += loss.item()
            
            # Calculate accuracy and F1 score
            for t in range(seq_len):
                probs = torch.softmax(logits[:, t], dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                labels = phases[:, t].cpu().numpy()
                
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
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], step)
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Log epoch metrics
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
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
        all_preds = []
        all_labels = []
        
        # Use tqdm for progress bar
        dataloader = self.dataloaders['val']
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} (Val)")
        
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                # Get data
                frames = batch['frames'].to(self.device)  # [batch_size, seq_len, 3, H, W]
                phases = batch['phases'].to(self.device)  # [batch_size, seq_len]
                
                # Forward pass with mixed precision
                with amp.autocast(enabled=self.config['general']['mixed_precision']):
                    # Forward pass through model
                    logits, _ = self.model(frames)
                    
                    # Compute loss for each frame in the sequence
                    batch_size, seq_len = frames.shape[:2]
                    loss = 0
                    for t in range(seq_len):
                        loss += self.criterion(logits[:, t], phases[:, t])
                    loss = loss / seq_len
                
                # Update metrics
                running_loss += loss.item()
                
                # Calculate accuracy and F1 score
                for t in range(seq_len):
                    probs = torch.softmax(logits[:, t], dim=1)
                    preds = torch.argmax(probs, dim=1).cpu().numpy()
                    labels = phases[:, t].cpu().numpy()
                    
                    all_preds.extend(preds)
                    all_labels.extend(labels)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{running_loss / (i+1):.4f}"
                })
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
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
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        # Set labels
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
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
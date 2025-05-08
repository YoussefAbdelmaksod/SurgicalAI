"""
Base trainer class for SurgicalAI models.

This module implements a base trainer class that provides common functionality
for training and evaluating different types of models in the SurgicalAI system.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer(ABC):
    """
    Abstract base trainer class for all SurgicalAI models.
    
    Provides common functionality for model training, evaluation, and checkpointing.
    Subclasses should implement train_epoch and validate methods.
    """
    
    def __init__(self, model, optimizer, device, 
                 train_loader=None, val_loader=None, 
                 scheduler=None, checkpoint_dir=None,
                 early_stopping_patience=None,
                 use_mixed_precision=False):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer for training
            device: Device to train on
            train_loader: Training data loader
            val_loader: Validation data loader
            scheduler: Learning rate scheduler
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            use_mixed_precision: Whether to use mixed precision training
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir or 'checkpoints'
        
        # Create checkpoint directory if it doesn't exist
        if self.checkpoint_dir and not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Early stopping setup
        self.early_stopping_patience = early_stopping_patience
        self.best_val_metric = float('inf')  # Default: lower is better
        self.val_metric_higher_is_better = False
        self.epochs_without_improvement = 0
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.lr_history = []
        
        # Epoch tracking
        self.current_epoch = 0
        self.best_epoch = 0
        
        # Mixed precision settings
        self.use_mixed_precision = use_mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        # Checkpointing and resuming
        self.last_checkpoint_path = None
    
    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        pass
    
    def train(self, num_epochs: int, val_frequency: int = 1, 
              primary_val_metric: str = 'loss',
              val_metric_higher_is_better: bool = False,
              save_best_only: bool = True,
              checkpoint_frequency: int = 10,
              resume: bool = False) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            val_frequency: Validate every n epochs
            primary_val_metric: Primary validation metric to track for early stopping
            val_metric_higher_is_better: Whether higher values are better for the primary metric
            save_best_only: Whether to save only the best model
            checkpoint_frequency: Save checkpoint every n epochs
            resume: Whether to resume from a checkpoint
            
        Returns:
            Dictionary of training history
        """
        self.val_metric_higher_is_better = val_metric_higher_is_better
        self.best_val_metric = float('-inf') if val_metric_higher_is_better else float('inf')
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Primary validation metric: {primary_val_metric} ({'higher' if val_metric_higher_is_better else 'lower'} is better)")
        
        # Resume from checkpoint if required
        if resume and self.checkpoint_dir is not None:
            self._resume_from_checkpoint()
        
        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch + 1
            start_time = time.time()
            
            # Train for one epoch
            self.model.train()
            train_metrics = self.train_epoch(self.train_loader)
            
            # Record learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)
            
            # Log training metrics
            train_log = f"Epoch {self.current_epoch}/{num_epochs}"
            for metric_name, metric_value in train_metrics.items():
                train_log += f", {metric_name}: {metric_value:.4f}"
            train_log += f", lr: {current_lr:.6f}"
            logger.info(train_log)
            
            # Store train loss (assuming 'loss' is always in train_metrics)
            if 'loss' in train_metrics:
                self.train_losses.append(train_metrics['loss'])
            
            # Validate if required
            if self.val_loader is not None and (epoch % val_frequency == 0 or epoch == num_epochs - 1):
                self.model.eval()
                val_metrics = self.validate(self.val_loader)
                
                # Log validation metrics
                val_log = f"Validation - Epoch {self.current_epoch}/{num_epochs}"
                for metric_name, metric_value in val_metrics.items():
                    val_log += f", {metric_name}: {metric_value:.4f}"
                logger.info(val_log)
                
                # Store validation loss
                if 'loss' in val_metrics:
                    self.val_losses.append(val_metrics['loss'])
                
                # Track primary metric for model selection
                if primary_val_metric in val_metrics:
                    val_metric = val_metrics[primary_val_metric]
                    self.val_metrics.append(val_metric)
                    
                    # Check if this is the best model so far
                    is_best = False
                    if val_metric_higher_is_better:
                        if val_metric > self.best_val_metric:
                            self.best_val_metric = val_metric
                            self.best_epoch = self.current_epoch
                            self.epochs_without_improvement = 0
                            is_best = True
                        else:
                            self.epochs_without_improvement += 1
                    else:
                        if val_metric < self.best_val_metric:
                            self.best_val_metric = val_metric
                            self.best_epoch = self.current_epoch
                            self.epochs_without_improvement = 0
                            is_best = True
                        else:
                            self.epochs_without_improvement += 1
                    
                    # Save checkpoint
                    if is_best or not save_best_only or (epoch % checkpoint_frequency == 0):
                        self.save_checkpoint(is_best=is_best)
                        
                    # Log best metric so far
                    if is_best:
                        logger.info(f"New best {primary_val_metric}: {val_metric:.4f}")
                
                # Check for early stopping
                if self.early_stopping_patience is not None and self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {self.current_epoch} epochs. "
                                f"No improvement for {self.epochs_without_improvement} epochs.")
                    break
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and self.val_loader is not None:
                    # ReduceLROnPlateau needs a validation metric
                    primary_metric = val_metrics.get(primary_val_metric, 0)
                    self.scheduler.step(primary_metric)
                else:
                    # Other schedulers just need to be stepped
                    self.scheduler.step()
            
            # Log epoch time
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {self.current_epoch} completed in {epoch_time:.2f}s")
        
        # Training completed
        logger.info(f"Training completed. Best {primary_val_metric}: {self.best_val_metric:.4f} at epoch {self.best_epoch}")
        
        # Return training history
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'val_metric': self.val_metrics,
            'lr': self.lr_history,
            'best_epoch': self.best_epoch,
            'best_val_metric': self.best_val_metric
        }
        
        return history
    
    def save_checkpoint(self, is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        if not self.checkpoint_dir:
            logger.warning("Checkpoint directory not specified. Skipping checkpoint.")
            return None
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'val_metric_higher_is_better': self.val_metric_higher_is_better,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'lr_history': self.lr_history
        }
        
        # Add scheduler state if available
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        regular_checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{self.current_epoch}.pth")
        torch.save(checkpoint, regular_checkpoint_path)
        
        # Save best checkpoint if needed
        if is_best:
            best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pth")
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"Saved best model checkpoint to {best_checkpoint_path}")
            return best_checkpoint_path
        
        logger.info(f"Saved checkpoint to {regular_checkpoint_path}")
        return regular_checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True, 
                       load_scheduler: bool = True) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            
        Returns:
            Loaded checkpoint data
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if requested
        if load_optimizer and self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if requested
        if load_scheduler and self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training history
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_metric = checkpoint.get('best_val_metric', float('inf'))
        self.val_metric_higher_is_better = checkpoint.get('val_metric_higher_is_better', False)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        self.lr_history = checkpoint.get('lr_history', [])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.current_epoch})")
        
        return checkpoint
    
    def predict(self, inputs, post_process_fn: Optional[Callable] = None) -> Any:
        """
        Make predictions with the model.
        
        Args:
            inputs: Model inputs
            post_process_fn: Optional function to post-process model outputs
            
        Returns:
            Model predictions
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            
            if post_process_fn is not None:
                outputs = post_process_fn(outputs)
                
            return outputs
    
    def _resume_from_checkpoint(self):
        """
        Resume training from the latest checkpoint.
        """
        if self.checkpoint_dir is None:
            print("No checkpoint directory specified, cannot resume")
            return False
        
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        if not os.path.exists(latest_path):
            print(f"No checkpoint found at {latest_path}, starting from scratch")
            return False
        
        # Load checkpoint
        checkpoint = torch.load(latest_path, map_location=self.device)
        
        # Restore model and optimizer state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler if available
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.best_val_metric = checkpoint['best_val_metric']
        self.val_metric_higher_is_better = checkpoint.get('val_metric_higher_is_better', False)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        self.lr_history = checkpoint.get('lr_history', [])
        
        print(f"Resumed from checkpoint at epoch {checkpoint['epoch']}")
        return True

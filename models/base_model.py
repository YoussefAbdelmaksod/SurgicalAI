"""
Base model class for SurgicalAI models.

This module provides a common base class with shared functionality for all models
used in the SurgicalAI system, including saving/loading checkpoints and metric logging.
"""

import torch
import torch.nn as nn
import os
import logging
import time
from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all SurgicalAI models.
    
    Provides common functionality such as checkpoint management, metric logging,
    and standardized interfaces for training and evaluation.
    """
    
    def __init__(self, name: str = "base_model"):
        """
        Initialize the base model.
        
        Args:
            name: Model name for logging and checkpoints
        """
        super().__init__()
        self.name = name
        self.metrics = {}
        self.best_metric = float('inf')  # Lower is better by default
        self.metric_higher_is_better = False
        self.epoch = 0
        self.train_steps = 0
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass - must be implemented by subclasses.
        """
        pass
    
    def save_checkpoint(self, directory: str, filename: Optional[str] = None, 
                        save_optimizer: bool = True, optimizer: Optional[torch.optim.Optimizer] = None,
                        scheduler: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save model checkpoint.
        
        Args:
            directory: Directory to save checkpoint
            filename: Checkpoint filename (defaults to model_name_epoch_X.pth)
            save_optimizer: Whether to save optimizer state
            optimizer: Optimizer instance to save
            scheduler: Learning rate scheduler to save
            metadata: Additional metadata to save
            
        Returns:
            Path to saved checkpoint
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        # Generate filename if not provided
        if filename is None:
            filename = f"{self.name}_epoch_{self.epoch}.pth"
        
        # Full path to checkpoint
        checkpoint_path = os.path.join(directory, filename)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': self.epoch,
            'train_steps': self.train_steps,
            'model': self.state_dict(),
            'metrics': self.metrics,
            'best_metric': self.best_metric,
            'metric_higher_is_better': self.metric_higher_is_better
        }
        
        # Save optimizer if requested
        if save_optimizer and optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
            
        # Save scheduler if provided
        if scheduler is not None:
            checkpoint['scheduler'] = scheduler.state_dict()
            
        # Add any additional metadata
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return checkpoint_path
    
    def save_best_checkpoint(self, directory: str, metric_name: str, metric_value: float,
                            higher_is_better: bool = False, **kwargs) -> Optional[str]:
        """
        Save checkpoint if the given metric is the best so far.
        
        Args:
            directory: Directory to save checkpoint
            metric_name: Name of the metric to compare
            metric_value: Current metric value
            higher_is_better: Whether higher metric values are better
            **kwargs: Additional arguments for save_checkpoint
            
        Returns:
            Path to saved checkpoint, or None if not saved
        """
        self.metric_higher_is_better = higher_is_better
        
        # Update metrics
        self.metrics[metric_name] = metric_value
        
        # Check if this is the best model so far
        is_best = False
        if higher_is_better:
            if metric_value > self.best_metric:
                self.best_metric = metric_value
                is_best = True
        else:
            if metric_value < self.best_metric:
                self.best_metric = metric_value
                is_best = True
                
        # Save if this is the best model
        if is_best:
            logger.info(f"New best {metric_name}: {metric_value:.4f}")
            filename = f"{self.name}_best_{metric_name}.pth"
            return self.save_checkpoint(directory, filename=filename, **kwargs)
        
        return None
    
    def load_checkpoint(self, checkpoint_path: str, map_location: Optional[str] = None,
                       load_optimizer: bool = False, optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            map_location: Device to map tensors to
            load_optimizer: Whether to load optimizer state
            optimizer: Optimizer instance to load state into
            scheduler: Learning rate scheduler to load state into
            
        Returns:
            Dictionary with loaded checkpoint metadata
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state
        self.load_state_dict(checkpoint['model'])
        
        # Load metadata
        self.epoch = checkpoint.get('epoch', 0)
        self.train_steps = checkpoint.get('train_steps', 0)
        self.metrics = checkpoint.get('metrics', {})
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        self.metric_higher_is_better = checkpoint.get('metric_higher_is_better', False)
        
        # Load optimizer if requested
        if load_optimizer and optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            
        # Load scheduler if requested
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.epoch})")
        
        return checkpoint
    
    def load(self, checkpoint_path: str, map_location: Optional[str] = None) -> None:
        """
        Simple interface to load only the model weights.
        
        Args:
            checkpoint_path: Path to checkpoint
            map_location: Device to map tensors to
        """
        self.load_checkpoint(checkpoint_path, map_location=map_location, load_optimizer=False)
    
    def num_parameters(self, trainable_only: bool = False) -> int:
        """
        Count the number of parameters in the model.
        
        Args:
            trainable_only: Whether to count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def summary(self) -> str:
        """
        Get a string summary of the model.
        
        Returns:
            String with model information
        """
        total_params = self.num_parameters()
        trainable_params = self.num_parameters(trainable_only=True)
        
        summary_str = (
            f"Model: {self.name}\n"
            f"Total parameters: {total_params:,}\n"
            f"Trainable parameters: {trainable_params:,}\n"
            f"Epochs trained: {self.epoch}\n"
            f"Training steps: {self.train_steps}\n"
        )
        
        if self.metrics:
            summary_str += "Metrics:\n"
            for metric_name, metric_value in self.metrics.items():
                summary_str += f"  {metric_name}: {metric_value:.4f}\n"
        
        return summary_str
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = '') -> None:
        """
        Log metrics to the model's metrics dictionary.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step (defaults to current train_steps)
            prefix: Prefix for metric names
        """
        if step is None:
            step = self.train_steps
            
        # Update metrics dictionary
        for name, value in metrics.items():
            metric_name = f"{prefix}{name}" if prefix else name
            self.metrics[metric_name] = value
            
        # Log to console
        metrics_str = ", ".join([f"{name}: {value:.4f}" for name, value in metrics.items()])
        if prefix:
            logger.info(f"[{prefix}] Step {step}: {metrics_str}")
        else:
            logger.info(f"Step {step}: {metrics_str}")
    
    def set_train_mode(self, mode: bool = True) -> None:
        """
        Set model to train or eval mode and perform any additional model-specific setup.
        
        Args:
            mode: True for train mode, False for eval mode
        """
        if mode:
            self.train()
        else:
            self.eval()

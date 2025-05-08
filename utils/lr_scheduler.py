"""
Learning rate scheduling utilities for SurgicalAI.

This module implements various learning rate schedulers and utility functions
for optimizing training convergence.
"""

import math
import torch
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    CosineAnnealingWarmRestarts
)
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Callable

logger = logging.getLogger(__name__)


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with warmup learning rate scheduler.
    
    Initially warms up the learning rate from warmup_start_lr to base_lr over
    warmup_epochs, then applies cosine annealing from base_lr to eta_min over
    the remaining epochs.
    """
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=0.0, 
                eta_min=0.0, last_epoch=-1):
        """
        Initialize WarmupCosineLR scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of epochs for warmup
            max_epochs: Total number of epochs
            warmup_start_lr: Starting learning rate for warmup
            eta_min: Minimum learning rate
            last_epoch: Last epoch to resume from
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        Calculate learning rate based on epoch.
        
        Returns:
            List of learning rates for each parameter group
        """
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr) 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            return [self.eta_min + 0.5 * (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / 
                                  (self.max_epochs - self.warmup_epochs)))
                    for base_lr in self.base_lrs]


def get_scheduler(scheduler_type: str, optimizer, **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler based on type.
    
    Args:
        scheduler_type: Type of scheduler
        optimizer: Optimizer to schedule
        **kwargs: Additional scheduler-specific parameters
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'multistep':
        milestones = kwargs.get('milestones', [30, 60, 90])
        gamma = kwargs.get('gamma', 0.1)
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler_type == 'exponential':
        gamma = kwargs.get('gamma', 0.95)
        return ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == 'cosine':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_type == 'plateau':
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.1)
        patience = kwargs.get('patience', 10)
        threshold = kwargs.get('threshold', 0.0001)
        cooldown = kwargs.get('cooldown', 0)
        min_lr = kwargs.get('min_lr', 0)
        return ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience,
            threshold=threshold, cooldown=cooldown, min_lr=min_lr
        )
    
    elif scheduler_type == 'onecycle':
        max_lr = kwargs.get('max_lr', 0.01)
        total_steps = kwargs.get('total_steps', 100)
        pct_start = kwargs.get('pct_start', 0.3)
        div_factor = kwargs.get('div_factor', 25.0)
        final_div_factor = kwargs.get('final_div_factor', 10000.0)
        return OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps,
            pct_start=pct_start, div_factor=div_factor,
            final_div_factor=final_div_factor
        )
    
    elif scheduler_type == 'warmup_cosine':
        warmup_epochs = kwargs.get('warmup_epochs', 5)
        max_epochs = kwargs.get('max_epochs', 100)
        warmup_start_lr = kwargs.get('warmup_start_lr', 0.0)
        eta_min = kwargs.get('eta_min', 0.0)
        return WarmupCosineLR(
            optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs,
            warmup_start_lr=warmup_start_lr, eta_min=eta_min
        )
    
    elif scheduler_type == 'cyclic_cosine':
        T_0 = kwargs.get('T_0', 10)
        T_mult = kwargs.get('T_mult', 2)
        eta_min = kwargs.get('eta_min', 0)
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def find_optimal_lr(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    min_lr: float = 1e-7,
    max_lr: float = 1.0,
    num_steps: int = 100,
    device: str = 'cuda',
    batch_size: int = None,
    optimizer_type: str = 'Adam',
    weight_decay: float = 0.0,
    beta1: float = 0.9,
    plot: bool = True,
    output_file: str = 'lr_finder.png'
) -> Dict[str, List[float]]:
    """
    Run learning rate finder to determine optimal learning rate.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        loss_fn: Loss function
        min_lr: Minimum learning rate to try
        max_lr: Maximum learning rate to try
        num_steps: Number of steps to try
        device: Device to run on
        batch_size: Batch size (for normalization; defaults to loader's batch size)
        optimizer_type: Type of optimizer to use
        weight_decay: Weight decay for optimizer
        beta1: Beta1 parameter for Adam
        plot: Whether to generate a plot
        output_file: Path to save the plot
        
    Returns:
        Dictionary with learning rates and corresponding losses
    """
    # Move model to device
    model.to(device)
    model.train()
    
    # Create optimizer based on type
    if optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=min_lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=min_lr, betas=(beta1, 0.999), weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr, betas=(beta1, 0.999), weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # Determine batch size for normalization
    if batch_size is None:
        batch_size = next(iter(train_loader))[0].size(0)
    
    # Calculate learning rate multiplier
    lr_mult = (max_lr / min_lr) ** (1 / (num_steps - 1))
    
    # Lists to store learning rates and losses
    learning_rates = []
    losses = []
    
    # Run learning rate finder
    logger.info(f"Running learning rate finder from {min_lr:.8f} to {max_lr:.8f} over {num_steps} steps")
    
    # Use a subset of the data for finding learning rate
    max_batches = min(num_steps, len(train_loader))
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
        
        # Get current learning rate
        lr = min_lr * (lr_mult ** batch_idx)
        
        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Process batch based on its format
        if isinstance(batch, (list, tuple)):
            # Handle different cases based on what the batch contains
            if len(batch) == 2:  # Typical (inputs, targets) case
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
            elif len(batch) == 3:  # Case with sequence data (sequences, lengths, targets)
                sequences, lengths, targets = batch
                sequences = sequences.to(device)
                if lengths is not None:
                    lengths = lengths.to(device)
                targets = targets.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(sequences, lengths)
                loss = loss_fn(outputs, targets)
            
            else:
                # Unsupported batch format
                raise ValueError(f"Unsupported batch format with {len(batch)} elements")
                
        else:
            # Single tensor input
            inputs = batch.to(device)
            
            # Forward pass (assuming autoencoder-like model)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record learning rate and loss
        learning_rates.append(lr)
        losses.append(loss.item())
        
        # Log progress
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Step {batch_idx + 1}/{max_batches}, lr={lr:.8f}, loss={loss.item():.4f}")
    
    # Generate plot if requested
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            # Create figure
            plt.figure(figsize=(10, 6))
            plt.plot(learning_rates, losses)
            plt.xscale('log')
            plt.xlabel('Learning Rate')
            plt.ylabel('Loss')
            plt.title('Learning Rate Finder')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            # Calculate smoothed loss for better visualization
            def smooth(y, box_pts):
                box = np.ones(box_pts) / box_pts
                y_smooth = np.convolve(y, box, mode='same')
                return y_smooth
            
            smoothed_losses = smooth(losses, min(5, len(losses) // 5))
            plt.plot(learning_rates, smoothed_losses, 'r-', linewidth=2)
            
            # Find optimal learning rate (point of steepest descent)
            # We use the smoothed curve for this
            derivatives = np.gradient(smoothed_losses, learning_rates)
            optimal_idx = np.argmin(derivatives)
            optimal_lr = learning_rates[optimal_idx]
            
            plt.axvline(x=optimal_lr, color='g', linestyle='--')
            plt.text(optimal_lr, min(losses), f' Suggested LR: {optimal_lr:.8f}', 
                    verticalalignment='bottom', horizontalalignment='left')
            
            # Save plot
            plt.savefig(output_file)
            logger.info(f"Learning rate finder plot saved to {output_file}")
            
            # Close figure to prevent display in notebook
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not installed. Skipping plot generation.")
    
    # Return results
    result = {
        'learning_rates': learning_rates,
        'losses': losses
    }
    
    # Find suggested learning rate (at minimum loss or at halfway point down the steepest section)
    min_loss_idx = np.argmin(losses)
    suggested_lr = learning_rates[min_loss_idx] / 10  # Conservative estimate
    
    logger.info(f"Suggested learning rate: {suggested_lr:.8f}")
    result['suggested_lr'] = suggested_lr
    
    return result


def visualize_lr_schedules(
    schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler],
    num_epochs: int = 100,
    output_file: str = 'lr_schedules.png'
) -> None:
    """
    Visualize multiple learning rate schedules.
    
    Args:
        schedulers: Dictionary mapping scheduler names to scheduler objects
        num_epochs: Number of epochs to visualize
        output_file: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        for name, scheduler in schedulers.items():
            # Create a copy of the scheduler to avoid modifying the original
            scheduler_copy = type(scheduler)(
                optimizer=scheduler.optimizer, 
                **{k: v for k, v in scheduler.__dict__.items() 
                  if k not in ['optimizer', 'verbose', 'step_count'] and not k.startswith('_')}
            )
            
            # Get base learning rate
            base_lr = scheduler_copy.optimizer.param_groups[0]['lr']
            
            # Track learning rates
            lrs = [base_lr]
            
            # Step through epochs
            for _ in range(num_epochs):
                # Special handling for ReduceLROnPlateau
                if isinstance(scheduler_copy, ReduceLROnPlateau):
                    # Simulate random validation loss decreasing then plateauing
                    val_loss = 1.0 - 0.8 * min(1.0, _ / (0.5 * num_epochs))
                    if _ >= 0.8 * num_epochs:
                        val_loss += 0.1  # Plateau
                    scheduler_copy.step(val_loss)
                else:
                    scheduler_copy.step()
                
                # Record current learning rate
                lrs.append(scheduler_copy.optimizer.param_groups[0]['lr'])
            
            # Plot learning rate schedule
            plt.plot(range(num_epochs + 1), lrs, label=name, linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedules')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.yscale('log')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Learning rate schedules visualization saved to {output_file}")
        
    except ImportError:
        logger.warning("matplotlib not installed. Skipping visualization.")


def create_optimizer(
    optimizer_type: str,
    model_parameters,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
) -> torch.optim.Optimizer:
    """
    Create an optimizer based on type.
    
    Args:
        optimizer_type: Type of optimizer to create
        model_parameters: Model parameters to optimize
        lr: Learning rate
        weight_decay: Weight decay
        momentum: Momentum for SGD
        beta1: Beta1 for Adam/AdamW
        beta2: Beta2 for Adam/AdamW
        eps: Epsilon for Adam/AdamW
        
    Returns:
        Optimizer
    """
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'sgd':
        return torch.optim.SGD(
            model_parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'adam':
        return torch.optim.Adam(
            model_parameters,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model_parameters,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'rmsprop':
        return torch.optim.RMSprop(
            model_parameters,
            lr=lr,
            momentum=momentum,
            alpha=0.99,
            eps=eps,
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'adadelta':
        return torch.optim.Adadelta(
            model_parameters,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay
        )
    
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

"""
Hyperparameter tuning utilities for SurgicalAI.

This module provides utilities for optimizing hyperparameters of models using
various search strategies including grid search, random search, and Bayesian optimization.
"""

import optuna
import numpy as np
import torch
import logging
import os
import yaml
import json
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Type
import time
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid, ParameterSampler

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Hyperparameter optimization for SurgicalAI models.
    
    Supports grid search, random search, and Bayesian optimization via Optuna.
    """
    
    def __init__(
        self,
        objective_fn: Callable,
        param_space: Dict[str, Any],
        method: str = 'bayesian',
        direction: str = 'minimize',
        n_trials: int = 30,
        n_jobs: int = 1,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        save_dir: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            objective_fn: Function to evaluate a parameter configuration. 
                         Should take a dict of parameters and return a score.
            param_space: Dictionary defining parameter search space, including:
                        - Fixed values for grid/random search
                        - (min, max, step) for numerical grid search
                        - (min, max) for continuous Bayesian optimization
                        - List of options for categorical parameters
            method: Optimization method ('grid', 'random', 'bayesian')
            direction: Optimization direction ('minimize' or 'maximize')
            n_trials: Number of trials for random/Bayesian optimization
                     (ignored for grid search)
            n_jobs: Number of parallel jobs
            study_name: Name for the Optuna study
            storage: Optuna storage URL (e.g., 'sqlite:///params.db')
            save_dir: Directory to save results
            seed: Random seed for reproducibility
        """
        self.objective_fn = objective_fn
        self.param_space = param_space
        self.method = method.lower()
        self.direction = direction
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.study_name = study_name or f"surgical_ai_hparam_search_{int(time.time())}"
        self.storage = storage
        self.save_dir = save_dir
        self.seed = seed
        
        # Set up storage directory
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        
        # Set random seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
        # Initialize results placeholder
        self.results = []
        self.best_params = None
        self.best_score = float('inf') if direction == 'minimize' else float('-inf')
        
        # Validate method
        if self.method not in ['grid', 'random', 'bayesian']:
            raise ValueError(f"Unknown optimization method: {self.method}. Use 'grid', 'random', or 'bayesian'.")
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Dictionary containing best parameters and score
        """
        logger.info(f"Starting hyperparameter optimization with {self.method} search")
        start_time = time.time()
        
        if self.method == 'grid':
            self._run_grid_search()
        elif self.method == 'random':
            self._run_random_search()
        elif self.method == 'bayesian':
            self._run_bayesian_search()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Hyperparameter optimization completed in {elapsed_time:.2f} seconds")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {self.best_score}")
        
        # Save results
        if self.save_dir:
            self._save_results()
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'results': self.results,
            'method': self.method,
            'elapsed_time': elapsed_time
        }
    
    def _run_grid_search(self):
        """Run grid search over parameter space."""
        # Convert parameter space to grid format
        grid_params = {}
        for param_name, param_values in self.param_space.items():
            if isinstance(param_values, tuple) and len(param_values) == 3:
                # Handle (min, max, step) format for numerical parameters
                min_val, max_val, step = param_values
                grid_params[param_name] = np.arange(min_val, max_val + step, step).tolist()
            elif isinstance(param_values, (list, tuple)):
                # Directly use list of values
                grid_params[param_name] = param_values
            else:
                # Single value
                grid_params[param_name] = [param_values]
        
        # Create parameter grid
        param_grid = ParameterGrid(grid_params)
        total_configs = len(param_grid)
        logger.info(f"Grid search: evaluating {total_configs} parameter configurations")
        
        # Evaluate each configuration
        for i, params in enumerate(param_grid):
            logger.info(f"Evaluating configuration {i+1}/{total_configs}: {params}")
            score = self.objective_fn(params)
            self.results.append((params, score))
            
            # Update best configuration
            if (self.direction == 'minimize' and score < self.best_score) or \
               (self.direction == 'maximize' and score > self.best_score):
                self.best_score = score
                self.best_params = params
                logger.info(f"New best score: {self.best_score}")
    
    def _run_random_search(self):
        """Run random search over parameter space."""
        # Convert parameter space to sampler format
        sampler_params = {}
        for param_name, param_values in self.param_space.items():
            if isinstance(param_values, tuple) and len(param_values) == 3:
                # Handle (min, max, step) format for numerical parameters
                min_val, max_val, step = param_values
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # Integer parameters
                    sampler_params[param_name] = np.arange(min_val, max_val + step, step).tolist()
                else:
                    # Float parameters - use uniform distribution
                    values = np.arange(min_val, max_val + step, step)
                    sampler_params[param_name] = values.tolist()
            elif isinstance(param_values, (list, tuple)):
                # Directly use list of values
                sampler_params[param_name] = param_values
            else:
                # Single value
                sampler_params[param_name] = [param_values]
        
        # Create parameter sampler
        rng = np.random.RandomState(self.seed) if self.seed is not None else None
        param_sampler = ParameterSampler(sampler_params, n_iter=self.n_trials, random_state=rng)
        
        logger.info(f"Random search: sampling {self.n_trials} configurations")
        
        # Evaluate sampled configurations
        for i, params in enumerate(param_sampler):
            logger.info(f"Evaluating configuration {i+1}/{self.n_trials}: {params}")
            score = self.objective_fn(params)
            self.results.append((params, score))
            
            # Update best configuration
            if (self.direction == 'minimize' and score < self.best_score) or \
               (self.direction == 'maximize' and score > self.best_score):
                self.best_score = score
                self.best_params = params
                logger.info(f"New best score: {self.best_score}")
    
    def _run_bayesian_search(self):
        """Run Bayesian optimization using Optuna."""
        # Create Optuna study
        study_direction = 'minimize' if self.direction == 'minimize' else 'maximize'
        study = optuna.create_study(
            direction=study_direction,
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        
        # Define objective function wrapper for Optuna
        def optuna_objective(trial):
            params = {}
            for param_name, param_values in self.param_space.items():
                if isinstance(param_values, tuple) and len(param_values) == 2:
                    # Continuous parameter range (min, max)
                    min_val, max_val = param_values
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer parameter
                        params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                    else:
                        # Float parameter
                        params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                elif isinstance(param_values, tuple) and len(param_values) == 3:
                    # Step parameter (min, max, step)
                    min_val, max_val, step = param_values
                    if isinstance(min_val, int) and isinstance(max_val, int) and isinstance(step, int):
                        # Integer parameter with step
                        params[param_name] = trial.suggest_int(param_name, min_val, max_val, step=step)
                    else:
                        # Float parameter with step
                        params[param_name] = trial.suggest_float(param_name, min_val, max_val, step=step)
                elif isinstance(param_values, list):
                    # Categorical parameter
                    if all(isinstance(val, (int, float)) for val in param_values):
                        params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                else:
                    # Fixed parameter
                    params[param_name] = param_values
            
            # Evaluate parameters
            score = self.objective_fn(params)
            return score
        
        # Run optimization
        logger.info(f"Bayesian optimization: running {self.n_trials} trials")
        study.optimize(optuna_objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        
        # Get results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Store all trial results
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                self.results.append((trial.params, trial.value))
        
        # Optionally visualize optimization
        if self.save_dir:
            self._save_optuna_plots(study)
    
    def _save_results(self):
        """Save optimization results to disk."""
        # Save best parameters
        best_params_path = os.path.join(self.save_dir, 'best_params.yaml')
        with open(best_params_path, 'w') as f:
            yaml.dump(self.best_params, f)
        
        # Save all results
        results_path = os.path.join(self.save_dir, 'all_results.json')
        results_data = {
            'method': self.method,
            'direction': self.direction,
            'best_score': float(self.best_score),
            'best_params': self.best_params,
            'all_results': [
                {'params': params, 'score': float(score)}
                for params, score in self.results
            ]
        }
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Saved optimization results to {self.save_dir}")
    
    def _save_optuna_plots(self, study):
        """Save Optuna visualization plots."""
        try:
            import optuna.visualization as vis
            
            # Parameter importance
            fig = vis.plot_param_importances(study)
            fig.write_image(os.path.join(self.save_dir, 'param_importances.png'))
            
            # Optimization history
            fig = vis.plot_optimization_history(study)
            fig.write_image(os.path.join(self.save_dir, 'optimization_history.png'))
            
            # Parameter relationships
            fig = vis.plot_contour(study)
            fig.write_image(os.path.join(self.save_dir, 'contour.png'))
            
            # Parallel coordinate
            fig = vis.plot_parallel_coordinate(study)
            fig.write_image(os.path.join(self.save_dir, 'parallel_coordinate.png'))
            
            logger.info(f"Saved Optuna visualization plots to {self.save_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate Optuna plots: {e}")

# Utility functions for common hyperparameter tuning tasks

def tune_learning_rate(
    model_fn: Callable,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    lr_range: Tuple[float, float] = (1e-6, 1e-1),
    num_points: int = 20,
    optimizer_name: str = 'adam',
    batch_size: int = 4,
    epochs_per_point: int = 2,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Find optimal learning rate using learning rate range test.
    
    Args:
        model_fn: Function that returns a model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        lr_range: (min_lr, max_lr) tuple
        num_points: Number of learning rates to test
        optimizer_name: Name of optimizer to use
        batch_size: Batch size
        epochs_per_point: Number of epochs to train for each learning rate
        device: Device to use for training
        
    Returns:
        Dictionary with optimal learning rate and test results
    """
    min_lr, max_lr = lr_range
    lr_values = np.logspace(np.log10(min_lr), np.log10(max_lr), num=num_points)
    
    results = []
    min_loss = float('inf')
    optimal_lr = None
    
    for lr in lr_values:
        # Initialize model and optimizer
        model = model_fn().to(device)
        
        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Train for a few epochs
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs_per_point):
            # Training
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                # Implementation depends on model interface
                # This is a simplified template
                optimizer.zero_grad()
                loss = model.compute_loss(batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            train_loss = epoch_loss / len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    # Similar simplified template
                    loss = model.compute_loss(batch)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)
        
        # Use final validation loss as the score
        final_val_loss = val_losses[-1]
        results.append((lr, final_val_loss))
        
        if final_val_loss < min_loss:
            min_loss = final_val_loss
            optimal_lr = lr
    
    # Return results
    return {
        'optimal_lr': optimal_lr,
        'min_loss': min_loss,
        'results': results
    }

def plot_lr_finder_results(results: List[Tuple[float, float]], save_path: Optional[str] = None):
    """
    Plot learning rate finder results.
    
    Args:
        results: List of (learning_rate, loss) tuples
        save_path: Optional path to save the plot
    """
    lr_values = [lr for lr, _ in results]
    loss_values = [loss for _, loss in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lr_values, loss_values, 'o-')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Loss')
    plt.title('Learning Rate vs. Validation Loss')
    plt.grid(True)
    
    # Find and mark optimal learning rate
    min_loss_idx = np.argmin(loss_values)
    optimal_lr = lr_values[min_loss_idx]
    plt.scatter([optimal_lr], [loss_values[min_loss_idx]], c='r', s=100, zorder=10)
    plt.annotate(f'Optimal LR: {optimal_lr:.1e}',
                xy=(optimal_lr, loss_values[min_loss_idx]),
                xytext=(0, 20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.5'),
                ha='center')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    plt.show()

def suggest_batch_size(
    model_fn: Callable,
    sample_input_size: Tuple[int, ...],
    max_batch_size: int = 64,
    device: str = 'cuda'
) -> int:
    """
    Suggest maximum batch size that fits in memory.
    
    Args:
        model_fn: Function that returns a model instance
        sample_input_size: Size of a single input sample
        max_batch_size: Maximum batch size to try
        device: Device to use for testing
        
    Returns:
        Maximum batch size that fits in memory
    """
    model = model_fn().to(device)
    model.eval()
    
    # Start with batch size of 1 and double until out of memory
    batch_size = 1
    max_feasible_batch = 1
    
    with torch.no_grad():
        while batch_size <= max_batch_size:
            try:
                # Try to create a batch of the current size
                dummy_input = torch.randn((batch_size,) + sample_input_size).to(device)
                
                # Run a forward pass
                _ = model(dummy_input)
                
                # If successful, update max feasible batch and double the size
                max_feasible_batch = batch_size
                batch_size *= 2
                
                # Free memory
                del dummy_input
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    # Memory error - revert to the last successful batch size
                    break
                else:
                    # Some other error
                    raise e
    
    return max_feasible_batch

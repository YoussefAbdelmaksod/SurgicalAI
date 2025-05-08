"""
Training script for surgical phase recognition model.

This script trains a Vision Transformer + LSTM model for recognizing surgical phases
in laparoscopic video sequences.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
import logging

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase_recognition import ViTLSTM, ViTTransformerTemporal
from models.base_model import BaseModel
from training.trainer import Trainer
from utils.helpers import load_config, setup_logging, get_device
from utils.lr_scheduler import get_scheduler
from utils.evaluation import compute_accuracy, compute_f1_score


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Phase Recognition Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for model checkpoints')
    parser.add_argument('--config', type=str, default='config/training_config.yaml', help='Path to config file')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of phase classes')
    parser.add_argument('--vit_model', type=str, default='vit_base_patch16_224', help='ViT model name')
    parser.add_argument('--freeze_vit', action='store_true', help='Freeze ViT weights')
    parser.add_argument('--model_type', type=str, default='vit_lstm', choices=['vit_lstm', 'vit_transformer'], 
                        help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--sequence_length', type=int, default=16, help='Sequence length')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    return parser.parse_args()


def get_transforms(training=True, image_size=224):
    """Get transforms for image preprocessing."""
    if training:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class PhaseRecognitionTrainer(Trainer):
    """
    Trainer class for surgical phase recognition.
    """
    def __init__(self, model, criterion, optimizer, device, 
                train_loader=None, val_loader=None, scheduler=None, 
                checkpoint_dir=None, early_stopping_patience=None):
        """
        Initialize the trainer.
        
        Args:
            model: The ViT-LSTM model
            criterion: Loss function
            optimizer: Model optimizer
            device: Training device
            train_loader: Training data loader
            val_loader: Validation data loader
            scheduler: Learning rate scheduler
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
        """
        super().__init__(model, optimizer, device, train_loader, val_loader, 
                        scheduler, checkpoint_dir, early_stopping_patience)
        self.criterion = criterion
    
    def train_epoch(self, train_loader):
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Prepare progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch_idx, (sequences, sequence_lengths, labels) in enumerate(progress_bar):
            # Move data to device
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            if sequence_lengths is not None:
                sequence_lengths = sequence_lengths.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(sequences, sequence_lengths)
            
            # Calculate loss
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]  # Take the main output if multiple returns
                
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss / (batch_idx + 1):.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        # Log metrics
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
        return metrics
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            # Prepare progress bar
            progress_bar = tqdm(val_loader, desc=f"Epoch {self.current_epoch} [Val]")
            
            for batch_idx, (sequences, sequence_lengths, labels) in enumerate(progress_bar):
                # Move data to device
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                if sequence_lengths is not None:
                    sequence_lengths = sequence_lengths.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences, sequence_lengths)
                
                # Calculate loss
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]  # Take the main output if multiple returns
                    
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Save predictions and labels for F1 score
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{total_loss / (batch_idx + 1):.4f}",
                    'acc': f"{100. * correct / total:.2f}%"
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate F1 score
        f1_score = compute_f1_score(all_labels, all_predictions, average='weighted')
        
        # Log metrics
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1_score
        }
        
        return metrics


def main():
    """Main training function."""
    args = parse_args()
    
    # Set up logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Log arguments
    logger.info(f"Arguments: {args}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data loaders
    train_dataset, val_dataset = get_datasets(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        debug=args.debug
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    if args.model_type == 'vit_lstm':
        model = ViTLSTM(
            num_classes=args.num_classes,
            vit_model=args.vit_model,
            hidden_size=config['model'].get('phase_recognition', {}).get('hidden_size', 512),
            num_layers=config['model'].get('phase_recognition', {}).get('num_layers', 2),
            dropout=config['model'].get('phase_recognition', {}).get('dropout', 0.3),
            pretrained=True,
            bidirectional=True,
            freeze_vit=args.freeze_vit,
            use_temporal_attention=True
        )
        logger.info("Using ViT-LSTM model for phase recognition")
    else:
        model = ViTTransformerTemporal(
            num_classes=args.num_classes,
            vit_model=args.vit_model,
            hidden_dim=config['model'].get('phase_recognition', {}).get('hidden_size', 512),
            nhead=config['model'].get('phase_recognition', {}).get('num_heads', 8),
            num_encoder_layers=config['model'].get('phase_recognition', {}).get('num_layers', 6),
            dropout=config['model'].get('phase_recognition', {}).get('dropout', 0.1),
            pretrained=True,
            freeze_vit=args.freeze_vit,
            max_seq_length=args.sequence_length
        )
        logger.info("Using ViT-Transformer model for phase recognition")
    
    # Move model to device
    model = model.to(device)
    
    # Print model summary
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = PhaseRecognitionTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.output_dir,
        logger=logger
    )
    
    # Train model
    best_model, best_metrics = trainer.train()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Return best model and metrics
    return best_model, best_metrics


if __name__ == '__main__':
    main() 
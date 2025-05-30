"""
Training module for surgical tool detection.

This module implements the training pipeline for the tool detection model using Faster R-CNN.
"""

import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import logging
import yaml
from pathlib import Path
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.cuda.amp as amp
from torchvision.utils import make_grid
import cv2

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tool_detection import AdvancedToolDetectionModel
from training.surgical_datasets import get_dataloader

logger = logging.getLogger(__name__)

class ToolDetectionTrainer:
    """
    Trainer for surgical tool detection model.
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
        self.checkpoint_dir = Path('training/checkpoints/tool_detection')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Set up logging directory
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path(f'training/logs/tool_detection_{current_time}')
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
        # Note: Object detection models may have issues with mixed precision
        self.use_amp = self.config['general']['mixed_precision']
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        
        # Initialize metrics tracking
        self.best_val_map = 0.0
        self.epochs_without_improvement = 0
        
        logger.info("Tool detection trainer initialized")
    
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
        """Initialize the tool detection model."""
        model_config = self.config['tool_detection']['model']
        
        self.model = AdvancedToolDetectionModel(
            num_classes=model_config['num_classes'],
            architecture=model_config['name'],
            backbone_name=model_config['backbone'],
            use_fpn=model_config['use_fpn'],
            pretrained=model_config['pretrained'],
            score_threshold=model_config['score_threshold']
        ).to(self.device)
        
        logger.info(f"Initialized {model_config['name']} model with {model_config['num_classes']} classes")
    
    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        train_config = self.config['tool_detection']['training']
        
        # Separate parameters for backbone and detection head
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Apply lower learning rate to pretrained backbone parameters
        param_groups = [
            {'params': backbone_params, 'lr': train_config['learning_rate'] * 0.1},
            {'params': head_params, 'lr': train_config['learning_rate']}
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
                momentum=train_config['momentum'],
                weight_decay=train_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")
        
        # Initialize scheduler
        if train_config['lr_scheduler'].lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=train_config['lr_step_size'], 
                gamma=train_config['lr_gamma']
            )
        elif train_config['lr_scheduler'].lower() == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=0.1, 
                patience=5
            )
        elif train_config['lr_scheduler'].lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=train_config['epochs'],
                eta_min=train_config['learning_rate'] * 0.01
            )
        else:
            self.scheduler = None
        
        logger.info(f"Initialized {train_config['optimizer']} optimizer with "
                   f"learning rate {train_config['learning_rate']}")
    
    def _init_dataloaders(self):
        """Initialize training and validation dataloaders."""
        data_config = self.config['tool_detection']['data']
        train_config = self.config['tool_detection']['training']
        
        # Determine dataset path
        data_dir = Path(f"data/{data_config['dataset']}")
        
        # Create dataloaders
        self.dataloaders = get_dataloader(
            dataset_name=data_config['dataset'],
            data_dir=str(data_dir),
            batch_size=train_config['batch_size'],
            num_workers=self.config['general']['num_workers'],
            img_size=(512, 512),  # Fixed size for object detection
            min_visibility=data_config['augmentations'].get('min_visibility', 0.5),
            min_size=data_config['augmentations'].get('min_size', 10)
        )
        
        logger.info(f"Initialized dataloaders for {data_config['dataset']} dataset")
    
    def train(self):
        """
        Train the tool detection model.
        
        Returns:
            Path to the best model checkpoint
        """
        train_config = self.config['tool_detection']['training']
        num_epochs = train_config['epochs']
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self._train_epoch(epoch)
            
            # Validation phase
            if (epoch + 1) % self.config['general']['val_interval'] == 0:
                val_loss, val_map = self._validate_epoch(epoch)
                
                # Log validation results
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('mAP/val', val_map, epoch)
                
                # Update learning rate if using ReduceLROnPlateau
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_map)
                
                # Save model if it's the best so far
                if val_map > self.best_val_map:
                    self.best_val_map = val_map
                    checkpoint_path = self.checkpoint_dir / f'best_model_epoch_{epoch}.pth'
                    self._save_checkpoint(checkpoint_path, epoch, val_map)
                    logger.info(f"New best model saved with mAP: {val_map:.4f}")
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
                self._save_checkpoint(checkpoint_path, epoch, val_map if 'val_map' in locals() else 0.0)
        
        # Save final model
        final_checkpoint_path = self.checkpoint_dir / 'final_model.pth'
        self._save_checkpoint(final_checkpoint_path, num_epochs-1, self.best_val_map)
        
        logger.info(f"Training completed. Best validation mAP: {self.best_val_map:.4f}")
        
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
            Average loss
        """
        self.model.train()
        
        running_loss = 0.0
        running_loss_classifier = 0.0
        running_loss_box_reg = 0.0
        running_loss_objectness = 0.0
        running_loss_rpn_box_reg = 0.0
        
        # Use tqdm for progress bar
        dataloader = self.dataloaders['train']
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} (Train)")
        
        for i, batch in enumerate(pbar):
            # Get data
            images = batch['images'].to(self.device)
            
            # Prepare targets
            targets = []
            for j in range(len(batch['boxes'])):
                boxes = batch['boxes'][j].to(self.device)
                labels = batch['labels'][j].to(self.device)
                
                if len(boxes) > 0:
                    target = {
                        'boxes': boxes,
                        'labels': labels,
                    }
                    targets.append(target)
                else:
                    # Skip this example if there are no boxes
                    continue
            
            # Skip batch if all examples have no boxes
            if len(targets) == 0:
                continue
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            with amp.autocast(enabled=self.use_amp):
                loss_dict = self.model.compute_loss(images, targets)
                
                # Sum all losses
                losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass with mixed precision
            self.scaler.scale(losses).backward()
            
            # Optimize
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            running_loss += losses.item()
            running_loss_classifier += loss_dict.get('loss_classifier', 0.0)
            running_loss_box_reg += loss_dict.get('loss_box_reg', 0.0)
            running_loss_objectness += loss_dict.get('loss_objectness', 0.0)
            running_loss_rpn_box_reg += loss_dict.get('loss_rpn_box_reg', 0.0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss / (i+1):.4f}"
            })
            
            # Log training loss
            if i % self.config['general']['log_interval'] == 0:
                step = epoch * len(dataloader) + i
                self.writer.add_scalar('Loss/train_step', losses.item(), step)
                self.writer.add_scalar('Loss/classifier', loss_dict.get('loss_classifier', 0.0), step)
                self.writer.add_scalar('Loss/box_reg', loss_dict.get('loss_box_reg', 0.0), step)
                self.writer.add_scalar('Loss/objectness', loss_dict.get('loss_objectness', 0.0), step)
                self.writer.add_scalar('Loss/rpn_box_reg', loss_dict.get('loss_rpn_box_reg', 0.0), step)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], step)
                
                # Visualize training examples periodically
                if i % (self.config['general']['log_interval'] * 10) == 0:
                    self._visualize_detections(images, targets, epoch, step, prefix='train')
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(dataloader)
        avg_loss_classifier = running_loss_classifier / len(dataloader)
        avg_loss_box_reg = running_loss_box_reg / len(dataloader)
        avg_loss_objectness = running_loss_objectness / len(dataloader)
        avg_loss_rpn_box_reg = running_loss_rpn_box_reg / len(dataloader)
        
        # Log epoch metrics
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('Loss/classifier_epoch', avg_loss_classifier, epoch)
        self.writer.add_scalar('Loss/box_reg_epoch', avg_loss_box_reg, epoch)
        self.writer.add_scalar('Loss/objectness_epoch', avg_loss_objectness, epoch)
        self.writer.add_scalar('Loss/rpn_box_reg_epoch', avg_loss_rpn_box_reg, epoch)
        
        logger.info(f"Epoch {epoch+1} (Train): Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _validate_epoch(self, epoch):
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (avg_loss, mAP)
        """
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Use tqdm for progress bar
        dataloader = self.dataloaders['val']
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} (Val)")
        
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                # Get data
                images = batch['images'].to(self.device)
                
                # Prepare targets
                targets = []
                for j in range(len(batch['boxes'])):
                    boxes = batch['boxes'][j].to(self.device)
                    labels = batch['labels'][j].to(self.device)
                    
                    if len(boxes) > 0:
                        target = {
                            'boxes': boxes,
                            'labels': labels,
                        }
                        targets.append(target)
                        all_targets.append(target)
                    else:
                        # Skip this example if there are no boxes
                        continue
                
                # Skip batch if all examples have no boxes
                if len(targets) == 0:
                    continue
                
                # Forward pass with optional mixed precision
                with amp.autocast(enabled=self.use_amp):
                    loss_dict = self.model.compute_loss(images, targets)
                    
                    # Sum all losses
                    losses = sum(loss for loss in loss_dict.values())
                
                # Get predictions
                predictions = self.model(images)
                all_predictions.extend(predictions)
                
                # Update metrics
                running_loss += losses.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{running_loss / (i+1):.4f}"
                })
                
                # Visualize validation examples periodically
                if i % 50 == 0:
                    self._visualize_detections(images, predictions, epoch, i, prefix='val')
        
        # Calculate metrics
        avg_loss = running_loss / len(dataloader)
        
        # Calculate mAP
        map_value = self._calculate_map(all_predictions, all_targets)
        
        logger.info(f"Epoch {epoch+1} (Val): Loss: {avg_loss:.4f}, mAP: {map_value:.4f}")
        
        return avg_loss, map_value
    
    def _visualize_detections(self, images, targets_or_preds, epoch, step, prefix='train'):
        """
        Visualize detections on images.
        
        Args:
            images: Batch of images
            targets_or_preds: Targets or predictions
            epoch: Current epoch
            step: Current step
            prefix: Prefix for tensorboard log
        """
        # Take first image from batch
        image = images[0].cpu().permute(1, 2, 0).numpy()
        
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image * std + mean) * 255
        image = image.astype(np.uint8)
        
        # Convert to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get boxes and labels
        if isinstance(targets_or_preds[0], dict) and 'boxes' in targets_or_preds[0]:
            # This is a target
            boxes = targets_or_preds[0]['boxes'].cpu().numpy()
            labels = targets_or_preds[0]['labels'].cpu().numpy()
            scores = np.ones(len(labels))  # All 1.0 for targets
        else:
            # This is a prediction
            boxes = targets_or_preds[0]['boxes'].cpu().numpy()
            labels = targets_or_preds[0]['labels'].cpu().numpy()
            scores = targets_or_preds[0]['scores'].cpu().numpy()
        
        # Draw boxes on image
        for box, label, score in zip(boxes, labels, scores):
            # Only draw high-confidence predictions
            if score < 0.5:
                continue
                
            # Get coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"Class {label}: {score:.2f}"
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert back to RGB for tensorboard
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Log to tensorboard
        self.writer.add_image(f'{prefix}/detections', image, global_step=epoch*1000 + step, dataformats='HWC')
    
    def _calculate_map(self, predictions, targets):
        """
        Calculate mean Average Precision.
        
        Args:
            predictions: List of prediction dicts
            targets: List of target dicts
            
        Returns:
            mAP value
        """
        # Simple implementation of mAP calculation
        # For a production system, use a more comprehensive implementation
        iou_thresholds = [0.5]
        
        # For each class, calculate AP
        num_classes = self.config['tool_detection']['model']['num_classes']
        aps = []
        
        for class_id in range(1, num_classes):  # Skip background class (0)
            # Get all predictions and targets for this class
            class_preds = []
            class_targets = []
            
            for pred in predictions:
                mask = pred['labels'] == class_id
                class_preds.append({
                    'boxes': pred['boxes'][mask].cpu().numpy(),
                    'scores': pred['scores'][mask].cpu().numpy()
                })
            
            for target in targets:
                mask = target['labels'] == class_id
                class_targets.append({
                    'boxes': target['boxes'][mask].cpu().numpy()
                })
            
            # Calculate AP for this class
            ap = self._calculate_ap_for_class(class_preds, class_targets, iou_thresholds[0])
            aps.append(ap)
        
        # Calculate mAP
        map_value = np.mean(aps) if aps else 0.0
        
        return map_value
    
    def _calculate_ap_for_class(self, predictions, targets, iou_threshold=0.5):
        """
        Calculate Average Precision for a single class.
        
        Args:
            predictions: List of prediction dicts for this class
            targets: List of target dicts for this class
            iou_threshold: IoU threshold for matching
            
        Returns:
            AP value
        """
        # Flatten predictions and targets
        all_pred_boxes = []
        all_pred_scores = []
        all_target_boxes = []
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            # Add image index to each box
            for box, score in zip(pred['boxes'], pred['scores']):
                all_pred_boxes.append(np.append(box, i))
                all_pred_scores.append(score)
            
            for box in target['boxes']:
                all_target_boxes.append(np.append(box, i))
        
        all_pred_boxes = np.array(all_pred_boxes)
        all_pred_scores = np.array(all_pred_scores)
        all_target_boxes = np.array(all_target_boxes)
        
        # No predictions or targets
        if len(all_pred_boxes) == 0 or len(all_target_boxes) == 0:
            return 0.0
        
        # Sort predictions by score
        sort_idx = np.argsort(-all_pred_scores)
        all_pred_boxes = all_pred_boxes[sort_idx]
        all_pred_scores = all_pred_scores[sort_idx]
        
        # Calculate precision and recall
        tp = np.zeros(len(all_pred_boxes))
        fp = np.zeros(len(all_pred_boxes))
        
        for i, pred_box in enumerate(all_pred_boxes):
            # Get image index
            img_idx = int(pred_box[4])
            
            # Get target boxes for this image
            img_target_boxes = all_target_boxes[all_target_boxes[:, 4] == img_idx]
            
            if len(img_target_boxes) == 0:
                fp[i] = 1
                continue
            
            # Calculate IoU with each target box
            ious = self._calculate_iou(pred_box[:4], img_target_boxes[:, :4])
            
            # Find best matching target box
            max_iou = np.max(ious)
            max_idx = np.argmax(ious)
            
            if max_iou >= iou_threshold:
                # Match found
                tp[i] = 1
                
                # Remove matched target box to prevent double counting
                all_target_boxes = np.delete(all_target_boxes, max_idx, axis=0)
            else:
                # No match found
                fp[i] = 1
        
        # Calculate cumulative precision and recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / len(all_target_boxes) if len(all_target_boxes) > 0 else np.zeros_like(cum_tp)
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
        
        return ap
    
    def _calculate_iou(self, box, boxes):
        """
        Calculate IoU between a box and an array of boxes.
        
        Args:
            box: Single box [x1, y1, x2, y2]
            boxes: Array of boxes [N, 4]
            
        Returns:
            Array of IoU values
        """
        # Calculate intersection area
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = box_area + boxes_area - intersection
        
        # Calculate IoU
        iou = intersection / union
        
        return iou
    
    def _save_checkpoint(self, path, epoch, val_map):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            epoch: Current epoch number
            val_map: Validation mAP
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_map': val_map,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        
        # Also save as best model if it's the best so far
        if val_map >= self.best_val_map:
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
        
        self.best_val_map = checkpoint.get('val_map', 0.0)
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} with val mAP: {self.best_val_map:.4f}")
        
        return checkpoint['epoch'] 
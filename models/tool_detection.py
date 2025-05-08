"""
Surgical tool detection models for SurgicalAI.

This module implements models for detecting surgical tools in laparoscopic
video frames, using Faster R-CNN and other object detection architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import OrderedDict
import os

logger = logging.getLogger(__name__)

# For torchvision detection models
try:
    import torchvision
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn, 
        maskrcnn_resnet50_fpn,
        fasterrcnn_mobilenet_v3_large_fpn,
        retinanet_resnet50_fpn
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    TORCHVISION_AVAILABLE = True
except ImportError:
    logger.warning("torchvision package not installed. Detection models will not be available.")
    TORCHVISION_AVAILABLE = False


class AdvancedToolDetectionModel(nn.Module):
    """
    Advanced surgical tool detection model supporting multiple architectures.
    """
    def __init__(self, num_classes=8, architecture='faster_rcnn', backbone_name='resnet50',
                 pretrained=True, use_fpn=True, min_size=800, max_size=1333,
                 score_threshold=0.5, nms_threshold=0.5, detections_per_img=100):
        """
        Initialize surgical tool detection model.
        
        Args:
            num_classes: Number of tool classes (including background)
            architecture: Model architecture ('faster_rcnn', 'mask_rcnn', 'retinanet')
            backbone_name: Backbone network ('resnet50', 'resnet101', 'mobilenet_v3')
            pretrained: Whether to use pretrained weights
            use_fpn: Whether to use Feature Pyramid Network
            min_size: Minimum size of the image to be rescaled
            max_size: Maximum size of the image to be rescaled
            score_threshold: Threshold for object detection scores
            nms_threshold: NMS IoU threshold
            detections_per_img: Maximum number of detections per image
        """
        super().__init__()
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for tool detection models.")
        
        self.num_classes = num_classes
        self.architecture = architecture
        self.backbone_name = backbone_name
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.detections_per_img = detections_per_img
        
        # Initialize detection model based on architecture and backbone
        self.model = self._initialize_model(
            architecture=architecture,
            backbone_name=backbone_name,
            pretrained=pretrained,
            use_fpn=use_fpn,
            min_size=min_size,
            max_size=max_size
        )
        
        # Replace the pre-trained head with a new one for our number of classes
        self._replace_classifier_head(num_classes)
    
    def _initialize_model(self, architecture, backbone_name, pretrained, use_fpn, min_size, max_size):
        """Initialize the base detection model."""
        if architecture == 'faster_rcnn':
            if backbone_name == 'resnet50':
                model = fasterrcnn_resnet50_fpn(
                    pretrained=pretrained, 
                    pretrained_backbone=pretrained,
                    min_size=min_size,
                    max_size=max_size
                )
            elif backbone_name == 'resnet101':
                # ResNet101 backbone needs manual configuration
                from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
                backbone = resnet_fpn_backbone('resnet101', pretrained=pretrained)
                model = torchvision.models.detection.FasterRCNN(
                    backbone, 
                    num_classes=91,  # COCO classes
                    min_size=min_size,
                    max_size=max_size
                )
            elif backbone_name == 'mobilenet_v3':
                model = fasterrcnn_mobilenet_v3_large_fpn(
                    pretrained=pretrained,
                    pretrained_backbone=pretrained,
                    min_size=min_size,
                    max_size=max_size
                )
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")
                
        elif architecture == 'mask_rcnn':
            if backbone_name == 'resnet50':
                model = maskrcnn_resnet50_fpn(
                    pretrained=pretrained,
                    pretrained_backbone=pretrained,
                    min_size=min_size,
                    max_size=max_size
                )
            elif backbone_name == 'resnet101':
                # ResNet101 backbone needs manual configuration
                from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
                backbone = resnet_fpn_backbone('resnet101', pretrained=pretrained)
                model = torchvision.models.detection.MaskRCNN(
                    backbone, 
                    num_classes=91,  # COCO classes
                    min_size=min_size,
                    max_size=max_size
                )
            else:
                raise ValueError(f"Unsupported backbone for Mask R-CNN: {backbone_name}")
                
        elif architecture == 'retinanet':
            if backbone_name == 'resnet50':
                model = retinanet_resnet50_fpn(
                    pretrained=pretrained,
                    pretrained_backbone=pretrained,
                    min_size=min_size,
                    max_size=max_size
                )
            else:
                raise ValueError(f"Unsupported backbone for RetinaNet: {backbone_name}")
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Configure NMS threshold and detections per image
        if hasattr(model, 'roi_heads') and hasattr(model.roi_heads, 'nms_thresh'):
            model.roi_heads.nms_thresh = self.nms_threshold
            model.roi_heads.detections_per_img = self.detections_per_img
        
        # For RetinaNet
        if hasattr(model, 'head') and hasattr(model.head, 'nms_thresh'):
            model.head.nms_thresh = self.nms_threshold
            model.head.detections_per_img = self.detections_per_img
        
        return model
    
    def _replace_classifier_head(self, num_classes):
        """Replace the classifier head with a new one for our number of classes."""
        if self.architecture == 'faster_rcnn' or self.architecture == 'mask_rcnn':
            # Replace the box predictor
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
            # For Mask R-CNN, also replace the mask predictor
            if self.architecture == 'mask_rcnn':
                in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
                hidden_layer = 256
                self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
                    in_features_mask, hidden_layer, num_classes
                )
                
        elif self.architecture == 'retinanet':
            # RetinaNet requires a different approach
            # This is a simplified version - a complete implementation would need to modify the head
            logger.warning("For RetinaNet, replacing the classifier is more complex. Using default head.")
    
    def forward(self, images, targets=None):
        """
        Forward pass for tool detection.
        
        Args:
            images: Input images (list of tensors or batched tensor)
            targets: Optional targets for training (list of dicts with boxes and labels)
            
        Returns:
            Detected tool boxes, scores, and labels
        """
        # Convert single batched tensor to list of tensors if needed
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            images_list = [img for img in images]
        else:
            images_list = images
            
        # Set model to training/evaluation mode based on targets
        self.model.train(targets is not None)
        
        # Forward pass through the model
        if targets is not None:
            loss_dict = self.model(images_list, targets)
            return loss_dict
        else:
            detections = self.model(images_list)
            return detections
    
    def load(self, checkpoint_path):
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check if the checkpoint contains the entire model or just state dict
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.load_state_dict(checkpoint['model'])
        else:
            self.load_state_dict(checkpoint)
            
        logger.info(f"Loaded weights from {checkpoint_path}")
    
    def save(self, save_path):
        """
        Save model weights to checkpoint.
        
        Args:
            save_path: Path to save the model checkpoint
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the model
        torch.save(self.state_dict(), save_path)
        logger.info(f"Saved model to {save_path}")
    
    def post_process_detections(self, detections, score_threshold=None):
        """
        Post-process detection results.
        
        Args:
            detections: Raw detection output from model
            score_threshold: Override the default score threshold
            
        Returns:
            Processed detections with boxes, scores, and labels above threshold
        """
        threshold = score_threshold if score_threshold is not None else self.score_threshold
        processed_results = []
        
        for detection in detections:
            boxes = detection['boxes']
            scores = detection['scores']
            labels = detection['labels']
            
            # Filter by score threshold
            mask = scores >= threshold
            filtered_boxes = boxes[mask]
            filtered_scores = scores[mask]
            filtered_labels = labels[mask]
            
            # Prepare result dictionary
            result = {
                'boxes': filtered_boxes,
                'scores': filtered_scores,
                'labels': filtered_labels
            }
            
            # Include masks if available (for Mask R-CNN)
            if 'masks' in detection:
                result['masks'] = detection['masks'][mask]
                
            processed_results.append(result)
            
        return processed_results


class ToolDetectionEnsemble(nn.Module):
    """
    Ensemble of tool detection models for improved accuracy.
    """
    def __init__(self, models, ensemble_method='weighted', weights=None, nms_threshold=0.5):
        """
        Initialize tool detection ensemble.
        
        Args:
            models: List of detection models in the ensemble
            ensemble_method: Method to combine detections ('weighted', 'nms')
            weights: Weights for each model (for weighted averaging)
            nms_threshold: NMS IoU threshold for combining detections
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.nms_threshold = nms_threshold
        
        # Validate and normalize weights
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError(f"Number of weights ({len(weights)}) does not match number of models ({len(models)})")
            # Normalize weights to sum to 1
            self.weights = [w / sum(weights) for w in weights]
        else:
            # Equal weights if not specified
            self.weights = [1.0 / len(models) for _ in models]
    
    def forward(self, images, targets=None):
        """
        Forward pass for ensemble detection.
        
        Args:
            images: Input images
            targets: Optional targets for training
            
        Returns:
            Ensemble detection results
        """
        if targets is not None:
            # Training mode - return loss from first model only
            # Note: Training ensemble models is complex, this is simplified
            return self.models[0](images, targets)
        else:
            # Inference mode - combine predictions from all models
            all_detections = []
            
            # Get detections from each model
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    detections = model(images)
                all_detections.append(detections)
            
            # Combine detections using the specified method
            if self.ensemble_method == 'weighted':
                combined_detections = self._weighted_ensemble(all_detections)
            elif self.ensemble_method == 'nms':
                combined_detections = self._nms_ensemble(all_detections)
            else:
                raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
                
            return combined_detections
    
    def _weighted_ensemble(self, all_detections):
        """
        Combine detections using weighted averaging.
        
        This approach combines predictions by weighted averaging of confidence scores
        for boxes that significantly overlap.
        """
        num_images = len(all_detections[0])
        combined_results = []
        
        for img_idx in range(num_images):
            # Collect all boxes, scores, and labels from all models
            all_boxes = []
            all_scores = []
            all_labels = []
            all_model_indices = []
            
            for model_idx, model_detections in enumerate(all_detections):
                img_detection = model_detections[img_idx]
                boxes = img_detection['boxes']
                scores = img_detection['scores']
                labels = img_detection['labels']
                
                if len(boxes) > 0:
                    all_boxes.append(boxes)
                    all_scores.append(scores * self.weights[model_idx])  # Apply model weight
                    all_labels.append(labels)
                    all_model_indices.extend([model_idx] * len(boxes))
            
            if not all_boxes:  # No detections
                combined_results.append({
                    'boxes': torch.empty((0, 4), device=self.models[0].device),
                    'scores': torch.empty(0, device=self.models[0].device),
                    'labels': torch.empty(0, dtype=torch.long, device=self.models[0].device)
                })
                continue
                
            # Concatenate all detections
            all_boxes = torch.cat(all_boxes)
            all_scores = torch.cat(all_scores)
            all_labels = torch.cat(all_labels)
            all_model_indices = torch.tensor(all_model_indices, device=all_boxes.device)
            
            # Group detections by label
            unique_labels = torch.unique(all_labels)
            final_boxes = []
            final_scores = []
            final_labels = []
            
            for label in unique_labels:
                # Get boxes with this label
                label_mask = all_labels == label
                label_boxes = all_boxes[label_mask]
                label_scores = all_scores[label_mask]
                
                if len(label_boxes) > 0:
                    # Apply NMS within each class
                    keep_indices = torchvision.ops.nms(
                        label_boxes, label_scores, self.nms_threshold
                    )
                    
                    final_boxes.append(label_boxes[keep_indices])
                    final_scores.append(label_scores[keep_indices])
                    final_labels.append(torch.full_like(keep_indices, label))
            
            if final_boxes:  # Some boxes survived NMS
                result = {
                    'boxes': torch.cat(final_boxes),
                    'scores': torch.cat(final_scores),
                    'labels': torch.cat(final_labels)
                }
            else:  # No boxes survived NMS
                result = {
                    'boxes': torch.empty((0, 4), device=all_boxes.device),
                    'scores': torch.empty(0, device=all_boxes.device),
                    'labels': torch.empty(0, dtype=torch.long, device=all_boxes.device)
                }
                
            combined_results.append(result)
            
        return combined_results
    
    def _nms_ensemble(self, all_detections):
        """
        Combine detections using Non-Maximum Suppression.
        
        This approach gathers all predictions from all models and applies 
        NMS with model confidence weighting.
        """
        num_images = len(all_detections[0])
        combined_results = []
        
        for img_idx in range(num_images):
            # Collect all boxes, scores, and labels from all models
            all_boxes = []
            all_scores = []
            all_labels = []
            
            for model_idx, model_detections in enumerate(all_detections):
                img_detection = model_detections[img_idx]
                boxes = img_detection['boxes']
                scores = img_detection['scores']
                labels = img_detection['labels']
                
                if len(boxes) > 0:
                    all_boxes.append(boxes)
                    all_scores.append(scores * self.weights[model_idx])  # Apply model weight
                    all_labels.append(labels)
            
            if not all_boxes:  # No detections
                combined_results.append({
                    'boxes': torch.empty((0, 4), device=self.models[0].device),
                    'scores': torch.empty(0, device=self.models[0].device),
                    'labels': torch.empty(0, dtype=torch.long, device=self.models[0].device)
                })
                continue
                
            # Concatenate all detections
            all_boxes = torch.cat(all_boxes)
            all_scores = torch.cat(all_scores)
            all_labels = torch.cat(all_labels)
            
            # Group detections by label
            unique_labels = torch.unique(all_labels)
            final_boxes = []
            final_scores = []
            final_labels = []
            
            for label in unique_labels:
                # Get boxes with this label
                label_mask = all_labels == label
                label_boxes = all_boxes[label_mask]
                label_scores = all_scores[label_mask]
                
                if len(label_boxes) > 0:
                    # Apply NMS within each class
                    keep_indices = torchvision.ops.nms(
                        label_boxes, label_scores, self.nms_threshold
                    )
                    
                    final_boxes.append(label_boxes[keep_indices])
                    final_scores.append(label_scores[keep_indices])
                    final_labels.append(torch.full_like(keep_indices, label))
            
            if final_boxes:  # Some boxes survived NMS
                result = {
                    'boxes': torch.cat(final_boxes),
                    'scores': torch.cat(final_scores),
                    'labels': torch.cat(final_labels)
                }
            else:  # No boxes survived NMS
                result = {
                    'boxes': torch.empty((0, 4), device=all_boxes.device),
                    'scores': torch.empty(0, device=all_boxes.device),
                    'labels': torch.empty(0, dtype=torch.long, device=all_boxes.device)
                }
                
            combined_results.append(result)
            
        return combined_results
    
    def load(self, checkpoint_paths):
        """
        Load weights for all models in the ensemble.
        
        Args:
            checkpoint_paths: List of paths to model checkpoints
        """
        if isinstance(checkpoint_paths, str):
            # Single path - load for first model only
            self.models[0].load(checkpoint_paths)
        elif isinstance(checkpoint_paths, list):
            # List of paths - load for each model
            if len(checkpoint_paths) != len(self.models):
                raise ValueError(f"Number of checkpoints ({len(checkpoint_paths)}) does not match number of models ({len(self.models)})")
            
            for model, path in zip(self.models, checkpoint_paths):
                model.load(path)
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(checkpoint_paths)}")
    
    def save(self, save_paths):
        """
        Save weights for all models in the ensemble.
        
        Args:
            save_paths: List of paths to save model checkpoints
        """
        if isinstance(save_paths, str):
            # Single path - save first model only
            self.models[0].save(save_paths)
        elif isinstance(save_paths, list):
            # List of paths - save each model
            if len(save_paths) != len(self.models):
                raise ValueError(f"Number of save paths ({len(save_paths)}) does not match number of models ({len(self.models)})")
            
            for model, path in zip(self.models, save_paths):
                model.save(path)
        else:
            raise ValueError(f"Unsupported save path format: {type(save_paths)}")

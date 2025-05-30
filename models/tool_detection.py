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
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

logger = logging.getLogger(__name__)

# For torchvision detection models
try:
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
    Advanced tool detection model using Faster R-CNN.
    
    This model uses a Faster R-CNN architecture with various backbone options
    for detecting surgical tools in laparoscopic videos.
    """
    
    def __init__(self, 
                 num_classes=8,  # Background + 7 tool classes
                 architecture='faster_rcnn',
                 backbone_name='resnet50',
                 use_fpn=True,
                 pretrained=True,
                 score_threshold=0.5):
        """
        Initialize tool detection model.
        
        Args:
            num_classes: Number of classes (including background)
            architecture: Architecture type ('faster_rcnn', 'retinanet', etc.)
            backbone_name: Backbone model name ('resnet50', 'resnet101', etc.)
            use_fpn: Whether to use Feature Pyramid Network
            pretrained: Whether to use pretrained backbone
            score_threshold: Score threshold for detection predictions
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.architecture = architecture
        self.backbone_name = backbone_name
        self.score_threshold = score_threshold
        
        # Initialize model based on architecture
        if architecture.lower() == 'faster_rcnn':
            self.model = self._create_faster_rcnn(backbone_name, use_fpn, pretrained, num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def _create_faster_rcnn(self, backbone_name, use_fpn, pretrained, num_classes):
        """
        Create Faster R-CNN model with specified backbone.
        
        Args:
            backbone_name: Name of the backbone model
            use_fpn: Whether to use Feature Pyramid Network
            pretrained: Whether to use pretrained backbone
            num_classes: Number of classes
            
        Returns:
            Initialized Faster R-CNN model
        """
        # Create backbone
        if use_fpn:
            backbone = resnet_fpn_backbone(
                backbone_name=backbone_name,
                pretrained=pretrained
            )
        else:
            # Create ResNet backbone without FPN
            backbone = getattr(torchvision.models, backbone_name)(pretrained=pretrained)
            
            # Remove the last two layers (avgpool and fc)
            backbone = nn.Sequential(*list(backbone.children())[:-2])
            
            # Set output layer and feature dims
            out_channels = {
                'resnet18': 512,
                'resnet34': 512,
                'resnet50': 2048,
                'resnet101': 2048,
                'resnet152': 2048,
            }[backbone_name]
            
            # Create anchor generator
            anchor_generator = AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),),
                aspect_ratios=((0.5, 1.0, 2.0),)
            )
            
            # Create ROI pooler
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0'],
                output_size=7,
                sampling_ratio=2
            )
            
            # Create Faster R-CNN model with custom backbone
            model = FasterRCNN(
                backbone,
                num_classes=num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                min_size=800,
                max_size=1333,
                box_score_thresh=self.score_threshold
            )
            
            return model
        
        # For FPN backbone, create Faster R-CNN with predefined structure
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False,
            pretrained_backbone=pretrained,
            num_classes=num_classes,
            min_size=800,
            max_size=1333,
            box_score_thresh=self.score_threshold
        )
        
        return model
    
    def forward(self, images, targets=None):
        """
        Forward pass.
        
        Args:
            images: Input images tensor
            targets: Optional target boxes and labels
            
        Returns:
            Dict with detection results or loss dict
        """
        return self.model(images, targets)
    
    def compute_loss(self, images, targets):
        """
        Compute loss for training.
        
        Args:
            images: Input images tensor
            targets: Target boxes and labels
            
        Returns:
            Dict with loss components
        """
        return self.model(images, targets)
    
    def predict(self, images):
        """
        Run prediction on images.
        
        Args:
            images: Input images tensor
            
        Returns:
            List of prediction dicts with boxes, labels, and scores
        """
        self.eval()
        with torch.no_grad():
            predictions = self.model(images)
        return predictions
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dictionary.
        
        Args:
            state_dict: State dictionary
            strict: Whether to strictly enforce matching keys
        """
        self.model.load_state_dict(state_dict, strict=strict)


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


def get_tool_detection_model(config):
    """
    Factory function to create a tool detection model from config.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized tool detection model
    """
    return AdvancedToolDetectionModel(
        num_classes=config.get('num_classes', 8),
        architecture=config.get('name', 'faster_rcnn'),
        backbone_name=config.get('backbone', 'resnet50'),
        use_fpn=config.get('use_fpn', True),
        pretrained=config.get('pretrained', True),
        score_threshold=config.get('score_threshold', 0.5)
    )

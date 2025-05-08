"""
Data augmentation utilities for SurgicalAI.

This module provides functions for augmenting images and bounding boxes
for improved training of surgical tool detection models.
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from typing import Dict, Any, List, Tuple, Union, Optional
import logging
import random

logger = logging.getLogger(__name__)

class SurgicalDataAugmentation:
    """Class for surgical data augmentation with various augmentation levels."""
    
    def __init__(self, augmentation_level: str = 'medium', target_size: Tuple[int, int] = (800, 800)):
        """
        Initialize the augmentation pipeline.
        
        Args:
            augmentation_level: Augmentation intensity ('light', 'medium', 'heavy')
            target_size: Target image size (height, width)
        """
        self.augmentation_level = augmentation_level
        self.target_size = target_size
        
        # Create transformation pipeline based on level
        self.transform = self._create_augmentation_pipeline(augmentation_level, target_size)
    
    def _create_augmentation_pipeline(self, level: str, target_size: Tuple[int, int]):
        """
        Create augmentation pipeline based on specified level.
        
        Args:
            level: Augmentation level ('light', 'medium', 'heavy')
            target_size: Target image size (height, width)
            
        Returns:
            Albumentations transform pipeline
        """
        height, width = target_size
        
        # Basic transformations for all levels
        basic_transforms = [
            A.LongestMaxSize(max_size=max(height, width)),
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        if level == 'light':
            # Light augmentations - basic color adjustments
            transform = A.Compose(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.3),
                ] + basic_transforms,
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
            )
        
        elif level == 'medium':
            # Medium augmentations - more color adjustments + mild geometric transforms
            transform = A.Compose(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.2),
                    A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
                ] + basic_transforms,
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
            )
        
        elif level == 'heavy':
            # Heavy augmentations - aggressive color and geometric transforms
            transform = A.Compose(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),
                    A.GaussianBlur(blur_limit=5, p=0.3),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
                    A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.1),
                    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_height=8, min_width=8, p=0.2),
                ] + basic_transforms,
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
            )
        
        else:
            # Default - just resize and normalize without augmentations
            logger.warning(f"Unknown augmentation level: {level}, defaulting to basic transformations")
            transform = A.Compose(
                basic_transforms,
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
            )
        
        return transform
    
    def __call__(self, image, bboxes=None, class_labels=None):
        """
        Apply augmentations to image and bounding boxes.
        
        Args:
            image: Input image (H x W x C)
            bboxes: List of bounding boxes in Pascal VOC format [x_min, y_min, x_max, y_max]
            class_labels: List of class labels for each bounding box
            
        Returns:
            Augmented image and bounding boxes
        """
        if bboxes is None:
            bboxes = []
        if class_labels is None:
            class_labels = []
            
        # Apply transformations
        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        return {
            'image': transformed['image'],
            'bboxes': transformed.get('bboxes', []),
            'class_labels': transformed.get('class_labels', [])
        }

# Create factory function for convenient initialization
def create_transforms(augmentation_level: str = 'medium', target_size: Tuple[int, int] = (800, 800)):
    """
    Create augmentation pipeline for surgical data.
    
    Args:
        augmentation_level: Augmentation intensity ('light', 'medium', 'heavy', 'none')
        target_size: Target image size (height, width)
        
    Returns:
        Augmentation pipeline instance
    """
    if augmentation_level.lower() == 'none':
        # Create minimal transforms - just resize and normalize
        transform = A.Compose(
            [
                A.LongestMaxSize(max_size=max(target_size)),
                A.PadIfNeeded(
                    min_height=target_size[0],
                    min_width=target_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
        )
        return lambda image, bboxes=None, class_labels=None: transform(
            image=image, bboxes=bboxes or [], class_labels=class_labels or []
        )
    else:
        return SurgicalDataAugmentation(augmentation_level, target_size)

# Specialized functions for test-time augmentation
def create_tta_transforms(num_augmentations: int = 5, target_size: Tuple[int, int] = (800, 800)):
    """
    Create test-time augmentation pipeline.
    
    Args:
        num_augmentations: Number of augmented versions to create
        target_size: Target image size (height, width)
        
    Returns:
        List of augmentation transforms
    """
    height, width = target_size
    
    # Basic normalization and conversion to tensor
    basic_transforms = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    # Create different augmentation pipelines
    transforms = []
    
    # Original image (just resized)
    transforms.append(A.Compose(
        [
            A.LongestMaxSize(max_size=max(height, width)),
            A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT),
        ] + basic_transforms
    ))
    
    # Horizontal flip
    transforms.append(A.Compose(
        [
            A.LongestMaxSize(max_size=max(height, width)),
            A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=1.0),
        ] + basic_transforms
    ))
    
    # Add more transforms if needed
    if num_augmentations > 2:
        # Brightness adjustment
        transforms.append(A.Compose(
            [
                A.LongestMaxSize(max_size=max(height, width)),
                A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0),
            ] + basic_transforms
        ))
    
    if num_augmentations > 3:
        # Contrast adjustment
        transforms.append(A.Compose(
            [
                A.LongestMaxSize(max_size=max(height, width)),
                A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT),
                A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=1.0),
            ] + basic_transforms
        ))
    
    if num_augmentations > 4:
        # Slight rotation
        transforms.append(A.Compose(
            [
                A.LongestMaxSize(max_size=max(height, width)),
                A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT),
                A.Rotate(limit=10, p=1.0),
            ] + basic_transforms
        ))
    
    return transforms

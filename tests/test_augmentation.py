"""
Tests for data augmentation utilities.
"""

import unittest
import numpy as np
import sys
import os
import cv2
import torch
from unittest.mock import patch, MagicMock

# Add project root to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.augmentation import SurgicalDataAugmentation, create_transforms, create_tta_transforms


class TestDataAugmentation(unittest.TestCase):
    """Test cases for the SurgicalDataAugmentation class."""
    
    def setUp(self):
        # Create a sample image
        self.test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        # Add some basic content to the image
        cv2.rectangle(self.test_image, (50, 50), (200, 200), (255, 0, 0), -1)
        cv2.circle(self.test_image, (150, 150), 50, (0, 255, 0), -1)
        
        # Create sample bounding boxes in Pascal VOC format [x_min, y_min, x_max, y_max]
        self.test_boxes = [[50, 50, 200, 200], [100, 100, 200, 200]]
        
        # Class labels for the boxes
        self.test_labels = [1, 2]
        
        # Skip tests if albumentations is not available
        try:
            import albumentations as A
            self.albumentations_available = True
        except ImportError:
            self.albumentations_available = False
    
    def test_init_with_different_levels(self):
        """Test initialization with different augmentation levels."""
        if not self.albumentations_available:
            self.skipTest("albumentations not available")
        
        # Test all augmentation levels
        for level in ['light', 'medium', 'heavy']:
            augmentation = SurgicalDataAugmentation(augmentation_level=level)
            self.assertEqual(augmentation.augmentation_level, level)
            self.assertIsNotNone(augmentation.transform)
    
    def test_transform_application(self):
        """Test that transforms are correctly applied to images and boxes."""
        if not self.albumentations_available:
            self.skipTest("albumentations not available")
        
        augmentation = SurgicalDataAugmentation(augmentation_level='light')
        
        # Apply transform to image and boxes
        result = augmentation(
            image=self.test_image,
            bboxes=self.test_boxes,
            class_labels=self.test_labels
        )
        
        # Check the outputs
        self.assertIn('image', result)
        self.assertIn('bboxes', result)
        self.assertIn('class_labels', result)
        
        # Verify the image was transformed to tensor
        self.assertIsInstance(result['image'], torch.Tensor)
        # Verify image dimensions (C, H, W)
        self.assertEqual(result['image'].ndim, 3)
        self.assertEqual(result['image'].shape[0], 3)  # RGB channels
        
        # Verify bounding boxes were preserved
        self.assertEqual(len(result['bboxes']), len(self.test_boxes))
        
        # Verify class labels were preserved
        self.assertEqual(len(result['class_labels']), len(self.test_labels))
    
    def test_create_transforms_factory(self):
        """Test the create_transforms factory function."""
        if not self.albumentations_available:
            self.skipTest("albumentations not available")
        
        # Test the factory function with 'none' level
        transform_fn = create_transforms(augmentation_level='none')
        result = transform_fn(
            image=self.test_image,
            bboxes=self.test_boxes,
            class_labels=self.test_labels
        )
        
        # Verify outputs
        self.assertIn('image', result)
        self.assertIn('bboxes', result)
        self.assertIn('class_labels', result)
        
        # Verify the image was transformed to tensor
        self.assertIsInstance(result['image'], torch.Tensor)
        
        # Test with other augmentation levels to make sure they return SurgicalDataAugmentation instances
        for level in ['light', 'medium', 'heavy']:
            transform = create_transforms(augmentation_level=level)
            self.assertIsInstance(transform, SurgicalDataAugmentation)
    
    def test_tta_transforms(self):
        """Test test-time augmentation transforms."""
        if not self.albumentations_available:
            self.skipTest("albumentations not available")
        
        # Create TTA transforms
        transforms = create_tta_transforms(num_augmentations=3)
        
        # Verify number of transforms
        self.assertEqual(len(transforms), 3)
        
        # Test applying each transform
        for transform in transforms:
            result = transform(image=self.test_image)
            
            # Verify the output is a tensor
            self.assertIsInstance(result['image'], torch.Tensor)
            # Verify dimensions (C, H, W)
            self.assertEqual(result['image'].ndim, 3)
            self.assertEqual(result['image'].shape[0], 3)


if __name__ == '__main__':
    unittest.main() 
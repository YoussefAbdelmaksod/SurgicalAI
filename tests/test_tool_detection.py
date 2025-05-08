"""
Tests for tool detection model.
"""

import unittest
import torch
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.tool_detection import AdvancedToolDetectionModel, ToolDetectionEnsemble


class TestToolDetectionModel(unittest.TestCase):
    """Test cases for the AdvancedToolDetectionModel."""
    
    def setUp(self):
        # Skip tests if torchvision is not available
        try:
            import torchvision
            self.torchvision_available = True
        except ImportError:
            self.torchvision_available = False
    
    def test_model_initialization(self):
        """Test that model initializes with default parameters."""
        if not self.torchvision_available:
            self.skipTest("torchvision not available")
        
        # Test with minimal mocking to prevent actual model loading
        with patch('models.tool_detection.fasterrcnn_resnet50_fpn') as mock_frcnn:
            mock_model = MagicMock()
            mock_model.roi_heads.box_predictor.cls_score.in_features = 1024
            mock_frcnn.return_value = mock_model
            
            model = AdvancedToolDetectionModel(num_classes=8)
            
            # Verify model was created with expected parameters
            self.assertEqual(model.num_classes, 8)
            self.assertEqual(model.architecture, 'faster_rcnn')
            self.assertEqual(model.backbone_name, 'resnet50')
            mock_frcnn.assert_called_once()
    
    def test_ensemble_initialization(self):
        """Test ensemble model initialization."""
        if not self.torchvision_available:
            self.skipTest("torchvision not available")
        
        # Mock individual models
        with patch('models.tool_detection.AdvancedToolDetectionModel') as mock_model_class:
            mock_model1 = MagicMock()
            mock_model2 = MagicMock()
            mock_model_class.side_effect = [mock_model1, mock_model2]
            
            # Create ensemble with two mock models
            ensemble = ToolDetectionEnsemble(
                models=[mock_model1, mock_model2],
                ensemble_method='weighted',
                weights=[0.6, 0.4]
            )
            
            # Verify ensemble properties
            self.assertEqual(len(ensemble.models), 2)
            self.assertEqual(ensemble.ensemble_method, 'weighted')
            self.assertEqual(ensemble.weights, [0.6, 0.4])
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_model_to_device(self):
        """Test that model can be moved to GPU if available."""
        if not self.torchvision_available:
            self.skipTest("torchvision not available")
        
        with patch('models.tool_detection.fasterrcnn_resnet50_fpn') as mock_frcnn:
            mock_model = MagicMock()
            mock_model.roi_heads.box_predictor.cls_score.in_features = 1024
            mock_frcnn.return_value = mock_model
            
            model = AdvancedToolDetectionModel(num_classes=8)
            model.to('cuda')
            
            # Verify model was moved to CUDA
            mock_model.to.assert_called_with('cuda')


class TestToolDetectionInference(unittest.TestCase):
    """Test cases for tool detection inference."""
    
    def setUp(self):
        # Skip tests if torchvision is not available
        try:
            import torchvision
            self.torchvision_available = True
        except ImportError:
            self.torchvision_available = False

        # Set up patch for model forward
        self.patch_model = patch('models.tool_detection.AdvancedToolDetectionModel')
        self.mock_model_class = self.patch_model.start()
        self.mock_model = MagicMock()
        self.mock_model_class.return_value = self.mock_model
    
    def tearDown(self):
        self.patch_model.stop()
    
    def test_inference_with_batched_tensor(self):
        """Test inference with batched tensor input."""
        if not self.torchvision_available:
            self.skipTest("torchvision not available")
        
        # Mock detection results
        mock_detection_result = [{
            'boxes': torch.tensor([[100, 100, 200, 200]]),
            'labels': torch.tensor([1]),
            'scores': torch.tensor([0.95])
        }]
        self.mock_model.return_value = mock_detection_result
        
        # Create test input
        batch_tensor = torch.randn(1, 3, 800, 800)
        
        # Run inference
        model = AdvancedToolDetectionModel(num_classes=8)
        results = model(batch_tensor)
        
        # Verify results
        self.mock_model.assert_called_once()
        self.assertEqual(len(results), 1)
        self.assertIn('boxes', results[0])
        self.assertIn('labels', results[0])
        self.assertIn('scores', results[0])
    
    def test_nms_threshold_configuration(self):
        """Test NMS threshold configuration."""
        if not self.torchvision_available:
            self.skipTest("torchvision not available")
        
        # Test with custom NMS threshold
        custom_nms = 0.3
        
        with patch('models.tool_detection.fasterrcnn_resnet50_fpn') as mock_frcnn:
            mock_model = MagicMock()
            mock_model.roi_heads.box_predictor.cls_score.in_features = 1024
            mock_model.roi_heads.nms_thresh = 0.7  # Default value
            mock_frcnn.return_value = mock_model
            
            model = AdvancedToolDetectionModel(num_classes=8, nms_threshold=custom_nms)
            
            # Verify NMS threshold was set
            self.assertEqual(model.nms_threshold, custom_nms)


if __name__ == '__main__':
    unittest.main() 
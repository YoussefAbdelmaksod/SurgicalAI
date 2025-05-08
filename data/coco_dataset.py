"""
COCO format dataset for tool detection in SurgicalAI.
"""

import os
import cv2
import json
import torch
import numpy as np
import logging
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pycocotools.coco import COCO

from data.dataset import SurgicalDataset

logger = logging.getLogger(__name__)

class COCOSurgicalToolDataset(Dataset):
    """
    Dataset for surgical tool detection using COCO format annotations.
    This dataset is specifically designed to work with the SurgicalAI tool detection
    images and annotations.
    """
    
    def __init__(self, data_dir, transform=None, training=True, skip_invalid=True):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing the dataset images and annotations.
            transform (callable, optional): Optional transform to be applied on images.
            training (bool): Whether this is for training or validation.
            skip_invalid (bool): Whether to skip invalid images/annotations instead of raising errors.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.training = training
        self.skip_invalid = skip_invalid
        
        # Path to COCO annotations file
        if training:
            self.ann_file = os.path.join(data_dir, '_annotations.coco.json')
            # If annotation file doesn't exist in the exact dir, look for it one level up
            if not os.path.exists(self.ann_file):
                parent_dir = os.path.dirname(data_dir)
                self.ann_file = os.path.join(parent_dir, '_annotations.coco.json')
        else:
            self.ann_file = os.path.join(data_dir, '_annotations.coco.json')
        
        # Check if annotation file exists
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"COCO annotation file not found: {self.ann_file}")
        
        # Load COCO dataset
        try:
            self.coco = COCO(self.ann_file)
        except Exception as e:
            logger.error(f"Failed to load COCO annotations: {e}")
            raise ValueError(f"Invalid COCO annotation file: {self.ann_file}") from e
        
        # Get image ids
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        
        # Class mapping for the dataset
        self.class_names = {
            0: 'background',
            1: 'Bipolar',
            2: 'Clipper',
            3: 'Grasper',
            4: 'Hook',
            5: 'Irrigator',
            6: 'Scissors',
            7: 'Specimen Bag'
        }
        
        # Filter out invalid images if skip_invalid is True
        if skip_invalid:
            self._filter_invalid_images()
        
        logger.info(f"Loaded {len(self.img_ids)} images from {self.ann_file}")
    
    def _filter_invalid_images(self):
        """Filter out images that don't exist or have issues."""
        valid_img_ids = []
        
        for img_id in self.img_ids:
            try:
                img_info = self.coco.loadImgs(img_id)[0]
                image_path = os.path.join(self.data_dir, img_info['file_name'])
                
                # Try parent directory if image not found
                if not os.path.exists(image_path):
                    parent_dir = os.path.dirname(self.data_dir)
                    image_path = os.path.join(parent_dir, img_info['file_name'])
                
                # Skip if image doesn't exist
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found, skipping: {img_info['file_name']}")
                    continue
                
                # Check if image can be opened
                try:
                    with Image.open(image_path) as img:
                        img.verify()  # Verify image is not corrupted
                    valid_img_ids.append(img_id)
                except (UnidentifiedImageError, OSError) as e:
                    logger.warning(f"Corrupted image, skipping: {image_path} - {e}")
                    continue
                
            except Exception as e:
                logger.warning(f"Error checking image {img_id}: {e}")
                continue
        
        skipped = len(self.img_ids) - len(valid_img_ids)
        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid images out of {len(self.img_ids)}")
        
        self.img_ids = valid_img_ids
    
    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        """
        Get the image and target for the given index.
        
        Returns a tuple of (image, target) where target is a dictionary containing:
            - boxes: The bounding boxes in [x, y, w, h] format
            - labels: The class labels for each bounding box
            - image_id: The image id
            - area: The area of the bounding boxes
            - iscrowd: Whether the target is a crowd
        """
        try:
            img_id = self.img_ids[idx]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Load image
            img_info = self.coco.loadImgs(img_id)[0]
            image_path = os.path.join(self.data_dir, img_info['file_name'])
            
            # Check if image exists
            if not os.path.exists(image_path):
                parent_dir = os.path.dirname(self.data_dir)
                image_path = os.path.join(parent_dir, img_info['file_name'])
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {img_info['file_name']}")
            
            # Load image with PIL to ensure RGB format
            try:
                image = Image.open(image_path).convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                if self.skip_invalid:
                    # Return a default empty example if we're skipping
                    logger.warning(f"Error loading image {image_path}: {e}, returning empty sample")
                    return self._get_empty_sample()
                else:
                    raise
            
            # Prepare target
            boxes = []
            labels = []
            areas = []
            iscrowd = []
            
            for ann in anns:
                try:
                    # Get bbox in [x, y, w, h] format from COCO
                    bbox = ann['bbox']
                    
                    # Validate bbox
                    if len(bbox) != 4 or not all(isinstance(x, (int, float)) for x in bbox):
                        logger.warning(f"Invalid bbox format: {bbox}, skipping annotation")
                        continue
                    
                    # Make sure width and height are positive
                    x, y, w, h = bbox
                    if w <= 0 or h <= 0:
                        logger.warning(f"Invalid bbox dimensions: {bbox}, skipping annotation")
                        continue
                    
                    boxes.append([x, y, w, h])
                    
                    # Get class label
                    labels.append(ann['category_id'])
                    
                    # Get area
                    areas.append(ann['area'])
                    
                    # Get iscrowd
                    iscrowd.append(ann['iscrowd'])
                except KeyError as e:
                    logger.warning(f"Missing key in annotation: {e}, skipping annotation")
                    continue
            
            # If no valid boxes, return empty sample
            if len(boxes) == 0 and self.skip_invalid:
                logger.warning(f"No valid annotations for image {image_path}, returning empty sample")
                return self._get_empty_sample()
            
            # Convert lists to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
            
            # Create target dictionary
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([img_id]),
                'area': areas,
                'iscrowd': iscrowd
            }
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            else:
                # Convert PIL image to tensor
                image = transforms.ToTensor()(image)
            
            return image, target
            
        except Exception as e:
            logger.error(f"Error getting item {idx}: {e}")
            if self.skip_invalid:
                return self._get_empty_sample()
            else:
                raise
    
    def _get_empty_sample(self):
        """Return an empty sample for when we need to skip an invalid image."""
        # Create a small black image
        image = torch.zeros(3, 224, 224, dtype=torch.float32)
        
        # Create empty target
        target = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros(0, dtype=torch.int64),
            'image_id': torch.tensor([0]),
            'area': torch.zeros(0, dtype=torch.float32),
            'iscrowd': torch.zeros(0, dtype=torch.int64)
        }
        
        return image, target
    
    def get_img_info(self, idx):
        """Get image info for the given index."""
        img_id = self.img_ids[idx]
        return self.coco.loadImgs(img_id)[0]
    
    def convert_to_faster_rcnn_format(self, target):
        """
        Convert target dictionary to format expected by Faster R-CNN.
        This converts from [x, y, w, h] to [x1, y1, x2, y2] format.
        """
        boxes = target['boxes']
        if boxes.shape[0] == 0:
            # No boxes, return empty target
            return target
        
        # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        x, y, w, h = boxes.unbind(1)
        new_boxes = torch.stack([x, y, x + w, y + h], dim=1)
        target['boxes'] = new_boxes
        
        return target 
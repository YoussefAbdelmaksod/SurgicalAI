"""
Base dataset class for SurgicalAI.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from typing import Dict, List, Any, Tuple, Optional
import json
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
from glob import glob
import logging

logger = logging.getLogger(__name__)

class SurgicalDataset(Dataset):
    """
    Base class for all surgical datasets.
    """
    
    def __init__(self, root: str, transforms=None):
        """
        Initialize dataset.
        
        Args:
            root: Root directory
            transforms: Image transformations
        """
        self.root = root
        self.transforms = transforms
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Item dictionary
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def __len__(self) -> int:
        """Get dataset length."""
        raise NotImplementedError("Subclasses must implement this method")

class MistakeDetectionDataset(SurgicalDataset):
    """
    Dataset for surgical mistake detection.
    
    This dataset loads sequences of frames and associated metadata 
    for training the mistake detection model.
    """
    
    def __init__(self, root, sequence_length=16, temporal_stride=2, 
                 transforms=True, use_augmentation=False, image_size=224):
        """
        Initialize the dataset.
        
        Args:
            root: Dataset root directory
            sequence_length: Number of frames in each sequence
            temporal_stride: Stride between frames in sequence
            transforms: Whether to apply transforms to images
            use_augmentation: Whether to use data augmentation
            image_size: Size to resize images to
        """
        super().__init__(root, None)
        
        self.sequence_length = sequence_length
        self.temporal_stride = temporal_stride
        self.image_size = image_size
        
        # Set up transforms
        if transforms:
            if use_augmentation:
                self.transform = self._get_augmentation_transform()
            else:
                self.transform = self._get_transform()
        else:
            self.transform = None
        
        # Load annotations
        annotation_path = os.path.join(root, 'mistake_annotations.json')
        if not os.path.exists(annotation_path):
            logger.warning(f"Mistake annotations not found at {annotation_path}, using parent directory")
            parent_dir = os.path.dirname(root)
            annotation_path = os.path.join(parent_dir, 'mistake_annotations.json')
            
        if os.path.exists(annotation_path):
            try:
                with open(annotation_path, 'r') as f:
                    self.annotations = json.load(f)
                logger.info(f"Loaded mistake annotations from {annotation_path}")
            except Exception as e:
                logger.error(f"Failed to load annotations: {e}")
                self.annotations = self._create_dummy_annotations()
        else:
            logger.warning(f"Mistake annotations not found, creating dummy data")
            self.annotations = self._create_dummy_annotations()
        
        # Get video segments
        self.video_segments = []
        for video_id, video_data in self.annotations['videos'].items():
            for segment in video_data['segments']:
                self.video_segments.append({
                    'video_id': video_id,
                    'segment_id': segment['segment_id'],
                    'start_frame': segment['start_frame'],
                    'end_frame': segment['end_frame'],
                    'mistakes': segment.get('mistakes', []),
                    'risk_score': segment.get('risk_score', 0.0)
                })
        
        logger.info(f"Loaded {len(self.video_segments)} video segments for mistake detection")
        
        # Find video files
        self.video_files = {}
        video_dir = os.path.join(self.root, 'videos')
        if not os.path.exists(video_dir):
            video_dir = os.path.join(os.path.dirname(self.root), 'videos')
            
        if os.path.exists(video_dir):
            for video_file in glob(os.path.join(video_dir, '*.mp4')):
                video_id = os.path.basename(video_file).split('.')[0]
                self.video_files[video_id] = video_file
            
            logger.info(f"Found {len(self.video_files)} video files")
        
        # Extract frames from videos if needed
        self.frame_dir = os.path.join(self.root, 'frames')
        if not os.path.exists(self.frame_dir):
            os.makedirs(self.frame_dir, exist_ok=True)
            
            # Extract frames from videos
            self._extract_frames()
    
    def _extract_frames(self):
        """Extract frames from videos if they don't already exist."""
        logger.info("Extracting frames from videos (this might take a while)...")
        
        for video_id, video_path in self.video_files.items():
            # Create directory for video frames
            video_frame_dir = os.path.join(self.frame_dir, video_id)
            if not os.path.exists(video_frame_dir):
                os.makedirs(video_frame_dir, exist_ok=True)
            
            # Skip if frames already extracted
            if len(glob(os.path.join(video_frame_dir, '*.jpg'))) > 0:
                logger.info(f"Frames already extracted for video {video_id}, skipping")
                continue
            
            try:
                # Open video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error(f"Could not open video: {video_path}")
                    continue
                
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Save frame
                    frame_path = os.path.join(video_frame_dir, f"{frame_idx:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    frame_idx += 1
                
                cap.release()
                logger.info(f"Extracted {frame_idx} frames from video {video_id}")
                
            except Exception as e:
                logger.error(f"Error extracting frames from video {video_id}: {e}")
    
    def _get_transform(self):
        """Get basic transform for evaluation."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_augmentation_transform(self):
        """Get transform with augmentations for training."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _create_dummy_annotations(self):
        """Create dummy annotations for development."""
        dummy_annotations = {
            'videos': {},
            'mistake_types': {
                'wrong_tool': 'Using incorrect tool for the current task',
                'dangerous_movement': 'Movement that risks damaging tissue',
                'poor_visibility': 'Operating with inadequate visibility',
                'incorrect_clip_placement': 'Improper placement of surgical clips',
                'inadequate_exposure': 'Failure to properly expose target area'
            },
            'risk_thresholds': {
                'critical': 0.8,
                'major': 0.5,
                'minor': 0.0
            }
        }
        
        # Find videos in data/videos directory
        video_dir = os.path.join(os.path.dirname(self.root), 'videos')
        
        if os.path.exists(video_dir):
            video_files = glob(os.path.join(video_dir, '*.mp4'))
            
            # Create entries for each video
            for i, video_path in enumerate(video_files):
                video_id = os.path.basename(video_path).split('.')[0]
                
                dummy_annotations['videos'][video_id] = {
                    'video_name': os.path.basename(video_path),
                    'num_frames': 1000,  # Placeholder
                    'segments': []
                }
                
                # Create segments
                for j in range(5):  # 5 segments per video
                    start_frame = j * 200
                    end_frame = start_frame + 100
                    
                    # Randomly determine if this segment has mistakes
                    has_mistake = np.random.random() > 0.7
                    mistakes = []
                    
                    if has_mistake:
                        mistake_type = np.random.choice(list(dummy_annotations['mistake_types'].keys()))
                        mistakes.append({
                            'mistake_type': mistake_type,
                            'severity': np.random.choice(['minor', 'major', 'critical']),
                            'frame': start_frame + np.random.randint(0, 100),
                            'description': dummy_annotations['mistake_types'][mistake_type]
                        })
                    
                    # Random risk score
                    risk_score = np.random.random()
                    
                    dummy_annotations['videos'][video_id]['segments'].append({
                        'segment_id': f"{video_id}_segment_{j}",
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'mistakes': mistakes,
                        'risk_score': risk_score
                    })
        
        return dummy_annotations
    
    def _load_frame(self, video_id, frame_idx):
        """Load a specific frame from a video."""
        # Check if frame exists
        frame_path = os.path.join(self.frame_dir, video_id, f"{frame_idx:06d}.jpg")
        
        if os.path.exists(frame_path):
            try:
                image = Image.open(frame_path).convert('RGB')
                return image
            except Exception as e:
                logger.error(f"Error loading frame {frame_path}: {e}")
                return self._get_dummy_frame()
        else:
            logger.warning(f"Frame not found: {frame_path}")
            return self._get_dummy_frame()
    
    def _get_dummy_frame(self):
        """Get a dummy frame for error cases."""
        # Create a blank image
        return Image.new('RGB', (self.image_size, self.image_size), color='black')
    
    def _get_tool_id(self, video_id, frame_idx):
        """Get tool ID for a frame (placeholder)."""
        # In a real implementation, this would look up tool detections
        # For now, just return a random tool ID
        return torch.randint(0, 7, (1,)).item()
    
    def __len__(self):
        """Get dataset length."""
        return len(self.video_segments)
    
    def __getitem__(self, idx):
        """
        Get a video segment with mistake annotations.
        
        Returns:
            Dictionary containing:
                - visual_features: Tensor of frames [sequence_length, 3, H, W]
                - tool_ids: Tensor of tool IDs [sequence_length]
                - mistake_labels: Tensor of mistake class labels
                - risk_scores: Tensor of risk scores
        """
        segment = self.video_segments[idx]
        video_id = segment['video_id']
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        
        # Determine frames to load
        frames_to_load = []
        for i in range(self.sequence_length):
            # Calculate frame index with temporal stride
            frame_idx = start_frame + i * self.temporal_stride
            
            # Make sure frame is within segment bounds
            frame_idx = min(frame_idx, end_frame)
            
            frames_to_load.append(frame_idx)
        
        # Load frames
        frames = []
        for frame_idx in frames_to_load:
            frame = self._load_frame(video_id, frame_idx)
            
            # Apply transforms
            if self.transform:
                frame = self.transform(frame)
            else:
                # Convert to tensor manually
                frame = transforms.ToTensor()(frame)
            
            frames.append(frame)
        
        # Stack frames
        visual_features = torch.stack(frames)
        
        # Get tool IDs for each frame
        tool_ids = torch.tensor([self._get_tool_id(video_id, frame_idx) for frame_idx in frames_to_load])
        
        # Get mistake label
        # 0: no mistake, 1: minor mistake, 2: major/critical mistake
        if not segment['mistakes']:
            mistake_label = 0  # No mistake
        else:
            # Check severity of mistake
            severity = segment['mistakes'][0]['severity']
            if severity == 'minor':
                mistake_label = 1  # Minor mistake
            else:
                mistake_label = 2  # Major/critical mistake
        
        # Convert risk score to tensor
        risk_score = torch.tensor([segment['risk_score']], dtype=torch.float32)
        
        return {
            'visual_features': visual_features,
            'tool_ids': tool_ids,
            'mistake_labels': torch.tensor([mistake_label], dtype=torch.long),
            'risk_scores': risk_score
        }

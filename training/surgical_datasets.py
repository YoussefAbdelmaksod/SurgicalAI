"""
Surgical dataset loaders for SurgicalAI training.

This module provides dataset classes for loading and preprocessing
surgical datasets including Cholec80, m2cai16-tool-locations, and EndoScapes.
"""

import os
import json
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import logging

# Handle missing albumentations gracefully
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logging.warning("albumentations package not found. Using torchvision transforms as fallback.")

logger = logging.getLogger(__name__)

class PhaseRecognitionDataset(Dataset):
    """
    Dataset for surgical phase recognition using Cholec80 or similar datasets.
    
    Handles loading of video sequences and corresponding phase labels.
    """
    
    def __init__(self, 
                 data_dir, 
                 split='train',
                 sequence_length=10, 
                 overlap=5,
                 transform=None,
                 img_size=(224, 224),
                 fps=1):
        """
        Initialize the phase recognition dataset.
        
        Args:
            data_dir: Path to dataset
            split: 'train', 'val', or 'test'
            sequence_length: Number of frames in each sequence
            overlap: Overlap between consecutive sequences
            transform: Transforms to apply to images
            img_size: Target image size (height, width)
            fps: Frames per second to sample
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.img_size = img_size
        self.fps = fps
        
        # Set up default transforms if none provided
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
        
        # Load video paths and labels
        self.sequences = self._load_sequences()
        logger.info(f"Loaded {len(self.sequences)} sequences for {split} split")
    
    def _load_sequences(self):
        """
        Load video sequences and phase annotations.
        
        Returns:
            List of (video_path, start_idx, labels) tuples
        """
        # This implementation is specific to Cholec80 directory structure
        # Adjust according to your dataset organization
        annotations_dir = self.data_dir / 'phase_annotations'
        videos_dir = self.data_dir / 'videos'
        
        # Get videos for current split
        if not (annotations_dir / f'{self.split}_split.txt').exists():
            logger.warning(f"Split file not found: {self.split}_split.txt. Using all videos.")
            video_ids = [p.stem for p in videos_dir.glob('*.mp4')]
        else:
            with open(annotations_dir / f'{self.split}_split.txt', 'r') as f:
                video_ids = [line.strip() for line in f.readlines()]
        
        sequences = []
        
        for video_id in video_ids:
            video_path = videos_dir / f'{video_id}.mp4'
            anno_path = annotations_dir / f'{video_id}.txt'
            
            if not video_path.exists() or not anno_path.exists():
                logger.warning(f"Video or annotation not found for {video_id}. Skipping.")
                continue
            
            # Load phase annotations
            phases = []
            with open(anno_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        frame_idx, phase_idx = int(parts[0]), int(parts[1])
                        phases.append((frame_idx, phase_idx))
            
            # Create sequences with overlap
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Determine frame sampling based on FPS
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            sample_rate = max(1, int(video_fps / self.fps))
            
            # Create sequences
            for start_idx in range(0, total_frames - self.sequence_length * sample_rate, 
                                  (self.sequence_length - self.overlap) * sample_rate):
                # Get phases for this sequence
                sequence_phases = []
                for i in range(self.sequence_length):
                    frame_idx = start_idx + i * sample_rate
                    # Find closest phase annotation
                    phase_idx = self._get_phase_at_frame(phases, frame_idx)
                    sequence_phases.append(phase_idx)
                
                sequences.append((str(video_path), start_idx, sample_rate, sequence_phases))
        
        return sequences
    
    def _get_phase_at_frame(self, phases, frame_idx):
        """
        Get phase at a specific frame index.
        
        Args:
            phases: List of (frame_idx, phase_idx) tuples
            frame_idx: Frame index to query
            
        Returns:
            Phase index for the given frame
        """
        # Find the latest phase annotation before this frame
        prev_phase = 0
        for f, p in phases:
            if f > frame_idx:
                break
            prev_phase = p
        return prev_phase
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a video sequence and corresponding labels.
        
        Args:
            idx: Index
            
        Returns:
            Dict with 'frames' and 'phases'
        """
        video_path, start_idx, sample_rate, sequence_phases = self.sequences[idx]
        
        # Load video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        for i in range(self.sequence_length):
            frame_idx = start_idx + i * sample_rate
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # Handle end of video (repeat last frame)
                frames.append(frames[-1] if frames else np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8))
            else:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        
        # Apply transforms
        tensor_frames = []
        for frame in frames:
            # Convert to PIL Image for PyTorch transforms
            pil_image = Image.fromarray(frame)
            tensor_frame = self.transform(pil_image)
            tensor_frames.append(tensor_frame)
        
        # Stack along new dimension to create sequence
        frames_tensor = torch.stack(tensor_frames)
        phases_tensor = torch.tensor(sequence_phases, dtype=torch.long)
        
        return {
            'frames': frames_tensor,
            'phases': phases_tensor
        }


class ToolDetectionDataset(Dataset):
    """
    Dataset for surgical tool detection using m2cai16-tool-locations or similar datasets.
    
    Handles loading of images and corresponding tool annotations in COCO format.
    """
    
    def __init__(self, 
                 data_dir, 
                 split='train',
                 transform=None,
                 img_size=(512, 512),
                 min_visibility=0.1,
                 min_size=10):
        """
        Initialize the tool detection dataset.
        
        Args:
            data_dir: Path to dataset
            split: 'train', 'val', or 'test'
            transform: Albumentations transforms to apply
            img_size: Target image size (height, width)
            min_visibility: Minimum visibility for a box to be included
            min_size: Minimum box size to be included
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.min_visibility = min_visibility
        self.min_size = min_size
        
        # Set up default transforms if none provided
        if transform is None:
            if ALBUMENTATIONS_AVAILABLE:
                if split == 'train':
                    self.transform = A.Compose([
                        A.Resize(height=img_size[0], width=img_size[1]),
                        A.HorizontalFlip(p=0.5),
                        A.RandomBrightnessContrast(p=0.2),
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=min_visibility))
                else:
                    self.transform = A.Compose([
                        A.Resize(height=img_size[0], width=img_size[1]),
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=min_visibility))
            else:
                # Fallback to torchvision transforms (will not handle bounding boxes)
                logger.warning("Using torchvision transforms fallback for ToolDetectionDataset - bounding box augmentation is disabled!")
                if split == 'train':
                    self.transform = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                else:
                    self.transform = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        else:
            self.transform = transform
        
        # Load COCO annotations
        self.images, self.annotations = self._load_coco_data()
        logger.info(f"Loaded {len(self.images)} images for {split} split")
    
    def _load_coco_data(self):
        """
        Load COCO format annotations.
        
        Returns:
            Tuple of (images, annotations)
        """
        # Paths for COCO annotation files
        coco_path = self.data_dir / f'{self.split}'
        if not coco_path.exists():
            coco_path = self.data_dir / 'Cholec80.v5-cholec80-10-2.coco' / f'{self.split}'
            
        if not coco_path.exists():
            raise ValueError(f"Cannot find COCO data directory: {coco_path}")
        
        annotation_file = coco_path / '_annotations.coco.json'
        if not annotation_file.exists():
            raise ValueError(f"Cannot find COCO annotations file: {annotation_file}")
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Extract image info
        images = {}
        for img in coco_data['images']:
            img_id = img['id']
            img_path = coco_path / img['file_name']
            if img_path.exists():
                images[img_id] = {
                    'path': str(img_path),
                    'width': img['width'],
                    'height': img['height'],
                    'annotations': []
                }
        
        # Extract annotations for each image
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id in images:
                # Convert COCO format (x, y, width, height) to Pascal VOC format (xmin, ymin, xmax, ymax)
                x, y, w, h = ann['bbox']
                if w >= self.min_size and h >= self.min_size:
                    images[img_id]['annotations'].append({
                        'bbox': [x, y, x + w, y + h],
                        'category_id': ann['category_id'],
                        'area': ann['area'],
                        'iscrowd': ann['iscrowd']
                    })
        
        # Filter out images with no annotations
        filtered_images = {k: v for k, v in images.items() if v['annotations']}
        
        # Create list of image IDs and annotations
        image_list = list(filtered_images.values())
        
        return image_list, coco_data['categories']
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get an image and corresponding annotations.
        
        Args:
            idx: Index
            
        Returns:
            Dict with image and annotations
        """
        img_data = self.images[idx]
        
        # Load image
        img_path = img_data['path']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        boxes = [ann['bbox'] for ann in img_data['annotations']]
        labels = [ann['category_id'] for ann in img_data['annotations']]
        
        # Apply transforms
        if self.transform:
            if ALBUMENTATIONS_AVAILABLE and isinstance(self.transform, A.Compose):
                # Use albumentations
                transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
                image = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['class_labels']
                
                # Convert to tensor
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            else:
                # Use torchvision transforms (no bbox transforms)
                pil_image = Image.fromarray(image)
                image = self.transform(pil_image)
        else:
            # Convert to tensor without transforms
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        if not boxes:
            # Handle case with no boxes after transform
            return {
                'image': image,
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'image_id': idx
            }
        
        # Convert lists to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': idx
        }


class MistakeDetectionDataset(Dataset):
    """
    Dataset for surgical mistake detection using EndoScapes or similar datasets.
    
    Combines visual features, tool detection, and mistake annotations.
    """
    
    def __init__(self, 
                 data_dir, 
                 split='train',
                 transform=None,
                 img_size=(224, 224),
                 sequence_length=5,
                 use_synthetic=False,
                 synthetic_ratio=0.3,
                 supplementary_data_dir=None):
        """
        Initialize the mistake detection dataset.
        
        Args:
            data_dir: Path to dataset
            split: 'train', 'val', or 'test'
            transform: Transforms to apply to images
            img_size: Target image size (height, width)
            sequence_length: Number of frames in each sequence
            use_synthetic: Whether to use synthetic mistake data
            synthetic_ratio: Ratio of synthetic data to use
            supplementary_data_dir: Path to supplementary data
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.use_synthetic = use_synthetic
        self.synthetic_ratio = synthetic_ratio
        self.supplementary_data_dir = supplementary_data_dir
        
        # Set up default transforms if none provided
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
        
        # Load sequences and mistake annotations
        self.sequences = self._load_sequences()
        logger.info(f"Loaded {len(self.sequences)} sequences for {split} split")
        
        # Generate synthetic data if needed
        if use_synthetic and split == 'train':
            self._add_synthetic_data()
    
    def _load_sequences(self):
        """
        Load mistake sequences and annotations.
        
        Returns:
            List of sequences with their annotations
        """
        # This implementation is specific to EndoScapes directory structure
        # Adjust according to your dataset organization
        videos_dir = self.data_dir / 'videos'
        annotations_dir = self.data_dir / 'annotations'
        
        # Get videos for current split
        if not (self.data_dir / f'{self.split}_split.txt').exists():
            logger.warning(f"Split file not found: {self.split}_split.txt. Using all videos.")
            video_ids = [p.stem for p in videos_dir.glob('*.mp4') if p.is_file()]
        else:
            with open(self.data_dir / f'{self.split}_split.txt', 'r') as f:
                video_ids = [line.strip() for line in f.readlines()]
        
        sequences = []
        
        for video_id in video_ids:
            video_path = videos_dir / f'{video_id}.mp4'
            anno_path = annotations_dir / f'{video_id}.json'
            
            if not video_path.exists() or not anno_path.exists():
                logger.warning(f"Video or annotation not found for {video_id}. Skipping.")
                continue
            
            # Load mistake annotations
            with open(anno_path, 'r') as f:
                annotations = json.load(f)
            
            # Create sequences with specified length
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            for start_frame in range(0, total_frames, self.sequence_length):
                if start_frame + self.sequence_length > total_frames:
                    continue
                
                # Get mistake info for this sequence
                mistakes = []
                for i in range(self.sequence_length):
                    frame_idx = start_frame + i
                    mistake_info = self._get_mistake_at_frame(annotations, frame_idx)
                    mistakes.append(mistake_info)
                
                sequences.append({
                    'video_path': str(video_path),
                    'start_frame': start_frame,
                    'length': self.sequence_length,
                    'mistakes': mistakes
                })
        
        # Add supplementary data if available
        if self.supplementary_data_dir is not None and Path(self.supplementary_data_dir).exists():
            supplementary_sequences = self._load_supplementary_data()
            if supplementary_sequences:
                sequences.extend(supplementary_sequences)
                logger.info(f"Added {len(supplementary_sequences)} sequences from supplementary data")
        
        return sequences
    
    def _load_supplementary_data(self):
        """
        Load additional data from supplementary_data_dir
        
        Returns:
            List of additional sequences
        """
        if self.supplementary_data_dir is None:
            return []
        
        supplementary_dir = Path(self.supplementary_data_dir)
        if not supplementary_dir.exists():
            logger.warning(f"Supplementary data directory not found: {supplementary_dir}")
            return []
        
        # Look for image files in the ds/img directory
        img_dir = supplementary_dir / 'ds' / 'img'
        ann_dir = supplementary_dir / 'ds' / 'ann'
        
        if not img_dir.exists() or not ann_dir.exists():
            logger.warning(f"Required directories not found in supplementary data: {img_dir} or {ann_dir}")
            return []
        
        # Get all image files
        image_files = sorted(list(img_dir.glob('*.png')))
        if not image_files:
            return []
        
        # Group into sequences
        sequences = []
        for i in range(0, len(image_files), self.sequence_length):
            if i + self.sequence_length > len(image_files):
                continue
                
            sequence_files = image_files[i:i+self.sequence_length]
            
            # For supplementary data, we'll use "no mistake" as default label
            # This can be improved if annotations are available
            sequences.append({
                'supplementary': True,
                'image_files': [str(f) for f in sequence_files],
                'mistakes': [{'class': 0, 'severity': 0} for _ in range(self.sequence_length)]
            })
            
        # Only use a portion for training to balance with main dataset
        if self.split == 'train':
            split_idx = int(0.8 * len(sequences))
            sequences = sequences[:split_idx]
        elif self.split == 'val':
            split_idx = int(0.8 * len(sequences))
            test_split_idx = int(0.9 * len(sequences))
            sequences = sequences[split_idx:test_split_idx]
        else:  # test
            split_idx = int(0.9 * len(sequences))
            sequences = sequences[split_idx:]
            
        return sequences
    
    def _add_synthetic_data(self):
        """
        Add synthetic mistake data (for data augmentation).
        
        This function creates synthetic mistake examples by modifying existing data.
        """
        if not self.use_synthetic or len(self.sequences) == 0:
            return
        
        num_synthetic = int(len(self.sequences) * self.synthetic_ratio)
        synthetic_sequences = []
        
        # Create synthetic sequences by introducing artificial mistakes
        for i in range(num_synthetic):
            # Select a random sequence to base the synthetic example on
            base_idx = np.random.randint(0, len(self.sequences))
            base_seq = self.sequences[base_idx]
            
            # Create a copy with modified mistake information
            synthetic_seq = base_seq.copy()
            
            # Introduce synthetic mistakes
            # This is a simplified example - you would typically use more
            # sophisticated approaches for generating realistic mistakes
            synthetic_mistakes = []
            for j in range(len(base_seq['mistakes'])):
                if np.random.random() < 0.3:  # 30% chance of mistake
                    mistake_class = np.random.randint(1, 3)  # 1=low risk, 2=high risk
                    severity = np.random.random() * 0.7 + 0.3  # 0.3-1.0 severity
                    synthetic_mistakes.append({
                        'class': mistake_class,
                        'severity': severity
                    })
                else:
                    # No mistake
                    synthetic_mistakes.append({
                        'class': 0,
                        'severity': 0
                    })
            
            synthetic_seq['mistakes'] = synthetic_mistakes
            synthetic_seq['synthetic'] = True
            synthetic_sequences.append(synthetic_seq)
        
        logger.info(f"Added {len(synthetic_sequences)} synthetic sequences")
        self.sequences.extend(synthetic_sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a sequence of frames and mistake labels.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Dict with 'frames' tensor and 'mistakes' tensor
        """
        sequence = self.sequences[idx]
        
        # Load frames 
        if 'supplementary' in sequence and sequence['supplementary']:
            # Load frames from image files for supplementary data
            frames = []
            for img_path in sequence['image_files']:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Could not read image: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
        else:
            # Load frames from video
            video_path = sequence['video_path']
            start_frame = sequence['start_frame']
            length = sequence['length']
            
            cap = cv2.VideoCapture(video_path)
            frames = []
            for i in range(length):
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
                ret, frame = cap.read()
                if not ret:
                    # Handle frame reading error by duplicating last frame or creating empty
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8))
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            cap.release()
        
        # Process frames
        tensor_frames = []
        for frame in frames:
            # Convert to PIL for transforms
            pil_image = Image.fromarray(frame)
            tensor_frame = self.transform(pil_image)
            tensor_frames.append(tensor_frame)
        
        # Stack frames into tensor of shape [sequence_length, channels, height, width]
        tensor_frames = torch.stack(tensor_frames)
        
        # Get mistake labels
        mistake_labels = torch.tensor([m['class'] for m in sequence['mistakes']], dtype=torch.long)
        
        return {
            'frames': tensor_frames,
            'mistakes': mistake_labels
        }

    def _get_mistake_at_frame(self, annotations, frame_idx):
        """
        Get mistake information for a specific frame.
        
        Args:
            annotations: Dict containing mistake annotations
            frame_idx: Frame index to get mistake for
            
        Returns:
            Dict with 'class' and 'severity' of mistake
        """
        # Default: no mistake
        mistake_info = {'class': 0, 'severity': 0.0}
        
        # Check if annotations contain mistake information for this frame
        frame_key = str(frame_idx)
        if 'mistakes' in annotations and frame_key in annotations['mistakes']:
            mistake = annotations['mistakes'][frame_key]
            mistake_info['class'] = int(mistake.get('class', 0))
            mistake_info['severity'] = float(mistake.get('severity', 0.0))
        
        return mistake_info


def get_dataloader(dataset_name, data_dir, batch_size, num_workers=4, **kwargs):
    """
    Create data loaders for different surgical datasets.
    
    Args:
        dataset_name: Name of the dataset ('cholec80', 'm2cai16-tool-locations', 'endoscapes')
        data_dir: Path to the dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        **kwargs: Additional arguments for specific datasets
        
    Returns:
        Dict containing 'train', 'val', and 'test' dataloaders
    """
    split_names = ['train', 'val', 'test']
    loaders = {}
    
    supplementary_data_dir = kwargs.pop('supplementary_data_dir', None)
    
    for split in split_names:
        if dataset_name.lower() == 'cholec80':
            # Phase recognition dataset
            dataset = PhaseRecognitionDataset(
                data_dir=data_dir,
                split=split,
                sequence_length=kwargs.get('sequence_length', 10),
                overlap=kwargs.get('overlap', 5),
                img_size=kwargs.get('img_size', (224, 224)),
                fps=kwargs.get('fps', 1)
            )
            collate_fn = None
            
        elif dataset_name.lower() == 'm2cai16-tool-locations':
            # Tool detection dataset
            dataset = ToolDetectionDataset(
                data_dir=data_dir,
                split=split,
                img_size=kwargs.get('img_size', (512, 512)),
                min_visibility=kwargs.get('min_visibility', 0.1),
                min_size=kwargs.get('min_size', 10)
            )
            collate_fn = tool_detection_collate_fn
            
        elif dataset_name.lower() == 'endoscapes':
            # Mistake detection dataset
            dataset = MistakeDetectionDataset(
                data_dir=data_dir,
                split=split,
                img_size=kwargs.get('img_size', (224, 224)),
                sequence_length=kwargs.get('sequence_length', 5),
                use_synthetic=kwargs.get('use_synthetic_data', False),
                synthetic_ratio=kwargs.get('synthetic_ratio', 0.3),
                supplementary_data_dir=supplementary_data_dir
            )
            collate_fn = None
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    return loaders


def tool_detection_collate_fn(batch):
    """
    Custom collate function for tool detection dataset.
    
    Args:
        batch: List of samples
        
    Returns:
        Dict with batched tensors
    """
    images = []
    boxes = []
    labels = []
    image_ids = []
    
    for sample in batch:
        images.append(sample['image'])
        boxes.append(sample['boxes'])
        labels.append(sample['labels'])
        image_ids.append(sample['image_id'])
    
    # Stack images
    images = torch.stack(images)
    image_ids = torch.tensor(image_ids)
    
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'image_ids': image_ids
    } 
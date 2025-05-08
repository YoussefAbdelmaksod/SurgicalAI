"""
Helper functions for SurgicalAI.

This module provides various utility functions used throughout the system,
including configuration loading, logging setup, and image processing.
"""

import os
import yaml
import logging
import torch
import cv2
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List
import random
import time
import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, logs to console only)
        
    Returns:
        Logger instance
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is provided
    if log_file is not None:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_device() -> torch.device:
    """
    Get the appropriate device (CPU or GPU) for computations.
    
    Returns:
        torch.device: Device to use
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        logging.info(f"Using GPU: {device_name}")
    else:
        device = torch.device('cpu')
        logging.info("GPU not available, using CPU instead")
    
    return device


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logging.info(f"Random seed set to {seed}")


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image while preserving aspect ratio.
    
    Args:
        image: Input image (BGR format)
        target_size: Target size (width, height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create blank canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Paste resized image at center
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def normalize_image(image: np.ndarray, mean: List[float] = [0.485, 0.456, 0.406], 
                   std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Normalize an image for neural network input.
    
    Args:
        image: Input image (BGR format, 0-255)
        mean: Mean values for each channel
        std: Standard deviation for each channel
        
    Returns:
        Normalized image (RGB format, normalized)
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to float and scale to 0-1
    image_norm = image_rgb.astype(np.float32) / 255.0
    
    # Normalize using mean and std
    for i in range(3):
        image_norm[:, :, i] = (image_norm[:, :, i] - mean[i]) / std[i]
    
    return image_norm


def tensor_to_image(tensor: torch.Tensor, denormalize: bool = True, 
                   mean: List[float] = [0.485, 0.456, 0.406], 
                   std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Convert a PyTorch tensor to a numpy image.
    
    Args:
        tensor: Input tensor [C, H, W]
        denormalize: Whether to denormalize the image
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
        
    Returns:
        Image as numpy array (BGR format, 0-255)
    """
    # Convert to numpy and transpose
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    
    # Denormalize if requested
    if denormalize:
        for i in range(3):
            image[:, :, i] = image[:, :, i] * std[i] + mean[i]
    
    # Clip to 0-1 range
    image = np.clip(image, 0, 1)
    
    # Scale to 0-255 and convert to uint8
    image = (image * 255).astype(np.uint8)
    
    # Convert RGB to BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image_bgr


def draw_bounding_boxes(image: np.ndarray, boxes: List[List[float]], 
                       labels: List[int], scores: List[float], 
                       class_names: List[str], threshold: float = 0.5) -> np.ndarray:
    """
    Draw bounding boxes on an image.
    
    Args:
        image: Input image (BGR format)
        boxes: List of bounding boxes [x1, y1, x2, y2]
        labels: List of class labels
        scores: List of confidence scores
        class_names: List of class names
        threshold: Score threshold for visualization
        
    Returns:
        Image with bounding boxes
    """
    # Make a copy of the image
    image_copy = image.copy()
    
    # Define colors for different classes (HSV to ensure good visibility)
    colors = []
    for i in range(len(class_names)):
        hue = int(255 * i / len(class_names))
        hsv = np.array([hue, 255, 255], dtype=np.uint8).reshape(1, 1, 3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    
    # Draw each box
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        
        # Get box coordinates
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get color for this class
        color = colors[label % len(colors)]
        
        # Draw box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        
        # Create label text
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        label_text = f"{class_name}: {score:.2f}"
        
        # Calculate text size
        text_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background
        cv2.rectangle(image_copy, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(image_copy, label_text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image_copy


def overlay_segmentation_masks(image: np.ndarray, masks: np.ndarray, 
                              labels: List[int], alpha: float = 0.5) -> np.ndarray:
    """
    Overlay segmentation masks on an image.
    
    Args:
        image: Input image (BGR format)
        masks: Binary masks [N, H, W]
        labels: Class labels for each mask
        alpha: Transparency factor
        
    Returns:
        Image with overlaid masks
    """
    # Make a copy of the image
    image_copy = image.copy()
    
    # Define colors for different classes (HSV to ensure good visibility)
    colors = []
    for i in range(max(labels) + 1):
        hue = int(255 * i / (max(labels) + 1))
        hsv = np.array([hue, 255, 255], dtype=np.uint8).reshape(1, 1, 3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    
    # Create overlay
    overlay = image_copy.copy()
    
    # Draw each mask
    for mask, label in zip(masks, labels):
        color = colors[label % len(colors)]
        
        # Apply mask
        overlay[mask > 0.5] = color
    
    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, image_copy, 1 - alpha, 0, image_copy)
    
    return image_copy


def create_timestamp() -> str:
    """
    Create a timestamp string.
    
    Returns:
        Timestamp string
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


class Timer:
    """Simple timer class for measuring elapsed time."""
    
    def __init__(self, name: str = "Operation"):
        """
        Initialize timer.
        
        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time = None
        self.lap_time = None
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()
        self.lap_time = self.start_time
    
    def lap(self, lap_name: str = "Lap") -> float:
        """
        Measure time since last lap or start.
        
        Args:
            lap_name: Name of the lap
            
        Returns:
            Elapsed time since last lap
        """
        if self.lap_time is None:
            raise RuntimeError("Timer not started")
        
        current_time = time.time()
        elapsed = current_time - self.lap_time
        self.lap_time = current_time
        
        logging.info(f"{self.name} - {lap_name}: {format_time(elapsed)}")
        
        return elapsed
    
    def stop(self) -> float:
        """
        Stop the timer.
        
        Returns:
            Total elapsed time
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        elapsed = time.time() - self.start_time
        logging.info(f"{self.name} completed in {format_time(elapsed)}")
        
        return elapsed

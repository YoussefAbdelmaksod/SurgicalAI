"""
Data loading utilities for SurgicalAI.
"""

import cv2
import os
import numpy as np
from typing import List, Dict, Any, Optional

def load_video(video_path: str, frame_rate: Optional[int] = None) -> List[np.ndarray]:
    """
    Load video frames from a file.
    
    Args:
        video_path: Path to video file
        frame_rate: Number of frames to extract per second (if None, extract all frames)
        
    Returns:
        List of frames (BGR format)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine frame extraction interval
    if frame_rate is not None:
        # Extract frames at specified rate
        frame_interval = int(original_fps / frame_rate)
        frame_interval = max(1, frame_interval)  # Ensure at least 1
    else:
        # Extract all frames
        frame_interval = 1
    
    # Extract frames
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)
        
        frame_count += 1
    
    # Release video capture
    cap.release()
    
    return frames 
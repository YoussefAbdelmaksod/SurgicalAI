#!/usr/bin/env python
"""
Data preprocessing script for SurgicalAI.

This script prepares data for training all SurgicalAI models, including:
1. Tool Detection
2. Phase Recognition 
3. Mistake Detection
4. GPT Assistant

It handles data splitting, validation, frame extraction, and dataset preparation.
"""

import os
import sys
import argparse
import yaml
import json
import logging
import cv2
import numpy as np
import random
import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.helpers import setup_logging, load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess data for SurgicalAI')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the dataset')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (defaults to data_dir)')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite existing processed data')
    parser.add_argument('--split_ratio', type=float, default=0.2,
                        help='Train/validation split ratio')
    parser.add_argument('--frame_interval', type=int, default=5,
                        help='Interval for extracting frames from videos')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--tasks', nargs='+', default=['all'],
                       choices=['all', 'tool_detection', 'phase_recognition', 
                                'mistake_detection', 'gpt_assistant'],
                       help='Which preprocessing tasks to run')
    return parser.parse_args()


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def validate_coco_annotations(annotation_file: str, logger: logging.Logger) -> Tuple[bool, Dict]:
    """
    Validate COCO annotation file structure.
    
    Args:
        annotation_file: Path to annotation file
        logger: Logger instance
        
    Returns:
        Tuple of (is_valid, data)
    """
    try:
        # Check if file exists
        if not os.path.exists(annotation_file):
            logger.error(f"Annotation file not found: {annotation_file}")
            return False, {}
        
        # Load JSON
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Check required keys
        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            if key not in data:
                logger.error(f"Missing required key in annotations: {key}")
                return False, {}
        
        # Check content
        if not data['images']:
            logger.error("No images found in annotations")
            return False, {}
        
        if not data['annotations']:
            logger.error("No annotations found in annotations")
            return False, {}
        
        if not data['categories']:
            logger.error("No categories found in annotations")
            return False, {}
        
        # Check first image entry
        first_image = data['images'][0]
        required_image_keys = ['id', 'file_name', 'width', 'height']
        for key in required_image_keys:
            if key not in first_image:
                logger.error(f"Missing required key in image annotation: {key}")
                return False, {}
        
        # Check first annotation entry
        first_annotation = data['annotations'][0]
        required_annotation_keys = ['id', 'image_id', 'category_id', 'bbox']
        for key in required_annotation_keys:
            if key not in first_annotation:
                logger.error(f"Missing required key in annotation entry: {key}")
                return False, {}
        
        # Validate bbox format (should be [x, y, width, height])
        if len(first_annotation['bbox']) != 4:
            logger.error(f"Invalid bbox format: {first_annotation['bbox']}")
            return False, {}
        
        logger.info(f"Annotation file validated: {len(data['images'])} images, "
                   f"{len(data['annotations'])} annotations, "
                   f"{len(data['categories'])} categories")
        return True, data
    
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in annotation file: {annotation_file}")
        return False, {}
    
    except Exception as e:
        logger.error(f"Error validating annotation file: {e}")
        return False, {}


def split_coco_dataset(data: Dict, split_ratio: float, random_seed: int = 42) -> Tuple[Dict, Dict]:
    """
    Split COCO dataset into train and validation sets.
    
    Args:
        data: COCO data dictionary
        split_ratio: Validation split ratio
        random_seed: Random seed
        
    Returns:
        Tuple of (train_data, val_data)
    """
    # Set random seed
    random.seed(random_seed)
    
    # Get image IDs
    image_ids = [img['id'] for img in data['images']]
    random.shuffle(image_ids)
    
    # Split image IDs
    val_size = int(len(image_ids) * split_ratio)
    val_image_ids = set(image_ids[:val_size])
    train_image_ids = set(image_ids[val_size:])
    
    # Create train data
    train_data = {
        'images': [img for img in data['images'] if img['id'] in train_image_ids],
        'annotations': [ann for ann in data['annotations'] if ann['image_id'] in train_image_ids],
        'categories': data['categories']
    }
    
    # Create val data
    val_data = {
        'images': [img for img in data['images'] if img['id'] in val_image_ids],
        'annotations': [ann for ann in data['annotations'] if ann['image_id'] in val_image_ids],
        'categories': data['categories']
    }
    
    return train_data, val_data


def save_coco_dataset(data: Dict, output_file: str):
    """
    Save COCO dataset to file.
    
    Args:
        data: COCO data dictionary
        output_file: Output file path
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f)


def extract_frames_from_video(video_path: str, output_dir: str, interval: int = 5, 
                              start_time: float = None, end_time: float = None) -> List[str]:
    """
    Extract frames from video.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
        interval: Frame interval
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        List of extracted frame paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # Set start and end frames
    start_frame = 0
    if start_time is not None:
        start_frame = int(start_time * fps)
    
    end_frame = frame_count
    if end_time is not None:
        end_frame = min(int(end_time * fps), frame_count)
    
    # Calculate frames to extract
    frames_to_extract = list(range(start_frame, end_frame, interval))
    
    # Extract frames
    extracted_frames = []
    for i, frame_idx in enumerate(tqdm(frames_to_extract, desc=f"Extracting frames from {os.path.basename(video_path)}")):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save frame
        timestamp = frame_idx / fps
        frame_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_{frame_idx:06d}_{timestamp:.2f}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        extracted_frames.append(frame_path)
    
    # Release video
    cap.release()
    
    return extracted_frames


def prepare_tool_detection_data(data_dir: str, output_dir: str, config: Dict, logger: logging.Logger, 
                              force: bool = False, split_ratio: float = 0.2, random_seed: int = 42):
    """
    Prepare data for tool detection model training.
    
    Args:
        data_dir: Data directory
        output_dir: Output directory
        config: Configuration dictionary
        logger: Logger instance
        force: Force overwrite existing data
        split_ratio: Validation split ratio
        random_seed: Random seed
    """
    logger.info("Preparing data for tool detection model training")
    
    # Define directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'valid')
    
    # Check if directories exist
    if os.path.exists(train_dir) and os.path.exists(val_dir) and not force:
        logger.info("Tool detection data already prepared, skipping")
        return
    
    # Create directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Look for annotation file
    annotation_file = os.path.join(data_dir, '_annotations.coco.json')
    
    if not os.path.exists(annotation_file):
        # Try to find annotations in train directory
        train_annotation_file = os.path.join(data_dir, 'train', '_annotations.coco.json')
        val_annotation_file = os.path.join(data_dir, 'valid', '_annotations.coco.json')
        
        if os.path.exists(train_annotation_file) and os.path.exists(val_annotation_file):
            logger.info("Found existing train/val split, copying directly")
            
            # Validate annotations
            train_valid, train_data = validate_coco_annotations(train_annotation_file, logger)
            val_valid, val_data = validate_coco_annotations(val_annotation_file, logger)
            
            if not train_valid or not val_valid:
                logger.error("Invalid annotations found, aborting")
                return
            
            # Copy annotations
            shutil.copy(train_annotation_file, os.path.join(train_dir, '_annotations.coco.json'))
            shutil.copy(val_annotation_file, os.path.join(val_dir, '_annotations.coco.json'))
            
            # Copy images
            train_image_dir = os.path.join(data_dir, 'train')
            val_image_dir = os.path.join(data_dir, 'valid')
            
            for img in train_data['images']:
                src_path = os.path.join(train_image_dir, img['file_name'])
                dst_path = os.path.join(train_dir, img['file_name'])
                
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
            
            for img in val_data['images']:
                src_path = os.path.join(val_image_dir, img['file_name'])
                dst_path = os.path.join(val_dir, img['file_name'])
                
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
            
            return
        else:
            logger.error(f"Could not find annotation file: {annotation_file}")
            return
    
    # Validate annotations
    valid, data = validate_coco_annotations(annotation_file, logger)
    if not valid:
        logger.error("Invalid annotations found, aborting")
        return
    
    # Split dataset
    train_data, val_data = split_coco_dataset(data, split_ratio, random_seed)
    
    # Save annotations
    save_coco_dataset(train_data, os.path.join(train_dir, '_annotations.coco.json'))
    save_coco_dataset(val_data, os.path.join(val_dir, '_annotations.coco.json'))
    
    # Copy images
    for img in tqdm(train_data['images'], desc="Copying train images"):
        src_path = os.path.join(data_dir, img['file_name'])
        dst_path = os.path.join(train_dir, img['file_name'])
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
    
    for img in tqdm(val_data['images'], desc="Copying val images"):
        src_path = os.path.join(data_dir, img['file_name'])
        dst_path = os.path.join(val_dir, img['file_name'])
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
    
    logger.info(f"Prepared tool detection data: {len(train_data['images'])} train images, "
                f"{len(val_data['images'])} val images")


def prepare_phase_recognition_data(data_dir: str, output_dir: str, config: Dict, logger: logging.Logger,
                                 force: bool = False, frame_interval: int = 5):
    """
    Prepare data for phase recognition model training.
    
    Args:
        data_dir: Data directory
        output_dir: Output directory
        config: Configuration dictionary
        logger: Logger instance
        force: Force overwrite existing data
        frame_interval: Frame interval for video extraction
    """
    logger.info("Preparing data for phase recognition model training")
    
    # Define directories
    frames_dir = os.path.join(output_dir, 'frames')
    phase_annotation_file = os.path.join(output_dir, 'phase_annotations.json')
    
    # Check if directories exist
    if os.path.exists(frames_dir) and os.path.exists(phase_annotation_file) and not force:
        logger.info("Phase recognition data already prepared, skipping")
        return
    
    # Create directories
    os.makedirs(frames_dir, exist_ok=True)
    
    # Look for videos
    video_dir = os.path.join(data_dir, 'videos')
    if not os.path.exists(video_dir):
        logger.error(f"Could not find video directory: {video_dir}")
        return
    
    # Extract frames from videos
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        logger.error(f"No video files found in: {video_dir}")
        return
    
    # Create phase annotation template
    phase_annotations = {
        'videos': {},
        'phases': {
            0: 'preparation',
            1: 'calot_triangle_dissection',
            2: 'clipping_and_cutting',
            3: 'gallbladder_dissection',
            4: 'gallbladder_packaging',
            5: 'cleaning_and_coagulation',
            6: 'gallbladder_extraction'
        }
    }
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        video_frames_dir = os.path.join(frames_dir, os.path.splitext(video_file)[0])
        
        # Extract frames
        frame_paths = extract_frames_from_video(
            video_path, 
            video_frames_dir, 
            interval=frame_interval
        )
        
        # Create placeholder annotations
        video_id = os.path.splitext(video_file)[0]
        phase_annotations['videos'][video_id] = {
            'frames': [os.path.basename(f) for f in frame_paths],
            'phase_annotations': {}
        }
    
    # Save phase annotations
    with open(phase_annotation_file, 'w') as f:
        json.dump(phase_annotations, f, indent=2)
    
    logger.info(f"Prepared phase recognition data: {len(video_files)} videos")
    logger.info(f"Created phase annotation template at: {phase_annotation_file}")
    logger.info("Please fill in phase annotations before training the phase recognition model")


def prepare_mistake_detection_data(data_dir: str, output_dir: str, config: Dict, logger: logging.Logger,
                                force: bool = False):
    """
    Prepare data for mistake detection model training.
    
    Args:
        data_dir: Data directory
        output_dir: Output directory
        config: Configuration dictionary
        logger: Logger instance
        force: Force overwrite existing data
    """
    logger.info("Preparing data for mistake detection model training")
    
    # Define files
    mistake_annotation_file = os.path.join(output_dir, 'mistake_annotations.json')
    
    # Check if files exist
    if os.path.exists(mistake_annotation_file) and not force:
        logger.info("Mistake detection data already prepared, skipping")
        return
    
    # Create mistake annotation template
    mistake_annotations = {
        'videos': {},
        'mistake_types': {
            'wrong_tool': 'Using incorrect tool for the current task',
            'dangerous_movement': 'Movement that risks damaging tissue',
            'poor_visibility': 'Operating with inadequate visibility',
            'incorrect_clip_placement': 'Improper placement of surgical clips',
            'inadequate_exposure': 'Failure to properly expose target area',
            'excessive_force': 'Using excessive force during dissection',
            'wrong_plane': 'Dissecting in incorrect anatomical plane'
        },
        'risk_thresholds': {
            'critical': 0.8,
            'major': 0.5,
            'minor': 0.0
        }
    }
    
    # Look for videos
    video_dir = os.path.join(data_dir, 'videos')
    if not os.path.exists(video_dir):
        logger.error(f"Could not find video directory: {video_dir}")
        return
    
    # Get video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        logger.error(f"No video files found in: {video_dir}")
        return
    
    # Create template for each video
    for video_file in video_files:
        video_id = os.path.splitext(video_file)[0]
        mistake_annotations['videos'][video_id] = {
            'mistakes': []
        }
    
    # Save mistake annotations
    with open(mistake_annotation_file, 'w') as f:
        json.dump(mistake_annotations, f, indent=2)
    
    logger.info(f"Prepared mistake detection data template for: {len(video_files)} videos")
    logger.info(f"Created mistake annotation template at: {mistake_annotation_file}")
    logger.info("Please fill in mistake annotations before training the mistake detection model")


def prepare_gpt_assistant_data(data_dir: str, output_dir: str, config: Dict, logger: logging.Logger,
                             force: bool = False):
    """
    Prepare data for GPT assistant model training.
    
    Args:
        data_dir: Data directory
        output_dir: Output directory
        config: Configuration dictionary
        logger: Logger instance
        force: Force overwrite existing data
    """
    logger.info("Preparing data for GPT assistant model training")
    
    # Define files
    procedure_knowledge_file = os.path.join(output_dir, 'procedure_knowledge.json')
    guidance_examples_file = os.path.join(output_dir, 'guidance_examples.json')
    
    # Check if files exist
    procedure_exists = os.path.exists(procedure_knowledge_file)
    guidance_exists = os.path.exists(guidance_examples_file)
    
    if procedure_exists and guidance_exists and not force:
        logger.info("GPT assistant data already prepared, skipping")
        return
    
    # Create template or copy existing procedure knowledge
    source_procedure_file = os.path.join(data_dir, 'procedure_knowledge.json')
    if os.path.exists(source_procedure_file) and not procedure_exists:
        shutil.copy(source_procedure_file, procedure_knowledge_file)
        logger.info(f"Copied procedure knowledge from: {source_procedure_file}")
    elif not procedure_exists:
        # Create procedure knowledge template
        procedure_knowledge = {
            "procedure_name": "Laparoscopic Cholecystectomy",
            "phases": [
                {
                    "phase_name": "preparation",
                    "description": "Initial setup and access creation",
                    "key_steps": [
                        "Position patient in reverse Trendelenburg position with left tilt",
                        "Establish pneumoperitoneum",
                        "Insert trocars under direct vision",
                        "Position camera and instruments"
                    ],
                    "recommended_tools": ["grasper", "hook"],
                    "critical_warnings": [
                        "Avoid injury to abdominal wall vessels during trocar insertion",
                        "Verify proper insufflation before trocar insertion"
                    ]
                },
                {
                    "phase_name": "calot_triangle_dissection",
                    "description": "Dissection of the Calot's triangle to identify cystic duct and artery",
                    "key_steps": [
                        "Grasp gallbladder fundus and retract superiorly and laterally",
                        "Dissect peritoneum overlying Calot's triangle",
                        "Identify cystic duct and cystic artery",
                        "Create critical view of safety"
                    ],
                    "recommended_tools": ["grasper", "hook", "scissors"],
                    "critical_warnings": [
                        "Maintain tension but avoid avulsion of gallbladder",
                        "Identify the correct plane for dissection",
                        "Stop if critical view cannot be achieved"
                    ]
                }
                # Additional phases would be filled in
            ]
        }
        
        with open(procedure_knowledge_file, 'w') as f:
            json.dump(procedure_knowledge, f, indent=2)
        
        logger.info(f"Created procedure knowledge template at: {procedure_knowledge_file}")
        logger.info("Please fill in procedure knowledge before training the GPT assistant model")
    
    # Create template or copy existing guidance examples
    source_guidance_file = os.path.join(data_dir, 'guidance_examples.json')
    if os.path.exists(source_guidance_file) and not guidance_exists:
        shutil.copy(source_guidance_file, guidance_examples_file)
        logger.info(f"Copied guidance examples from: {source_guidance_file}")
    elif not guidance_exists:
        # Create guidance examples template
        guidance_examples = {
            "examples": [
                {
                    "context": {
                        "phase": "calot_triangle_dissection",
                        "detected_tools": ["grasper", "hook"],
                        "current_actions": "Dissection of peritoneal covering"
                    },
                    "guidance": "Continue gentle dissection using hook diathermy. Maintain tension with grasper to expose the correct plane. Be careful to identify the cystic duct and artery separately."
                },
                {
                    "context": {
                        "phase": "clipping_and_cutting",
                        "detected_tools": ["grasper", "scissors"],
                        "current_actions": "Approaching cystic structures"
                    },
                    "guidance": "Switch to clipper tool. Apply three clips on the patient side and two clips on the gallbladder side for both cystic duct and artery. Verify correct placement before cutting."
                }
                # Additional examples would be filled in
            ]
        }
        
        with open(guidance_examples_file, 'w') as f:
            json.dump(guidance_examples, f, indent=2)
        
        logger.info(f"Created guidance examples template at: {guidance_examples_file}")
        logger.info("Please fill in guidance examples before training the GPT assistant model")


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting data preprocessing for SurgicalAI")
    
    # Set random seed
    set_random_seed(args.random_seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set output directory
    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which preprocessing tasks to run
    tasks = args.tasks
    if 'all' in tasks:
        tasks = ['tool_detection', 'phase_recognition', 'mistake_detection', 'gpt_assistant']
    
    # Run preprocessing tasks
    if 'tool_detection' in tasks:
        prepare_tool_detection_data(
            data_dir=args.data_dir,
            output_dir=output_dir,
            config=config,
            logger=logger,
            force=args.force,
            split_ratio=args.split_ratio,
            random_seed=args.random_seed
        )
    
    if 'phase_recognition' in tasks:
        prepare_phase_recognition_data(
            data_dir=args.data_dir,
            output_dir=output_dir,
            config=config,
            logger=logger,
            force=args.force,
            frame_interval=args.frame_interval
        )
    
    if 'mistake_detection' in tasks:
        prepare_mistake_detection_data(
            data_dir=args.data_dir,
            output_dir=output_dir,
            config=config,
            logger=logger,
            force=args.force
        )
    
    if 'gpt_assistant' in tasks:
        prepare_gpt_assistant_data(
            data_dir=args.data_dir,
            output_dir=output_dir,
            config=config,
            logger=logger,
            force=args.force
        )
    
    logger.info("Data preprocessing complete!")


if __name__ == "__main__":
    main() 
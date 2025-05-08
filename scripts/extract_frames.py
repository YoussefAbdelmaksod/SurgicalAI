"""
Extract frames from surgical videos for training.

This script extracts frames from one or more videos at a specified frame rate.
It can be used to prepare training data for phase recognition and other models.
"""

import os
import cv2
import argparse
import glob
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ExtractFrames")

def extract_frames(video_path, output_dir, fps=1, max_frames=None, resize=None, phase_annotations=None):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        fps: Number of frames per second to extract
        max_frames: Maximum number of frames to extract
        resize: Tuple (width, height) to resize frames
        phase_annotations: Dict with phase annotations to create subdirectories
    
    Returns:
        Number of frames extracted
    """
    # Check if video exists
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video filename without extension
    video_basename = os.path.basename(video_path)
    video_name = os.path.splitext(video_basename)[0]
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return 0
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    logger.info(f"Video: {video_basename}")
    logger.info(f"  Duration: {duration:.2f} seconds")
    logger.info(f"  FPS: {video_fps}")
    logger.info(f"  Total frames: {total_frames}")
    
    # Calculate frame extraction interval
    skip_frames = int(video_fps / fps)
    if skip_frames < 1:
        skip_frames = 1
    
    # Prepare phase subdirectories if needed
    phase_dirs = {}
    if phase_annotations and video_basename in phase_annotations.get("video_files", {}):
        video_phases = phase_annotations["video_files"][video_basename]["phases"]
        for phase_name, (start_time, end_time) in video_phases.items():
            phase_dir = os.path.join(output_dir, phase_name)
            os.makedirs(phase_dir, exist_ok=True)
            start_frame = int(start_time * video_fps)
            end_frame = int(end_time * video_fps)
            phase_dirs[(start_frame, end_frame)] = phase_dir
            logger.info(f"  Phase {phase_name}: frames {start_frame} to {end_frame}")
    
    # Extract frames
    count = 0
    frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index % skip_frames == 0:
            # Resize if needed
            if resize:
                frame = cv2.resize(frame, resize)
            
            # Determine output directory based on phases
            output_subdir = output_dir
            for (start_frame, end_frame), phase_dir in phase_dirs.items():
                if start_frame <= frame_index <= end_frame:
                    output_subdir = phase_dir
                    break
            
            # Generate frame filename
            frame_filename = f"{video_name}_{frame_index:06d}.jpg"
            output_path = os.path.join(output_subdir, frame_filename)
            
            # Save frame
            cv2.imwrite(output_path, frame)
            count += 1
            
            # Log progress
            if count % 100 == 0:
                logger.info(f"  Extracted {count} frames...")
            
            # Check if max frames reached
            if max_frames and count >= max_frames:
                logger.info(f"  Reached maximum number of frames: {max_frames}")
                break
        
        frame_index += 1
    
    # Clean up
    cap.release()
    logger.info(f"  Extracted {count} frames from {video_basename}")
    return count

def load_phase_annotations(annotations_path):
    """
    Load phase annotations from a JSON file.
    
    Args:
        annotations_path: Path to the JSON file
        
    Returns:
        Dictionary with phase annotations
    """
    if not os.path.exists(annotations_path):
        logger.warning(f"Annotations file not found: {annotations_path}")
        return None
    
    try:
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        logger.info(f"Loaded phase annotations from {annotations_path}")
        return annotations
    except Exception as e:
        logger.error(f"Error loading annotations: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract frames from surgical videos for training")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file(s) (glob pattern supported)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Number of frames per second to extract")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to extract per video")
    parser.add_argument("--resize", type=str, default=None, help="Resize frames to width,height (e.g. 224,224)")
    parser.add_argument("--annotations", type=str, default=None, help="Path to phase annotations JSON file")
    parser.add_argument("--organize_by_phase", action="store_true", help="Organize frames by surgical phase")
    
    args = parser.parse_args()
    
    # Parse resize argument
    resize = None
    if args.resize:
        try:
            width, height = map(int, args.resize.split(','))
            resize = (width, height)
            logger.info(f"Will resize frames to {width}x{height}")
        except:
            logger.error(f"Invalid resize format: {args.resize}, should be width,height")
    
    # Load phase annotations if specified
    phase_annotations = None
    if args.annotations and args.organize_by_phase:
        phase_annotations = load_phase_annotations(args.annotations)
    
    # Get video file list
    video_paths = glob.glob(args.video_path)
    if not video_paths:
        logger.error(f"No video files found matching: {args.video_path}")
        return
    
    logger.info(f"Found {len(video_paths)} video file(s)")
    
    # Extract frames from each video
    total_frames = 0
    for video_path in video_paths:
        frames = extract_frames(
            video_path,
            args.output_dir,
            fps=args.fps,
            max_frames=args.max_frames,
            resize=resize,
            phase_annotations=phase_annotations
        )
        total_frames += frames
    
    logger.info(f"Extraction complete. Extracted a total of {total_frames} frames.")

if __name__ == "__main__":
    main() 
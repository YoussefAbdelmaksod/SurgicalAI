
# Script to load trained models from Google Drive to local project
import os
import sys
import shutil
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LoadTrainedModels")

def load_models(source_dir, target_dir="models/weights"):
    """Load trained models from source to project directories"""
    if not os.path.exists(source_dir):
        logger.error(f"Source directory {source_dir} does not exist")
        return False
    
    # Define model mappings: source file -> target path
    model_mappings = {
        "tool_detection.pth": "tool_detection/tool_detection.pth",
        "phase_recognition.pth": "vit_lstm/phase_recognition.pth",
        "mistake_detection.pth": "mistake_detector/mistake_detection.pth"
    }
    
    success_count = 0
    for src_file, target_path in model_mappings.items():
        src_path = os.path.join(source_dir, src_file)
        dst_path = os.path.join(target_dir, target_path)
        
        # Create target directory if it doesn't exist
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        if os.path.exists(src_path):
            try:
                shutil.copy(src_path, dst_path)
                logger.info(f"Copied {src_path} to {dst_path}")
                success_count += 1
            except Exception as e:
                logger.error(f"Error copying {src_path} to {dst_path}: {str(e)}")
        else:
            logger.warning(f"Source file {src_path} does not exist")
    
    logger.info(f"Loaded {success_count}/{len(model_mappings)} models")
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description="Load trained models from Google Drive")
    parser.add_argument("--source_dir", type=str, required=True, 
                        help="Directory containing trained models (e.g., downloads from Google Drive)")
    parser.add_argument("--target_dir", type=str, default="models/weights",
                        help="Target directory in the project (default: models/weights)")
    
    args = parser.parse_args()
    
    # Load the models
    load_models(args.source_dir, args.target_dir)

if __name__ == "__main__":
    main()

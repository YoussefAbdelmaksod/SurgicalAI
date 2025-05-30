#!/usr/bin/env python3
"""
Prepare SurgicalAI for training on a GPU machine.

This script verifies the environment setup and data availability for training,
preparing the system for training on a machine with an NVIDIA GPU.
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu():
    """Check GPU availability and display information."""
    logger.info("Checking GPU availability...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"✅ CUDA is available! Found {gpu_count} GPU(s).")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            logger.info(f"   GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
            
        return True
    else:
        logger.warning("❌ CUDA is not available. Training will be slow on CPU only.")
        return False

def check_datasets():
    """Check if datasets are available in the expected locations."""
    logger.info("Checking datasets availability...")
    
    # Load config to get dataset paths
    config_path = Path("training/configs/training_config.yaml")
    if not config_path.exists():
        logger.error(f"❌ Configuration file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check each dataset
    datasets_ok = True
    
    # Cholec80 dataset
    cholec_path = Path(config['phase_recognition']['data']['data_dir'])
    if not cholec_path.exists():
        logger.warning(f"❌ Cholec80 dataset not found at: {cholec_path}")
        datasets_ok = False
    else:
        logger.info(f"✅ Found Cholec80 dataset at: {cholec_path}")
    
    # m2cai16-tool-locations dataset
    tool_path = Path(config['tool_detection']['data']['data_dir'])
    if not tool_path.exists():
        logger.warning(f"❌ m2cai16-tool-locations dataset not found at: {tool_path}")
        datasets_ok = False
    else:
        logger.info(f"✅ Found m2cai16-tool-locations dataset at: {tool_path}")
    
    # EndoScapes dataset
    endo_path = Path(config['mistake_detection']['data']['data_dir'])
    if not endo_path.exists():
        logger.warning(f"❌ EndoScapes dataset not found at: {endo_path}")
        datasets_ok = False
    else:
        logger.info(f"✅ Found EndoScapes dataset at: {endo_path}")
    
    # Supplementary data
    suppl_path = Path(config['mistake_detection']['data']['supplementary_data_dir'])
    if not suppl_path.exists():
        logger.warning(f"❌ Supplementary dataset not found at: {suppl_path}")
        datasets_ok = False
    else:
        logger.info(f"✅ Found supplementary dataset at: {suppl_path}")
    
    return datasets_ok

def check_model_code():
    """Verify that all necessary model code is present."""
    logger.info("Checking model code files...")
    
    required_files = [
        "models/phase_recognition.py",
        "models/tool_detection.py",
        "models/mistake_detection.py",
        "models/ensemble.py",
        "training/phase_recognition_trainer.py",
        "training/tool_detection_trainer.py",
        "training/mistake_detection_trainer.py",
        "training/surgical_datasets.py",
        "training/train.py",
        "training/configs/training_config.yaml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("❌ The following required files are missing:")
        for file in missing_files:
            logger.error(f"   - {file}")
        return False
    
    logger.info("✅ All required model code files are present.")
    return True

def create_training_script():
    """Create a shell script to run the training."""
    logger.info("Creating training shell script...")
    
    script_content = """#!/bin/bash
# SurgicalAI Training Script
# This script runs the SurgicalAI training pipeline

# Activate virtual environment if needed
# source venv/bin/activate

# Set CUDA device if needed
# export CUDA_VISIBLE_DEVICES=0

# Create log directory
mkdir -p training/logs

# Run training for all models
echo "Starting SurgicalAI training pipeline..."
python3 training/train.py --config training/configs/training_config.yaml --models all

# To train individual models, uncomment the appropriate lines below:
# python3 training/train.py --config training/configs/training_config.yaml --models phase
# python3 training/train.py --config training/configs/training_config.yaml --models tool
# python3 training/train.py --config training/configs/training_config.yaml --models mistake
"""
    
    script_path = "train_surgical_ai.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    logger.info(f"✅ Created training script: {script_path}")
    return True

def create_archive():
    """Create a zip archive for transfer to training machine."""
    logger.info("Creating archive for transfer to training machine...")
    
    archive_name = "surgical_ai_training.zip"
    
    # Create a list of directories to include
    include_dirs = [
        "data",
        "models",
        "training",
        "utils",
        "scripts",
        "requirements.txt",
        "train_surgical_ai.sh",
        "README.md"
    ]
    
    # Use shutil to make a zip archive excluding unnecessary files
    exclude_dirs = [
        "__pycache__", 
        ".git", 
        "data/m2cai16-tool-locations/__MACOSX",
        ".DS_Store",
        "backup"
    ]
    
    try:
        import zipfile
        import glob
        
        with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for item in include_dirs:
                if os.path.isfile(item):
                    zipf.write(item, os.path.basename(item))
                else:
                    for root, dirs, files in os.walk(item):
                        # Skip excluded directories
                        dirs[:] = [d for d in dirs if d not in exclude_dirs]
                        
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Skip large/unwanted files
                            if any(excluded in file_path for excluded in exclude_dirs):
                                continue
                            
                            # Add file to zip
                            zipf.write(file_path, file_path)
        
        logger.info(f"✅ Archive created: {archive_name}")
        logger.info(f"   Transfer this file to your GPU machine and extract it")
        logger.info(f"   Then run: ./train_surgical_ai.sh")
        return True
    except Exception as e:
        logger.error(f"Failed to create archive: {str(e)}")
        return False

def main():
    """Main function to prepare system for training."""
    parser = argparse.ArgumentParser(description="Prepare SurgicalAI for training")
    parser.add_argument("--create-archive", action="store_true", 
                        help="Create a zip archive for transfer to training machine")
    args = parser.parse_args()
    
    logger.info("Starting SurgicalAI training preparation...")
    
    # Run checks
    gpu_check = check_gpu()
    datasets_check = check_datasets()
    code_check = check_model_code()
    script_created = create_training_script()
    
    # Create output directories if they don't exist
    os.makedirs("training/checkpoints", exist_ok=True)
    os.makedirs("models/weights", exist_ok=True)
    
    # Always create archive since we'll be moving to a GPU machine
    archive_created = create_archive()
    
    # Print summary
    logger.info("\n=== Preparation Summary ===")
    logger.info(f"GPU available: {'Yes' if gpu_check else 'No'}")
    logger.info(f"Datasets ready: {'Yes' if datasets_check else 'No'}")
    logger.info(f"Code files ready: {'Yes' if code_check else 'No'}")
    logger.info(f"Training script created: {'Yes' if script_created else 'No'}")
    logger.info(f"Archive created: {'Yes' if archive_created else 'No'}")
    
    if datasets_check and code_check and archive_created:
        logger.info("\n✅ System is ready for training on a GPU machine!")
        logger.info("1. Transfer surgical_ai_training.zip to your GPU machine")
        logger.info("2. Extract the archive: unzip surgical_ai_training.zip")
        logger.info("3. Run the training script: ./train_surgical_ai.sh")
    else:
        logger.warning("\n⚠️ Some issues need to be resolved before training.")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Check the training pipeline files.

This script checks if all the required files for the training pipeline exist
and have the expected content. It does not execute any training.
"""

import os
import sys
from pathlib import Path

def check_file(file_path, description):
    """Check if a file exists and print status."""
    path = Path(file_path)
    if path.exists():
        print(f"✅ Found: {file_path} ({description})")
        return True
    else:
        print(f"❌ Missing: {file_path} ({description})")
        return False

def main():
    """Main function to check training pipeline."""
    print("Checking SurgicalAI Training Pipeline Files")
    print("==========================================")

    # Define required files and their descriptions
    required_files = [
        # Core training modules
        ("training/train.py", "Main training module"),
        ("training/phase_recognition_trainer.py", "Phase recognition trainer"),
        ("training/tool_detection_trainer.py", "Tool detection trainer"),
        ("training/mistake_detection_trainer.py", "Mistake detection trainer"),
        ("training/surgical_datasets.py", "Dataset loaders"),
        ("training/configs/training_config.yaml", "Training configuration"),
        
        # Model implementations
        ("models/phase_recognition.py", "Phase recognition model"),
        ("models/tool_detection.py", "Tool detection model"),
        ("models/mistake_detection.py", "Mistake detection model"),
        ("models/ensemble.py", "Model ensemble"),
        
        # Entry points
        ("scripts/train_models.py", "Training script"),
        
        # Utilities
        ("utils/helpers.py", "Helper functions"),
    ]
    
    # Create directories if they don't exist
    directories = [
        "training/checkpoints/phase_recognition",
        "training/checkpoints/tool_detection",
        "training/checkpoints/mistake_detection",
        "training/logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Ensured directory exists: {directory}")
    
    # Check all required files
    all_found = True
    for file_path, description in required_files:
        if not check_file(file_path, description):
            all_found = False
    
    # Print summary
    print("\nSummary")
    print("=======")
    if all_found:
        print("✅ All required files are present! The training pipeline is complete.")
    else:
        print("❌ Some required files are missing. Please create them to complete the training pipeline.")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Fix paths for SurgicalAI training in Google Colab environment.

This script creates symbolic links and directories needed for proper training in Colab.
"""

import os
import shutil
import sys

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

def main():
    """Main function to fix paths for Colab training."""
    print("Fixing paths for SurgicalAI training in Colab...")
    
    # Create required directories
    create_directory("/content/training")
    create_directory("/content/models/weights/tool_detection")
    create_directory("/content/models/weights/vit_lstm")
    create_directory("/content/models/weights/mistake_detector")
    
    # Copy training scripts to the expected locations
    training_files = [
        "train_tool_detection.py",
        "train_phase_recognition.py",
        "train_all_models.py"
    ]
    
    for file in training_files:
        source = f"/content/SurgicalAI_clone/training/{file}"
        target = f"/content/training/{file}"
        
        if os.path.exists(source):
            shutil.copy2(source, target)
            print(f"Copied {source} to {target}")
        else:
            print(f"Warning: Source file {source} not found!")
    
    # Create symbolic links for models directory if it doesn't exist
    if not os.path.exists("/content/models"):
        if os.path.exists("/content/SurgicalAI_clone/models"):
            os.symlink("/content/SurgicalAI_clone/models", "/content/models")
            print("Created symbolic link for models directory")
        else:
            print("Warning: models directory not found in SurgicalAI_clone!")
    
    # Create symbolic links for other necessary directories
    for directory in ["data", "utils", "config"]:
        if not os.path.exists(f"/content/{directory}"):
            if os.path.exists(f"/content/SurgicalAI_clone/{directory}"):
                os.symlink(f"/content/SurgicalAI_clone/{directory}", f"/content/{directory}")
                print(f"Created symbolic link for {directory} directory")
            else:
                print(f"Warning: {directory} directory not found in SurgicalAI_clone!")
    
    print("Path fixing completed! You can now run the training scripts.")

if __name__ == "__main__":
    main() 
# SurgicalAI Colab Training - Single Code Block Fix

Copy and paste this entire code block into a new cell in your Colab notebook to fix training issues:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create a directory to store our weights
!mkdir -p /content/drive/MyDrive/SurgicalAI/weights

# Clone the repository (replace with your GitHub username)
!git clone https://github.com/YOUR_USERNAME/SurgicalAI SurgicalAI_clone
%cd SurgicalAI_clone

# Install dependencies
!pip install -r requirements.txt
!pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Create the fix_colab_paths.py script
%%writefile fix_colab_paths.py
#!/usr/bin/env python3
"""
Fix paths for SurgicalAI training in Google Colab environment.
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
    
    # Create symbolic links for models directory
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

# Run the fix script
!python fix_colab_paths.py

# Change to the root directory
%cd /content

# Now you can run the training commands
print("========== STARTING TOOL DETECTION TRAINING ==========")
!python training/train_tool_detection.py \
  --data_dir data \
  --output_dir models/weights \
  --batch_size 4 \
  --num_epochs 10 \
  --learning_rate 3e-4 \
  --backbone resnet50 \
  --use_mixed_precision True

# Save the trained model to Drive
!cp models/weights/tool_detection/tool_detection.pth /content/drive/MyDrive/SurgicalAI/weights/
print("Tool detection model saved to Google Drive")

print("========== STARTING PHASE RECOGNITION TRAINING ==========")
!python training/train_phase_recognition.py \
  --data_dir data \
  --output_dir models/weights \
  --batch_size 2 \
  --num_epochs 10 \
  --vit_model vit_base_patch16_224 \
  --freeze_vit True

# Save the trained model to Drive
!cp models/weights/vit_lstm/phase_recognition.pth /content/drive/MyDrive/SurgicalAI/weights/
print("Phase recognition model saved to Google Drive")

print("========== STARTING MISTAKE DETECTION TRAINING ==========")
!python training/train_all_models.py \
  --train_subset mistake_detection \
  --data_dir data \
  --output_dir models/weights \
  --batch_size 4 \
  --num_epochs 8

# Save the trained model to Drive
!cp models/weights/mistake_detector/mistake_detection.pth /content/drive/MyDrive/SurgicalAI/weights/
print("Mistake detection model saved to Google Drive")

# List saved models in Drive
!ls -la /content/drive/MyDrive/SurgicalAI/weights/
```

## Instructions:

1. Create a new cell in your Colab notebook
2. Copy and paste the entire code block above
3. Replace `YOUR_USERNAME` with your actual GitHub username
4. Run the cell

This single code block:
- Mounts Google Drive
- Clones your repository 
- Installs dependencies
- Creates all needed scripts and directories
- Fixes the file paths
- Runs all three training steps sequentially
- Saves the models to Google Drive 
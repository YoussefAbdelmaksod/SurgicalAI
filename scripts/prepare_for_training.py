"""
Prepare the SurgicalAI project for training on Google Colab.

This script:
1. Creates necessary directory structure
2. Validates annotations and data
3. Prepares configuration files for Colab
4. Performs a lightweight check of the data
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("PrepareForTraining")

def setup_directory_structure():
    """Create necessary directories for training"""
    directories = [
        "data/annotations",
        "data/train_processed",
        "data/phases",
        "data/mistakes",
        "models/weights/tool_detection",
        "models/weights/vit_lstm",
        "models/weights/mistake_detector",
        "colab_training"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def validate_coco_annotations():
    """Validate and prepare COCO annotations for tool detection"""
    coco_file = "data/train/_annotations.coco.json"
    target_file = "data/annotations/tool_annotations.json"
    
    # Check if COCO annotations are valid
    if os.path.exists(coco_file):
        try:
            with open(coco_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith("version https://git-lfs.github.com"):
                    logger.warning(f"COCO annotations file {coco_file} appears to be a Git LFS pointer, not actual data")
                    logger.warning("You need to create or download actual COCO annotations")
                    # Create a template file
                    create_template_annotations()
                else:
                    # Try to copy the file
                    shutil.copy(coco_file, target_file)
                    logger.info(f"Copied annotations from {coco_file} to {target_file}")
        except Exception as e:
            logger.error(f"Error validating COCO annotations: {str(e)}")
            create_template_annotations()
    else:
        logger.warning(f"COCO annotations file {coco_file} not found")
        create_template_annotations()

def create_template_annotations():
    """Create template annotation files"""
    # Tool detection annotations template
    tool_annotations = {
        "info": {"description": "SurgicalAI Tool Detection Dataset"},
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Grasper", "supercategory": "Tool"},
            {"id": 2, "name": "Scissors", "supercategory": "Tool"},
            {"id": 3, "name": "Clipper", "supercategory": "Tool"},
            {"id": 4, "name": "Hook", "supercategory": "Tool"},
            {"id": 5, "name": "Irrigator", "supercategory": "Tool"}
        ]
    }
    
    with open("data/annotations/tool_annotations.json", 'w') as f:
        json.dump(tool_annotations, f, indent=2)
    logger.info("Created template tool annotations file")
    
    # Phase annotations template
    phase_annotations = {
        "phases": {
            "preparation": {"start_time": 0, "end_time": 120},
            "calot_triangle_dissection": {"start_time": 120, "end_time": 600},
            "clipping_and_cutting": {"start_time": 600, "end_time": 840},
            "gallbladder_dissection": {"start_time": 840, "end_time": 1620},
            "gallbladder_packaging": {"start_time": 1620, "end_time": 1800},
            "cleaning_and_coagulation": {"start_time": 1800, "end_time": 2040},
            "gallbladder_extraction": {"start_time": 2040, "end_time": 2220}
        },
        "video_files": {
            "Laparoscopic Cholecystectomy High Definition Full Length Video.mp4": {
                "duration": 2220,
                "fps": 30,
                "phases": {
                    "preparation": [0, 120],
                    "calot_triangle_dissection": [120, 600],
                    "clipping_and_cutting": [600, 840],
                    "gallbladder_dissection": [840, 1620],
                    "gallbladder_packaging": [1620, 1800],
                    "cleaning_and_coagulation": [1800, 2040],
                    "gallbladder_extraction": [2040, 2220]
                }
            }
        }
    }
    
    with open("data/annotations/phase_annotations.json", 'w') as f:
        json.dump(phase_annotations, f, indent=2)
    logger.info("Created template phase annotations file")

def copy_training_files():
    """Copy training images to processed directory"""
    train_dir = "data/train"
    processed_dir = "data/train_processed"
    
    if os.path.exists(train_dir) and os.path.isdir(train_dir):
        img_files = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if img_files:
            for img_file in img_files:
                src = os.path.join(train_dir, img_file)
                dst = os.path.join(processed_dir, img_file)
                shutil.copy(src, dst)
            logger.info(f"Copied {len(img_files)} images to {processed_dir}")
        else:
            logger.warning(f"No image files found in {train_dir}")
    else:
        logger.warning(f"Training directory {train_dir} not found or not a directory")

def create_colab_notebook():
    """Create a comprehensive Colab notebook for training"""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# SurgicalAI Training Notebook\n",
                    "\n",
                    "This notebook trains all components of the SurgicalAI system:\n",
                    "1. Tool Detection\n",
                    "2. Phase Recognition\n",
                    "3. Mistake Detection\n",
                    "\n",
                    "Each section can be run independently."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Setup environment and mount Google Drive\n",
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')\n",
                    "\n",
                    "# Create a directory to store our weights\n",
                    "!mkdir -p /content/drive/MyDrive/SurgicalAI/weights\n",
                    "\n",
                    "# Clone the repository\n",
                    "!git clone https://github.com/YOUR_USERNAME/SurgicalAI\n",
                    "%cd SurgicalAI\n",
                    "\n",
                    "# Install dependencies\n",
                    "!pip install -r requirements.txt\n",
                    "!pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Upload Training Data\n",
                    "\n",
                    "Upload your training data or copy from Drive if already uploaded."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Option 1: Upload data directly to this Colab session\n",
                    "# from google.colab import files\n",
                    "# uploaded = files.upload()  # Upload annotation files\n",
                    "\n",
                    "# Option 2: Copy from Google Drive if already uploaded\n",
                    "!mkdir -p data/annotations\n",
                    "!mkdir -p data/train_processed\n",
                    "!mkdir -p data/phases\n",
                    "\n",
                    "# Copy your data from Drive (uncomment and modify paths as needed)\n",
                    "# !cp /content/drive/MyDrive/SurgicalAI/data/annotations/* data/annotations/\n",
                    "# !cp -r /content/drive/MyDrive/SurgicalAI/data/train_processed/* data/train_processed/\n",
                    "# !cp -r /content/drive/MyDrive/SurgicalAI/data/phases/* data/phases/"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Tool Detection Training"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Train the tool detection model\n",
                    "!python training/train_tool_detection.py \\\n",
                    "  --data_dir data \\\n",
                    "  --output_dir models/weights \\\n",
                    "  --batch_size 4 \\\n",
                    "  --num_epochs 20 \\\n",
                    "  --learning_rate 3e-4 \\\n",
                    "  --backbone resnet50 \\\n",
                    "  --use_mixed_precision True\n",
                    "\n",
                    "# Save the trained model to Drive\n",
                    "!cp models/weights/tool_detection/tool_detection.pth /content/drive/MyDrive/SurgicalAI/weights/"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Phase Recognition Training"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Train the phase recognition model\n",
                    "!python training/train_phase_recognition.py \\\n",
                    "  --data_dir data \\\n",
                    "  --output_dir models/weights/vit_lstm \\\n",
                    "  --batch_size 2 \\\n",
                    "  --num_epochs 15 \\\n",
                    "  --vit_model vit_base_patch16_224 \\\n",
                    "  --freeze_vit True\n",
                    "\n",
                    "# Save the trained model to Drive\n",
                    "!cp models/weights/vit_lstm/phase_recognition.pth /content/drive/MyDrive/SurgicalAI/weights/"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Mistake Detection Training"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Train the mistake detection model\n",
                    "!python training/train_all_models.py \\\n",
                    "  --train_subset mistake_detection \\\n",
                    "  --data_dir data \\\n",
                    "  --output_dir models/weights \\\n",
                    "  --batch_size 4 \\\n",
                    "  --num_epochs 10\n",
                    "\n",
                    "# Save the trained model to Drive\n",
                    "!cp models/weights/mistake_detector/mistake_detection.pth /content/drive/MyDrive/SurgicalAI/weights/"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Verify and Download Models\n",
                    "\n",
                    "Check that all models are trained and saved."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# List saved models in Drive\n",
                    "!ls -la /content/drive/MyDrive/SurgicalAI/weights/\n",
                    "\n",
                    "# Download models directly from Colab if needed\n",
                    "from google.colab import files\n",
                    "\n",
                    "# Uncomment to download specific models\n",
                    "# files.download('/content/drive/MyDrive/SurgicalAI/weights/tool_detection.pth')\n",
                    "# files.download('/content/drive/MyDrive/SurgicalAI/weights/phase_recognition.pth')\n",
                    "# files.download('/content/drive/MyDrive/SurgicalAI/weights/mistake_detection.pth')"
                ]
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    with open("colab_training/SurgicalAI_Training.ipynb", 'w') as f:
        json.dump(notebook_content, f, indent=2)
    logger.info("Created Colab training notebook at colab_training/SurgicalAI_Training.ipynb")

def create_model_loader_script():
    """Create script to load trained models from Drive"""
    content = '''
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
'''
    
    with open("scripts/load_trained_models.py", 'w') as f:
        f.write(content)
    logger.info("Created model loader script at scripts/load_trained_models.py")

def update_readme():
    """Update README with training instructions"""
    readme_content = """# SurgicalAI Training Instructions

## Project Structure
The SurgicalAI project is set up with a structure that separates:
- Model architecture (in `models/`)
- Training code (in `training/`)
- Data processing (in `data/`)
- Application logic (in `app/`)

## Training Workflow

### 1. Prepare for Training
Run the preparation script:
```
python scripts/prepare_for_training.py
```

This script creates the necessary directories and template files for training.

### 2. Prepare Your Data

#### Tool Detection
- Place training images in `data/train_processed/`
- Update tool annotations in `data/annotations/tool_annotations.json`

#### Phase Recognition
- Extract frames from videos to `data/phases/`
- Update phase timestamps in `data/annotations/phase_annotations.json`

#### Mistake Detection
- Uses the existing `data/mistake_annotations.json` file
- Update with real mistake data if available

### 3. Train on Google Colab
1. Upload the `colab_training/SurgicalAI_Training.ipynb` notebook to Google Colab
2. Follow the instructions in the notebook to upload your data
3. Run the training cells for each model type
4. Save the trained models to Google Drive

### 4. Use Trained Models Locally
Download the trained models from Google Drive and load them:
```
python scripts/load_trained_models.py --source_dir path/to/downloaded/models
```

## GitHub Instructions
When pushing to GitHub:
1. Add `models/weights/*/` to your `.gitignore` file to avoid pushing large model files
2. Include all code and configuration files
3. Include small annotation files but exclude large datasets
"""
    
    with open("colab_training/README.md", 'w') as f:
        f.write(readme_content)
    logger.info("Created training README at colab_training/README.md")

def create_gitignore():
    """Create appropriate .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# PyCharm
.idea/

# VS Code
.vscode/

# Training data and models
models/weights/**/*.pth
data/videos/*.mp4
data/videos/*.ts
data/train/*.jpg
data/train/*.png
data/train_processed/*.jpg
data/train_processed/*.png
data/phases/*.jpg
data/phases/*.png

# Allow empty directories with .gitkeep
!**/.gitkeep
!models/weights/tool_detection/.gitkeep
!models/weights/vit_lstm/.gitkeep
!models/weights/mistake_detector/.gitkeep

# Logs
logs/
*.log
"""
    
    with open(".gitignore", 'w') as f:
        f.write(gitignore_content)
    logger.info("Created .gitignore file")

def create_gitkeep_files():
    """Create .gitkeep files to maintain directory structure in Git"""
    gitkeep_dirs = [
        "models/weights/tool_detection",
        "models/weights/vit_lstm",
        "models/weights/mistake_detector",
        "data/train_processed",
        "data/phases",
        "data/annotations"
    ]
    
    for directory in gitkeep_dirs:
        with open(os.path.join(directory, ".gitkeep"), 'w') as f:
            pass
        logger.info(f"Created .gitkeep in {directory}")

def check_data_integrity():
    """Check if necessary data files exist"""
    video_file = "data/videos/Laparoscopic Cholecystectomy High Definition Full Length Video.mp4"
    
    if os.path.exists(video_file):
        logger.info(f"Found main video file: {video_file}")
    else:
        logger.warning(f"Main video file not found: {video_file}")
        logger.warning("Please ensure you have the required video files for training")
    
    # Check number of training images
    train_dir = "data/train"
    if os.path.exists(train_dir):
        img_files = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        logger.info(f"Found {len(img_files)} training images in {train_dir}")
    else:
        logger.warning(f"Training directory {train_dir} not found")

def main():
    logger.info("Preparing SurgicalAI project for Colab training")
    
    # Create necessary directories
    setup_directory_structure()
    
    # Create .gitkeep files
    create_gitkeep_files()
    
    # Validate and prepare COCO annotations
    validate_coco_annotations()
    
    # Copy training files
    copy_training_files()
    
    # Create Colab notebook
    create_colab_notebook()
    
    # Create model loader script
    create_model_loader_script()
    
    # Create README with training instructions
    update_readme()
    
    # Create .gitignore
    create_gitignore()
    
    # Check data integrity
    check_data_integrity()
    
    logger.info("Project preparation completed. Ready for training on Google Colab!")
    logger.info("Next steps:")
    logger.info("1. Ensure your annotation files are properly filled with real data")
    logger.info("2. Push your project to GitHub (excluding large files as specified in .gitignore)")
    logger.info("3. Upload the colab_training/SurgicalAI_Training.ipynb notebook to Google Colab")
    logger.info("4. Follow the instructions in the notebook to train your models")

if __name__ == "__main__":
    main() 
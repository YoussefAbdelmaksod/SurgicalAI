# SurgicalAI Training Instructions

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

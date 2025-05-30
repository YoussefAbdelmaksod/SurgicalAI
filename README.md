# SurgicalAI

A personalized guidance system for laparoscopic cholecystectomy, powered by deep learning.

## Overview

SurgicalAI combines computer vision, temporal modeling, and personalized guidance to enhance surgical workflow during laparoscopic cholecystectomy procedures. The system uses a multi-model approach to recognize surgical phases, detect instruments, identify potential mistakes, and provide real-time guidance adapted to the surgeon's experience level.

Key features:
- **Phase recognition** with ViT-LSTM model for temporal understanding of surgical workflow
- **Tool detection** for instrument tracking using Faster R-CNN
- **Mistake detection** to identify potential errors with risk assessment
- **Personalized guidance** based on surgeon experience level
- **Voice assistant** for hands-free interaction

## Project Structure

```
SurgicalAI/
├── app/                    # Main application
│   └── main.py             # Entry point for the application
├── data/                   # Data directory
│   ├── Cholec80.v5-cholec80-10-2.coco/  # Cholec80 dataset in COCO format
│   ├── endoscapes/         # EndoScapes dataset
│   ├── m2cai16-tool-locations/ # m2cai16 tool dataset
│   ├── procedure_knowledge.json  # Knowledge base for procedures
│   └── videos/             # Input videos for inference
├── models/                 # Model implementations
│   ├── gpt_guidance.py     # GPT-based guidance module
│   ├── mistake_detection.py # Mistake detection model
│   ├── phase_recognition.py # Phase recognition model
│   ├── tool_detection.py   # Tool detection model
│   └── voice_assistant.py  # Voice assistant module
├── scripts/                # Utility scripts
│   ├── initialize_system.py # System initialization
│   ├── run_inference.py    # Run inference on videos
│   ├── setup_profiles.py   # Set up user profiles
│   └── train_models.py     # Train all models
├── training/               # Training pipeline
│   ├── configs/            # Training configurations
│   ├── checkpoints/        # Model checkpoints
│   ├── logs/               # Training logs
│   ├── mistake_detection_trainer.py # Mistake detection trainer
│   ├── phase_recognition_trainer.py # Phase recognition trainer
│   ├── surgical_datasets.py # Dataset loaders
│   ├── tool_detection_trainer.py # Tool detection trainer
│   └── train.py            # Main training script
└── utils/                  # Utilities
    ├── helpers.py          # Helper functions
    └── user_profiles.py    # User profile management
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SurgicalAI.git
cd SurgicalAI
```

2. Set up a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download datasets (instructions in `data/README.md`)

## Training Models

### Prepare the data

The system requires three datasets:
- Cholec80 (in COCO format) for phase recognition
- m2cai16-tool-locations for tool detection
- EndoScapes for mistake detection

Place these datasets in their respective folders under the `data/` directory.

### Configuration

Edit the training configuration in `training/configs/training_config.yaml` to customize hyperparameters, model architecture, and training settings.

### Train individual models

To train specific models:

```bash
# Train phase recognition model
python scripts/train_models.py --models phase

# Train tool detection model
python scripts/train_models.py --models tool

# Train mistake detection model
python scripts/train_models.py --models mistake
```

### Train all models

To train all models in sequence:

```bash
python scripts/train_models.py --models all
```

### Resume training

To resume training from a checkpoint:

```bash
python scripts/train_models.py --models phase --resume --phase-ckpt training/checkpoints/phase_recognition/best_model.pth
```

## Inference

Run inference on a video:

```bash
python scripts/run_inference.py --video data/videos/test_video.mp4 --output results/
```

## System Setup

Set up user profiles:

```bash
python scripts/setup_profiles.py
```

Initialize the system:

```bash
python scripts/initialize_system.py
```

## Main Application

Run the main application:

```bash
python app/main.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Cholec80 dataset: [http://camma.u-strasbg.fr/datasets](http://camma.u-strasbg.fr/datasets)
- m2cai16-tool-locations dataset: [http://camma.u-strasbg.fr/m2cai2016](http://camma.u-strasbg.fr/m2cai2016)
- EndoScapes dataset: [https://endoscapes.grand-challenge.org/](https://endoscapes.grand-challenge.org/)






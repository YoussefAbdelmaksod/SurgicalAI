# Model Weights

This directory is used to store model weights for the SurgicalAI project. To keep the repository lightweight, the actual model weight files (*.pth) are not stored in the repository.

## Pre-trained Models

The following pre-trained models are used by SurgicalAI:

1. **Phase Recognition Model** (`phase_recognition.pth`)
   - ViT-LSTM model with ResNet50 backbone
   - Expected size: ~102MB

2. **Tool Detection Model** (`tool_detection.pth`)
   - Faster R-CNN with ResNet50 backbone
   - Expected size: ~167MB

3. **Mistake Detection Model** (`mistake_detection.pth`)
   - Custom model with visual-temporal features
   - Expected size: ~96MB

## How to Obtain Model Weights

You can download the necessary model weights using one of the following methods:

### Option 1: Using the Download Script

Run the provided script:
```
python scripts/download_weights.py
```

### Option 2: Manual Download

1. Download the model weights from the following links:
   - Phase Recognition: [https://example.com/models/phase_recognition.pth](https://example.com/models/phase_recognition.pth)
   - Tool Detection: [https://example.com/models/tool_detection.pth](https://example.com/models/tool_detection.pth)
   - Mistake Detection: [https://example.com/models/mistake_detection.pth](https://example.com/models/mistake_detection.pth)

2. Place the downloaded files in this directory.

### Option 3: Train Your Own Models

You can train your own models using the training scripts provided in the `training/` directory:

```
python training/train_tool_detection.py --data_dir data --output_dir models/weights
```

## Directory Structure

The weights directory should have the following structure after downloading or training:

```
weights/
├── README.md                # This file
├── phase_recognition.pth    # Phase recognition model
├── tool_detection.pth       # Tool detection model
└── mistake_detection.pth    # Mistake detection model
```

## Note

For development and testing without full model weights, the system can run in "demo mode" with limited functionality. 
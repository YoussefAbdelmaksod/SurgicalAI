# SurgicalAI

SurgicalAI is a comprehensive AI system for surgical video analysis with real-time feedback and guidance. It combines computer vision and deep learning techniques to assist surgeons during laparoscopic procedures.

![SurgicalAI System](https://example.com/surgical_ai_image.png)

## Features

- **Surgical Phase Recognition**: Automatically identifies surgical phases using Vision Transformer (ViT) with LSTM temporal processing
- **Surgical Tool Detection**: Detects surgical instruments with Faster R-CNN and Feature Pyramid Networks
- **Mistake Detection**: Identifies potential surgical mistakes and provides risk assessment
- **Guidance Generation**: Offers real-time guidance based on detected phases, tools, and mistakes
- **Web Interface**: User-friendly web application for video processing and visualization

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- CUDA-compatible GPU (recommended for real-time processing)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SurgicalAI.git
   cd SurgicalAI
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Generate model weights:
   ```bash
   python scripts/download_weights.py
   ```

## Usage

### Running the Web Application

The simplest way to use SurgicalAI is through its web interface:

```bash
python app/main.py --mode web --host localhost --port 5000
```

This starts the web server at http://localhost:5000 where you can:
- Upload surgical videos for analysis
- View real-time analysis results
- Access visualizations of detected phases, tools, and mistakes

### Processing Videos via Command Line

You can also process videos directly from the command line:

```bash
python app/main.py --mode video --input path/to/surgery.mp4 --output results.mp4
```

### Testing the System

To verify that everything is working correctly:

```bash
python scripts/test_system.py --video data/videos/sample.mp4 --output test_output.mp4
```

## Training Models

SurgicalAI comes with pre-initialized model weights, but you can train them on your own data:

### Preparing Training Data

1. Place your surgical videos in `data/videos/`
2. Run the data preparation script:
   ```bash
   python scripts/preprocess_data.py --data_dir data --output_dir data --force
   ```

### Training Individual Models

Train the tool detection model:
```bash
python training/train_all_models.py --train_subset tool_detection
```

Train the phase recognition model:
```bash
python training/train_phase_recognition.py --data_dir data --output_dir models/weights/vit_lstm
```

Train the mistake detection model:
```bash
python training/train_all_models.py --train_subset mistake_detection
```

### Training All Models

To train all models at once:
```bash
python training/train_all_models.py
```

## Project Structure

```
SurgicalAI/
├── app/                # Web application
├── config/             # Configuration files
├── data/               # Dataset and preprocessing
│   ├── train/          # Training data
│   ├── valid/          # Validation data
│   └── videos/         # Surgical videos
├── models/             # Model implementations
│   ├── phase_recognition.py
│   ├── tool_detection.py
│   ├── mistake_detection.py
│   └── weights/        # Model weights
├── scripts/            # Utility scripts
├── training/           # Training implementations
├── utils/              # Shared utility functions
└── requirements.txt    # Dependencies
```

## Model Architecture Details

### Phase Recognition
- **Architecture**: Vision Transformer (ViT-B/16) with Bidirectional LSTM
- **Training Data**: Sequences of video frames with phase annotations
- **Output**: Classification of 7 surgical phases

### Tool Detection
- **Architecture**: Faster R-CNN with ResNet50 backbone
- **Training Data**: Annotated images with bounding boxes around surgical tools
- **Output**: Bounding boxes, class labels, and confidence scores for detected tools

### Mistake Detection
- **Architecture**: Multi-modal fusion of visual features and tool detections
- **Training Data**: Video segments with mistake annotations
- **Output**: Mistake classification and risk assessment

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The SurgicalAI project was developed with guidance from medical professionals
- Many of the surgical videos come from open-access surgical education resources
- The implementation leverages several open-source deep learning frameworks and models






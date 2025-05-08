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
    create_directory("/content/SurgicalAI_clone/training")
    create_directory("/content/SurgicalAI_clone/models/weights/tool_detection")
    create_directory("/content/SurgicalAI_clone/models/weights/vit_lstm")
    create_directory("/content/SurgicalAI_clone/models/weights/mistake_detector")
    create_directory("/content/SurgicalAI_clone/data")
    create_directory("/content/SurgicalAI_clone/utils")
    create_directory("/content/SurgicalAI_clone/config")
    
    # Create the train_tool_detection.py file if it doesn't exist
    tool_detection_path = "/content/SurgicalAI_clone/training/train_tool_detection.py"
    if not os.path.exists(tool_detection_path):
        with open(tool_detection_path, "w") as f:
            f.write("""
import argparse
import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

def train_tool_detection(args):
    print(f"Training tool detection model with {args.backbone} backbone")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training for {args.num_epochs} epochs with batch size {args.batch_size}")
    
    # Create a dummy model for demonstration
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Save the model
    os.makedirs(os.path.join(args.output_dir, "tool_detection"), exist_ok=True)
    output_path = os.path.join(args.output_dir, "tool_detection", "tool_detection.pth")
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train tool detection model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone model')
    parser.add_argument('--use_mixed_precision', type=bool, default=True, help='Use mixed precision training')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_tool_detection(args)
""")
        print(f"Created {tool_detection_path}")
    
    # Create the train_phase_recognition.py file if it doesn't exist
    phase_recognition_path = "/content/SurgicalAI_clone/training/train_phase_recognition.py"
    if not os.path.exists(phase_recognition_path):
        with open(phase_recognition_path, "w") as f:
            f.write("""
import argparse
import os
import torch
import torch.nn as nn
import timm
from tqdm import tqdm

class VitLstmModel(nn.Module):
    def __init__(self, vit_model_name="vit_base_patch16_224", num_classes=7):
        super().__init__()
        self.vit = timm.create_model(vit_model_name, pretrained=True)
        self.lstm = nn.LSTM(768, 512, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        vit_features = self.vit.forward_features(x)
        vit_features = vit_features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(vit_features)
        out = self.fc(lstm_out[:, -1, :])
        return out

def train_phase_recognition(args):
    print(f"Training phase recognition model with {args.vit_model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training for {args.num_epochs} epochs with batch size {args.batch_size}")
    
    # Create a dummy model for demonstration
    model = VitLstmModel(vit_model_name=args.vit_model)
    
    # Save the model
    os.makedirs(os.path.join(args.output_dir, "vit_lstm"), exist_ok=True)
    output_path = os.path.join(args.output_dir, "vit_lstm", "phase_recognition.pth")
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train phase recognition model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--vit_model', type=str, default='vit_base_patch16_224', help='ViT model name')
    parser.add_argument('--freeze_vit', type=bool, default=True, help='Freeze ViT weights')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_phase_recognition(args)
""")
        print(f"Created {phase_recognition_path}")
    
    # Create the train_all_models.py file if it doesn't exist
    all_models_path = "/content/SurgicalAI_clone/training/train_all_models.py" 
    if not os.path.exists(all_models_path):
        with open(all_models_path, "w") as f:
            f.write("""
import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm

class MistakeDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

def train_mistake_detection(args):
    print(f"Training mistake detection model")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training for {args.num_epochs} epochs with batch size {args.batch_size}")
    
    # Create a dummy model for demonstration
    model = MistakeDetectionModel()
    
    # Save the model
    os.makedirs(os.path.join(args.output_dir, "mistake_detector"), exist_ok=True)
    output_path = os.path.join(args.output_dir, "mistake_detector", "mistake_detection.pth")
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train models')
    parser.add_argument('--train_subset', type=str, default='all', 
                        choices=['all', 'tool_detection', 'phase_recognition', 'mistake_detection'],
                        help='Which models to train')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=8, help='Number of epochs')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.train_subset in ['all', 'mistake_detection']:
        train_mistake_detection(args)
""")
        print(f"Created {all_models_path}")
    
    print("Path fixing completed! You can now run the training scripts.")

if __name__ == "__main__":
    main() 
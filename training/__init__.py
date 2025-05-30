"""
SurgicalAI training module.

This module contains all the training pipelines for the SurgicalAI system.
"""

from training.surgical_datasets import (
    PhaseRecognitionDataset, 
    ToolDetectionDataset, 
    MistakeDetectionDataset, 
    get_dataloader
)

from training.phase_recognition_trainer import PhaseRecognitionTrainer
from training.tool_detection_trainer import ToolDetectionTrainer
from training.mistake_detection_trainer import MistakeDetectionTrainer
from training.train import main as train_main 
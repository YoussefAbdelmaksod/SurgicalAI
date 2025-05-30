#!/bin/bash
# SurgicalAI Training Script
# This script runs the SurgicalAI training pipeline

# Activate virtual environment if needed
# source venv/bin/activate

# Set CUDA device if needed
# export CUDA_VISIBLE_DEVICES=0

# Create log directory
mkdir -p training/logs

# Run training for all models
echo "Starting SurgicalAI training pipeline..."
python3 training/train.py --config training/configs/training_config.yaml --models all

# To train individual models, uncomment the appropriate lines below:
# python3 training/train.py --config training/configs/training_config.yaml --models phase
# python3 training/train.py --config training/configs/training_config.yaml --models tool
# python3 training/train.py --config training/configs/training_config.yaml --models mistake

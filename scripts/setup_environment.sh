#!/bin/bash
# SurgicalAI Environment Setup Script
# This script sets up the Python environment for SurgicalAI

set -e  # Exit on error

# Print with colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== SurgicalAI Environment Setup ===${NC}"

# Check if Python 3 is installed
if command -v python3 &>/dev/null; then
    python_version=$(python3 --version)
    echo -e "${GREEN}✓ Found $python_version${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.8 or later.${NC}"
    exit 1
fi

# Check for pip
if command -v pip3 &>/dev/null; then
    pip_version=$(pip3 --version)
    echo -e "${GREEN}✓ Found pip: $pip_version${NC}"
else
    echo -e "${RED}✗ pip3 not found. Please install pip for Python 3.${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping creation.${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Created virtual environment in 'venv' directory${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✓ Upgraded pip${NC}"

echo -e "\n${YELLOW}Installing core requirements...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Installed core requirements${NC}"

# Optional: Install PyTorch with CUDA if available
if [[ "$*" == *--with-cuda* ]]; then
    echo -e "\n${YELLOW}Installing PyTorch with CUDA support...${NC}"
    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo -e "${GREEN}✓ Installed PyTorch with CUDA support${NC}"
fi

# Create necessary directories
echo -e "\n${YELLOW}Creating necessary directories...${NC}"
mkdir -p data/{Cholec80.v5-cholec80-10-2.coco,endoscapes,m2cai16-tool-locations,videos}
mkdir -p training/{checkpoints,logs}/{phase_recognition,tool_detection,mistake_detection}
echo -e "${GREEN}✓ Created directories${NC}"

# Run verification script
echo -e "\n${YELLOW}Verifying installation...${NC}"
python3 scripts/check_training_pipeline.py

echo -e "\n${GREEN}=== SurgicalAI Environment Setup Complete ===${NC}"
echo -e "${YELLOW}To activate the environment in the future, run:${NC}"
echo -e "source venv/bin/activate"
echo -e "\n${YELLOW}To train models, run:${NC}"
echo -e "python scripts/train_models.py --models all"
echo -e "\n${YELLOW}To run inference, run:${NC}"
echo -e "python scripts/run_inference.py --video data/videos/your_video.mp4" 
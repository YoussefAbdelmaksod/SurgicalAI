FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

LABEL maintainer="SurgicalAI Team"
LABEL description="SurgicalAI - A deep learning system for laparoscopic cholecystectomy guidance"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-tk \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip3 install -r requirements.txt

# Copy the entire project
COPY . .

# Create directories for data, checkpoints, and logs
RUN mkdir -p data/Cholec80.v5-cholec80-10-2.coco \
    data/m2cai16-tool-locations \
    data/endoscapes \
    data/EndoSurgical \
    training/checkpoints \
    training/logs \
    evaluation/results

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Entry point script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python3", "app/main.py"] 
#!/bin/bash
set -e

# Function to check if datasets are mounted
check_datasets() {
    local required_datasets=("Cholec80.v5-cholec80-10-2.coco" "m2cai16-tool-locations" "endoscapes")
    local missing_datasets=()
    
    for dataset in "${required_datasets[@]}"; do
        if [ ! -d "/app/data/$dataset" ] || [ -z "$(ls -A /app/data/$dataset 2>/dev/null)" ]; then
            missing_datasets+=("$dataset")
        fi
    done
    
    if [ ${#missing_datasets[@]} -gt 0 ]; then
        echo "WARNING: The following datasets are missing or empty:"
        for dataset in "${missing_datasets[@]}"; do
            echo "  - $dataset"
        done
        echo ""
        echo "For training, please mount these datasets to /app/data/<dataset_name>"
        echo "Example: docker run -v /path/to/Cholec80:/app/data/Cholec80.v5-cholec80-10-2.coco ..."
        echo ""
    else
        echo "All required datasets are present."
    fi
}

# Check if we're running in training mode
if [ "$1" = "train" ]; then
    shift
    check_datasets
    
    if [ "$1" = "all" ]; then
        echo "Starting training of all models..."
        exec python3 scripts/train_models.py --models all
    elif [ "$1" = "phase" ] || [ "$1" = "tool" ] || [ "$1" = "mistake" ]; then
        echo "Starting training of $1 model..."
        exec python3 scripts/train_models.py --models "$1"
    else
        echo "Unknown training mode: $1"
        echo "Available modes: all, phase, tool, mistake"
        exit 1
    fi
# Check if we're running in evaluation mode
elif [ "$1" = "evaluate" ]; then
    shift
    check_datasets
    
    if [ "$1" = "all" ]; then
        echo "Starting evaluation of all models..."
        exec python3 scripts/evaluate_models.py --models all
    elif [ "$1" = "phase" ] || [ "$1" = "tool" ] || [ "$1" = "mistake" ]; then
        echo "Starting evaluation of $1 model..."
        exec python3 scripts/evaluate_models.py --models "$1"
    else
        echo "Unknown evaluation mode: $1"
        echo "Available modes: all, phase, tool, mistake"
        exit 1
    fi
# Check if we're running in inference mode
elif [ "$1" = "inference" ]; then
    shift
    
    if [ -z "$1" ]; then
        echo "No video specified for inference."
        echo "Usage: docker run ... inference <path_to_video>"
        exit 1
    fi
    
    echo "Running inference on video: $1"
    exec python3 scripts/run_inference.py --video "$1"
# Pass any other command to the container
elif [ "$1" = "bash" ] || [ "$1" = "python3" ] || [ "$1" = "sh" ]; then
    exec "$@"
else
    # Default: run the application
    check_datasets
    echo "Starting SurgicalAI application..."
    exec "$@"
fi 
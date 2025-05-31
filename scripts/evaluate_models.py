#!/usr/bin/env python
"""
Script for evaluating SurgicalAI models.

This script provides evaluation metrics for the trained models in the SurgicalAI system.
It supports individual model evaluation as well as end-to-end system evaluation.
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phase_recognition import PhaseRecognitionModel
from models.tool_detection import ToolDetectionModel
from models.mistake_detection import MistakeDetectionModel
from training.surgical_datasets import PhaseRecognitionDataset, ToolDetectionDataset, MistakeDetectionDataset
from utils.helpers import setup_logging, load_config

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate SurgicalAI models')
    
    parser.add_argument('--config', type=str, default='training/configs/training_config.yaml',
                      help='Path to training configuration file')
    
    parser.add_argument('--models', type=str, nargs='+', 
                      choices=['phase', 'tool', 'mistake', 'all'],
                      default=['all'],
                      help='Models to evaluate: phase, tool, mistake, or all')
    
    parser.add_argument('--model-paths', type=str, default='training/checkpoints/model_paths.yaml',
                      help='Path to YAML file containing model checkpoint paths')
    
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Root directory for datasets')
    
    parser.add_argument('--output-dir', type=str, default='evaluation/results',
                      help='Directory to save evaluation results')
    
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Batch size for evaluation')
    
    parser.add_argument('--no-cuda', action='store_true',
                      help='Disable CUDA even if available')
    
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of data loading workers')
    
    parser.add_argument('--split', type=str, default='test',
                      choices=['train', 'val', 'test'],
                      help='Dataset split to evaluate on')
    
    parser.add_argument('--visualize', action='store_true',
                      help='Generate visualizations of model predictions')
    
    return parser.parse_args()

def create_directories(args):
    """Create necessary directories for evaluation results."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model-specific directories
    os.makedirs(os.path.join(args.output_dir, 'phase_recognition'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'tool_detection'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'mistake_detection'), exist_ok=True)
    
    # Create timestamp directory for results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(args.output_dir, f'eval_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    return results_dir

def evaluate_phase_recognition(config, model_path, args, results_dir):
    """
    Evaluate the phase recognition model.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the model checkpoint
        args: Command-line arguments
        results_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating phase recognition model: {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found: {model_path}")
        return {"error": "Model checkpoint not found"}
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset_config = config['phase_recognition']['data']
    dataset = PhaseRecognitionDataset(
        data_dir=os.path.join(args.data_dir, dataset_config['dataset']),
        split=args.split,
        sequence_length=config['phase_recognition']['training']['sequence_length'],
        overlap=0,  # No overlap for evaluation
        img_size=(224, 224)
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Load model
    model_config = config['phase_recognition']['model']
    model = PhaseRecognitionModel(
        vit_model=model_config['vit_model'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        use_temporal_attention=model_config['use_temporal_attention'],
        num_classes=model_config['num_classes']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Evaluation
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating phase recognition"):
            frames, labels = batch['frames'].to(device), batch['phases'].to(device)
            
            # Forward pass
            outputs = model(frames)
            
            # Convert to predictions
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Some models return (outputs, features)
            
            preds = torch.argmax(outputs, dim=1)
            
            # Store for metrics
            all_labels.extend(labels.view(-1).cpu().numpy())
            all_preds.extend(preds.view(-1).cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Save results
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": conf_matrix.tolist()
    }
    
    with open(os.path.join(results_dir, 'phase_recognition_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Phase recognition evaluation results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    return metrics

def evaluate_tool_detection(config, model_path, args, results_dir):
    """
    Evaluate the tool detection model.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the model checkpoint
        args: Command-line arguments
        results_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating tool detection model: {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found: {model_path}")
        return {"error": "Model checkpoint not found"}
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset_config = config['tool_detection']['data']
    dataset = ToolDetectionDataset(
        data_dir=os.path.join(args.data_dir, dataset_config['dataset']),
        split=args.split,
        img_size=(512, 512),
        min_visibility=dataset_config['min_visibility'],
        min_size=dataset_config['min_size']
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda x: x  # Tool detection needs custom collation
    )
    
    # Load model
    model_config = config['tool_detection']['model']
    model = ToolDetectionModel(
        backbone=model_config['backbone'],
        pretrained=False,  # No need for pretrained during evaluation
        num_classes=model_config['num_classes'],
        use_fpn=model_config['use_fpn']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Metrics for object detection
    all_detections = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating tool detection"):
            for item in batch:
                image = item['image'].unsqueeze(0).to(device)
                target = item['boxes']
                
                # Forward pass
                detections = model(image)
                
                # Convert to CPU for evaluation
                detections = [{k: v.cpu() for k, v in det.items()} for det in detections]
                
                all_detections.append(detections[0])
                all_targets.append(target)
    
    # Calculate mAP using COCO evaluator
    from utils.coco_eval import CocoEvaluator
    evaluator = CocoEvaluator(dataset)
    evaluator.update(all_detections, all_targets)
    results = evaluator.compute()
    
    # Save results
    metrics = {
        "mAP": float(results["mAP"]),
        "mAP_50": float(results["mAP_50"]),
        "mAP_75": float(results["mAP_75"]),
        "mAP_small": float(results["mAP_small"]),
        "mAP_medium": float(results["mAP_medium"]),
        "mAP_large": float(results["mAP_large"]),
        "AR_1": float(results["AR_1"]),
        "AR_10": float(results["AR_10"]),
        "AR_100": float(results["AR_100"])
    }
    
    with open(os.path.join(results_dir, 'tool_detection_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Tool detection evaluation results:")
    logger.info(f"  mAP: {results['mAP']:.4f}")
    logger.info(f"  mAP_50: {results['mAP_50']:.4f}")
    logger.info(f"  mAP_75: {results['mAP_75']:.4f}")
    
    return metrics

def evaluate_mistake_detection(config, model_path, args, results_dir):
    """
    Evaluate the mistake detection model.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the model checkpoint
        args: Command-line arguments
        results_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating mistake detection model: {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found: {model_path}")
        return {"error": "Model checkpoint not found"}
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset_config = config['mistake_detection']['data']
    dataset = MistakeDetectionDataset(
        data_dir=os.path.join(args.data_dir, dataset_config['dataset']),
        split=args.split,
        sequence_length=5,
        use_synthetic=False,  # No synthetic data for evaluation
        supplementary_data_dir=os.path.join(args.data_dir, dataset_config['supplementary_data_dir'])
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Load model
    model_config = config['mistake_detection']['model']
    model = MistakeDetectionModel(
        visual_dim=model_config['visual_dim'],
        tool_dim=model_config['tool_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_classes=model_config['num_classes'],
        use_temporal=model_config['use_temporal']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Evaluation
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating mistake detection"):
            frames = batch['frames'].to(device)
            tool_features = batch['tool_features'].to(device) if 'tool_features' in batch else None
            labels = batch['mistake'].to(device)
            
            # Forward pass
            outputs = model(frames, tool_features)
            
            # Convert to predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Store for metrics
            all_labels.extend(labels.view(-1).cpu().numpy())
            all_preds.extend(preds.view(-1).cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Save results
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": conf_matrix.tolist()
    }
    
    with open(os.path.join(results_dir, 'mistake_detection_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Mistake detection evaluation results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    return metrics

def evaluate_end_to_end(config, model_paths, args, results_dir):
    """
    Evaluate the entire SurgicalAI system end-to-end.
    
    Args:
        config: Configuration dictionary
        model_paths: Dictionary of model checkpoint paths
        args: Command-line arguments
        results_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("Evaluating end-to-end system performance")
    
    # This would involve running the entire pipeline on test videos
    # and evaluating the overall system performance
    
    # For now, we'll leave this as a placeholder
    # Actual implementation would depend on specific end-to-end metrics
    
    metrics = {
        "note": "End-to-end evaluation is application-specific and would need to be customized"
    }
    
    with open(os.path.join(results_dir, 'end_to_end_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def main():
    """Main function to run evaluation."""
    # Parse command-line arguments
    args = parse_args()
    
    # Create necessary directories
    results_dir = create_directories(args)
    
    # Set up logging
    setup_logging(log_dir=results_dir)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model paths
    if os.path.exists(args.model_paths):
        with open(args.model_paths, 'r') as f:
            model_paths = yaml.safe_load(f)
    else:
        logger.error(f"Model paths file not found: {args.model_paths}")
        model_paths = {
            'phase_recognition': '',
            'tool_detection': '',
            'mistake_detection': ''
        }
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available() and not args.no_cuda
    logger.info(f"CUDA available: {cuda_available}")
    if cuda_available:
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if cuda_available:
        torch.cuda.manual_seed(args.seed)
    
    # Determine which models to evaluate
    models_to_evaluate = args.models
    if 'all' in models_to_evaluate:
        models_to_evaluate = ['phase', 'tool', 'mistake']
    
    # Evaluation results
    results = {}
    
    # Evaluate phase recognition model
    if 'phase' in models_to_evaluate and model_paths.get('phase_recognition'):
        results['phase'] = evaluate_phase_recognition(
            config, model_paths['phase_recognition'], args, results_dir
        )
    
    # Evaluate tool detection model
    if 'tool' in models_to_evaluate and model_paths.get('tool_detection'):
        results['tool'] = evaluate_tool_detection(
            config, model_paths['tool_detection'], args, results_dir
        )
    
    # Evaluate mistake detection model
    if 'mistake' in models_to_evaluate and model_paths.get('mistake_detection'):
        results['mistake'] = evaluate_mistake_detection(
            config, model_paths['mistake_detection'], args, results_dir
        )
    
    # Evaluate end-to-end system
    if len(models_to_evaluate) > 1:
        results['end_to_end'] = evaluate_end_to_end(
            config, model_paths, args, results_dir
        )
    
    # Print summary of evaluated models
    logger.info("Evaluation completed. Summary of evaluated models:")
    for model_name, model_result in results.items():
        if isinstance(model_result, dict) and 'error' not in model_result:
            metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in model_result.items() 
                                   if isinstance(v, (int, float)) and not isinstance(v, bool)])
            logger.info(f"  {model_name}: {metrics_str}")
        else:
            logger.info(f"  {model_name}: {model_result}")
    
    # Save summary results
    with open(os.path.join(results_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Evaluation results saved to {results_dir}")
    
    return results

if __name__ == '__main__':
    main() 
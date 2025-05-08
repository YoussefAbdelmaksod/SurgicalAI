#!/usr/bin/env python3
"""
Evaluation script for SurgicalAI models.

This script loads trained models and evaluates their performance on test data,
generating comprehensive metrics and visualizations.
"""

import os
import sys
import argparse
import logging
import yaml
import json
import time
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import project modules
from models.tool_detection import AdvancedToolDetectionModel, ToolDetectionEnsemble
from models.phase_recognition import ViTLSTM, ViTTransformerTemporal
from models.mistake_detection import SurgicalMistakeDetector
from data.coco_dataset import SurgicalToolDataset
from utils.evaluation import evaluate_tool_detection
from utils.helpers import load_config, setup_logging, get_device
from utils.visualization import (
    draw_bounding_boxes, 
    plot_confusion_matrix, 
    plot_precision_recall_curve
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SurgicalAI models")
    
    # Required arguments
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["tool_detection", "phase_recognition", "mistake_detection"],
                        help="Type of model to evaluate")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the test data")
    
    # Configuration and output options
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save evaluation results")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save logs")
    
    # Evaluation options
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of model predictions")
    parser.add_argument("--num_vis_samples", type=int, default=10,
                        help="Number of samples to visualize")
    
    # Device options
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, cpu, or specific GPU)")
    
    # Additional options
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save model predictions to file")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "valid", "test"],
                        help="Dataset split to evaluate on")
    
    return parser.parse_args()


def get_model(model_type, config, device, checkpoint_path):
    """Create model and load weights."""
    # First, create the model architecture
    if model_type == "tool_detection":
        model_config = config["model"]["tool_detection"]
        
        if model_config.get("use_ensemble", False):
            # Create ensemble of models
            models = []
            for i, arch in enumerate(model_config.get("ensemble_architectures", ["faster_rcnn"])):
                for j, backbone in enumerate(model_config.get("ensemble_backbones", ["resnet50"])):
                    model = AdvancedToolDetectionModel(
                        num_classes=model_config["num_classes"],
                        architecture=arch,
                        backbone_name=backbone,
                        pretrained=False,  # We'll load weights
                        use_fpn=model_config["use_fpn"],
                        score_threshold=model_config["score_threshold"],
                        nms_threshold=model_config["nms_threshold"]
                    )
                    models.append(model)
            
            # Create ensemble
            weights = model_config.get("ensemble_weights", None)
            model = ToolDetectionEnsemble(
                models=models,
                ensemble_method=model_config["ensemble_method"],
                weights=weights,
                nms_threshold=model_config["nms_threshold"]
            )
        else:
            # Create single model
            model = AdvancedToolDetectionModel(
                num_classes=model_config["num_classes"],
                architecture=model_config["architecture"],
                backbone_name=model_config["backbone"],
                pretrained=False,  # We'll load weights
                use_fpn=model_config["use_fpn"],
                score_threshold=model_config["score_threshold"],
                nms_threshold=model_config["nms_threshold"]
            )
    
    elif model_type == "phase_recognition":
        model_config = config["model"]["phase_recognition"]
        
        if model_config["model_type"] == "vit_lstm":
            model = ViTLSTM(
                num_classes=model_config["num_classes"],
                hidden_size=model_config["hidden_size"],
                num_layers=model_config["num_layers"],
                dropout=0.0,  # No dropout during evaluation
                bidirectional=model_config.get("bidirectional", True),
                use_temporal_attention=model_config["use_temporal_attention"],
                freeze_vit=True,  # Freezing irrelevant for evaluation
                pretrained=False  # We'll load weights
            )
        else:  # vit_transformer
            model = ViTTransformerTemporal(
                num_classes=model_config["num_classes"],
                hidden_size=model_config["hidden_size"],
                num_layers=model_config["num_layers"],
                num_heads=model_config["num_heads"],
                dropout=0.0,  # No dropout during evaluation
                freeze_vit=True,  # Freezing irrelevant for evaluation
                pretrained=False  # We'll load weights
            )
    
    elif model_type == "mistake_detection":
        model_config = config["model"]["mistake_detection"]
        
        model = SurgicalMistakeDetector(
            visual_dim=model_config["visual_dim"],
            tool_dim=model_config["tool_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_classes=model_config["num_classes"],
            use_temporal=model_config["use_temporal"],
            dropout=0.0  # No dropout during evaluation
        )
    
    # Load weights from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model


def get_dataset(model_type, data_dir, split, config):
    """Get dataset for evaluation."""
    # Extract dataset config
    dataset_config = config["training"]["dataset"]
    
    if model_type == "tool_detection":
        # Tool detection dataset (COCO format)
        dataset = SurgicalToolDataset(
            root=os.path.join(data_dir, split),
            transforms=True,
            augmentation_level="none",  # No augmentation for evaluation
            image_size=dataset_config.get("image_size", 800)
        )
    
    elif model_type == "phase_recognition":
        # Phase recognition dataset
        from data.dataset import VideoFrameDataset
        
        dataset = VideoFrameDataset(
            root=os.path.join(data_dir, split),
            sequence_length=dataset_config.get("sequence_length", 16),
            temporal_stride=dataset_config.get("temporal_stride", 2),
            transforms=True,
            use_augmentation=False,  # No augmentation for evaluation
            image_size=dataset_config.get("image_size", 224)
        )
    
    elif model_type == "mistake_detection":
        # Mistake detection dataset
        from data.dataset import MistakeDetectionDataset
        
        dataset = MistakeDetectionDataset(
            root=os.path.join(data_dir, split),
            sequence_length=dataset_config.get("sequence_length", 16),
            temporal_stride=dataset_config.get("temporal_stride", 2),
            transforms=True,
            use_augmentation=False,  # No augmentation for evaluation
            image_size=dataset_config.get("image_size", 224)
        )
    
    return dataset


def evaluate_tool_detection_model(model, dataset, device, config, output_dir, visualize=False, num_vis_samples=10,
                                 save_predictions=False, logger=None):
    """Evaluate tool detection model."""
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Process one image at a time for evaluation
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Get class names
    class_names = dataset.get_class_names()
    
    # Initialize metrics
    all_predictions = []
    all_targets = []
    
    # Process each image
    logger.info("Evaluating tool detection model...")
    for i, (images, targets) in enumerate(tqdm(data_loader)):
        # Move to device
        images = [img.to(device) for img in images]
        
        # Get predictions
        with torch.no_grad():
            predictions = model(images)
        
        # Store predictions and targets for later evaluation
        all_predictions.extend(predictions)
        all_targets.extend(targets)
        
        # Visualize some samples
        if visualize and i < num_vis_samples:
            # Convert to numpy for visualization
            img_np = images[0].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            # Draw predictions
            pred_boxes = predictions[0]["boxes"].cpu().numpy()
            pred_labels = predictions[0]["labels"].cpu().numpy()
            pred_scores = predictions[0]["scores"].cpu().numpy()
            
            img_with_preds = draw_bounding_boxes(
                img_np,
                pred_boxes,
                pred_labels,
                pred_scores,
                class_names=class_names
            )
            
            # Draw ground truth
            target_boxes = targets[0]["boxes"].cpu().numpy()
            target_labels = targets[0]["labels"].cpu().numpy()
            
            img_with_targets = draw_bounding_boxes(
                img_np,
                target_boxes,
                target_labels,
                class_names=class_names
            )
            
            # Create comparison figure
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(img_with_targets)
            plt.title("Ground Truth")
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            plt.imshow(img_with_preds)
            plt.title("Predictions")
            plt.axis("off")
            
            # Save figure
            os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
            plt.savefig(os.path.join(output_dir, "visualizations", f"sample_{i}.png"))
            plt.close()
    
    # Compute metrics
    metrics = evaluate_tool_detection(all_predictions, all_targets, class_names)
    
    # Print results
    logger.info(f"Tool detection evaluation results:")
    logger.info(f"mAP@0.5 = {metrics['mAP@0.5']:.4f}")
    logger.info(f"mAP@0.75 = {metrics['mAP@0.75']:.4f}")
    logger.info(f"mAP@0.5:0.95 = {metrics['mAP@0.5:0.95']:.4f}")
    
    # Print per-class metrics
    logger.info("Per-class metrics:")
    for class_id, class_name in enumerate(class_names):
        if class_id == 0:  # Skip background class
            continue
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {metrics['per_class_precision'][class_id]:.4f}")
        logger.info(f"    Recall: {metrics['per_class_recall'][class_id]:.4f}")
        logger.info(f"    AP@0.5: {metrics['per_class_AP@0.5'][class_id]:.4f}")
    
    # Save metrics
    with open(os.path.join(output_dir, "tool_detection_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Create precision-recall curves
    plt.figure(figsize=(12, 8))
    for class_id, class_name in enumerate(class_names):
        if class_id == 0:  # Skip background class
            continue
        plt.plot(
            metrics["per_class_recall_curve"][class_id], 
            metrics["per_class_precision_curve"][class_id],
            label=f"{class_name} (AP={metrics['per_class_AP@0.5'][class_id]:.2f})"
        )
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves by Class")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"))
    plt.close()
    
    # Save predictions if requested
    if save_predictions:
        logger.info("Saving predictions...")
        # Convert tensor data to serializable format
        serializable_preds = []
        for pred in all_predictions:
            serializable_pred = {
                "boxes": pred["boxes"].cpu().numpy().tolist(),
                "labels": pred["labels"].cpu().numpy().tolist(),
                "scores": pred["scores"].cpu().numpy().tolist()
            }
            serializable_preds.append(serializable_pred)
        
        with open(os.path.join(output_dir, "tool_detection_predictions.json"), "w") as f:
            json.dump(serializable_preds, f)
    
    return metrics


def evaluate_classification_model(model, dataset, device, config, output_dir, model_type,
                                visualize=False, num_vis_samples=10, save_predictions=False, logger=None):
    """Evaluate classification model (phase recognition or mistake detection)."""
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Get class names (this would be different for each model type)
    if model_type == "phase_recognition":
        class_names = [
            "preparation",
            "calot_triangle_dissection",
            "clipping_and_cutting",
            "gallbladder_dissection",
            "gallbladder_packaging",
            "cleaning_and_coagulation",
            "gallbladder_extraction"
        ]
    else:  # mistake_detection
        class_names = [
            "ok",
            "warning",
            "critical"
        ]
    
    # Initialize storage for predictions and targets
    all_predictions = []
    all_targets = []
    all_scores = []
    
    # Process data
    logger.info(f"Evaluating {model_type} model...")
    for i, batch in enumerate(tqdm(data_loader)):
        # Handle different batch formats
        if isinstance(batch, dict):
            # Extract inputs and targets from dictionary
            inputs = batch["features"].to(device)
            targets = batch["labels"].to(device)
        else:
            # Assume tuple/list format
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(inputs)
        
        # Get predicted classes and scores
        if isinstance(outputs, tuple):
            # Some models might return (logits, features)
            logits = outputs[0]
        else:
            logits = outputs
        
        # Convert logits to probabilities and predicted class
        scores = torch.nn.functional.softmax(logits, dim=1)
        predictions = torch.argmax(scores, dim=1)
        
        # Store results
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_scores.extend(scores.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_scores = np.array(all_scores)
    
    # Compute metrics
    # Classification report with precision, recall, f1
    report = classification_report(
        all_targets, 
        all_predictions, 
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Precision-recall curves (one-vs-rest for each class)
    precisions = {}
    recalls = {}
    average_precisions = {}
    
    for i, class_name in enumerate(class_names):
        # Create binary targets (1 for current class, 0 for others)
        binary_targets = (all_targets == i).astype(int)
        
        # Get scores for current class
        class_scores = all_scores[:, i]
        
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(binary_targets, class_scores)
        
        # Store results
        precisions[class_name] = precision
        recalls[class_name] = recall
        
        # Compute average precision
        ap = average_precision_score(binary_targets, class_scores)
        average_precisions[class_name] = ap
    
    # Consolidate metrics
    metrics = {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "average_precisions": average_precisions
    }
    
    # Print results
    logger.info(f"{model_type.capitalize()} evaluation results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Macro Precision: {metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall: {metrics['macro_recall']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    
    # Print per-class metrics
    logger.info("Per-class metrics:")
    for class_name in class_names:
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {report[class_name]['precision']:.4f}")
        logger.info(f"    Recall: {report[class_name]['recall']:.4f}")
        logger.info(f"    F1-score: {report[class_name]['f1-score']:.4f}")
        logger.info(f"    AP: {average_precisions[class_name]:.4f}")
    
    # Save metrics
    with open(os.path.join(output_dir, f"{model_type}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Visualizations
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(
        cm,
        class_names=class_names,
        normalize=True,
        title=f"{model_type.capitalize()} Confusion Matrix"
    )
    plt.savefig(os.path.join(output_dir, f"{model_type}_confusion_matrix.png"))
    plt.close()
    
    # Plot precision-recall curves
    plt.figure(figsize=(12, 8))
    for class_name in class_names:
        plt.plot(
            recalls[class_name],
            precisions[class_name],
            label=f"{class_name} (AP={average_precisions[class_name]:.2f})"
        )
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_type.capitalize()} Precision-Recall Curves")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_type}_precision_recall_curves.png"))
    plt.close()
    
    # Save predictions if requested
    if save_predictions:
        logger.info("Saving predictions...")
        predictions_data = {
            "predictions": all_predictions.tolist(),
            "targets": all_targets.tolist(),
            "scores": all_scores.tolist()
        }
        
        with open(os.path.join(output_dir, f"{model_type}_predictions.json"), "w") as f:
            json.dump(predictions_data, f)
    
    return metrics


def main():
    """Main evaluation function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"{args.model_type}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logging(config["app"]["log_level"], log_file=log_file)
    
    # Get device
    device = get_device() if args.device is None else args.device
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.model_type, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading {args.model_type} model from {args.checkpoint}...")
    model = get_model(args.model_type, config, device, args.checkpoint)
    
    # Get dataset
    logger.info(f"Loading {args.split} dataset...")
    dataset = get_dataset(args.model_type, args.data_dir, args.split, config)
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Evaluate model
    if args.model_type == "tool_detection":
        metrics = evaluate_tool_detection_model(
            model, dataset, device, config, output_dir,
            visualize=args.visualize,
            num_vis_samples=args.num_vis_samples,
            save_predictions=args.save_predictions,
            logger=logger
        )
    else:
        # Classification models (phase recognition or mistake detection)
        metrics = evaluate_classification_model(
            model, dataset, device, config, output_dir, args.model_type,
            visualize=args.visualize,
            num_vis_samples=args.num_vis_samples,
            save_predictions=args.save_predictions,
            logger=logger
        )
    
    # Save configuration
    with open(os.path.join(output_dir, "evaluation_config.yaml"), "w") as f:
        yaml.dump({
            "args": vars(args),
            "config": config
        }, f)
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
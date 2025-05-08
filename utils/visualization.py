"""
Visualization utilities for SurgicalAI.

This module provides visualization functions for model outputs, training metrics,
and debug information.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import io
import os
from matplotlib.figure import Figure
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Define color palette for different classes
COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (128, 0, 0),      # Maroon
    (0, 128, 0),      # Dark Green
    (0, 0, 128),      # Navy
    (128, 128, 0),    # Olive
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Teal
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (255, 165, 0),    # Orange
]

def draw_bounding_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    labels: List[int],
    scores: Optional[List[float]] = None,
    class_names: Optional[List[str]] = None,
    color_mapping: Optional[Dict[int, Tuple[int, int, int]]] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1
) -> np.ndarray:
    """
    Draw bounding boxes on the image.
    
    Args:
        image: Input image (OpenCV format, BGR)
        boxes: List of bounding boxes in format [x1, y1, x2, y2]
        labels: List of class labels (integers)
        scores: Optional list of confidence scores
        class_names: Optional list of class names for label display
        color_mapping: Optional mapping from class indices to colors
        line_thickness: Thickness of bounding box lines
        font_scale: Scale of font for labels
        font_thickness: Thickness of font for labels
        
    Returns:
        Image with bounding boxes drawn
    """
    img_copy = image.copy()
    
    # Use default color mapping if not provided
    if color_mapping is None:
        color_mapping = {i: COLORS[i % len(COLORS)] for i in range(100)}
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        # Convert box to integer coordinates
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Get color for the current class
        color = color_mapping.get(label, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, line_thickness)
        
        # Prepare label text
        if class_names is not None and 0 <= label < len(class_names):
            label_text = class_names[label]
        else:
            label_text = f"Class {label}"
            
        if scores is not None and i < len(scores):
            label_text += f" {scores[i]:.2f}"
            
        # Draw label background
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(img_copy, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(
            img_copy, label_text, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness
        )
    
    return img_copy

def plot_training_metrics(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Training Metrics"
) -> Optional[plt.Figure]:
    """
    Plot training metrics.
    
    Args:
        metrics: Dictionary of metrics, where keys are metric names and values are lists of values
        save_path: Optional path to save the plot
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure if not saved, otherwise None
    """
    fig, axes = plt.subplots(nrows=(len(metrics) + 1) // 2, ncols=2, figsize=figsize)
    axes = axes.flatten()
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        if i < len(axes):
            ax = axes[i]
            ax.plot(values)
            ax.set_title(f"{metric_name}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name)
            ax.grid(True)
    
    # Hide any unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return None
    
    return fig

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Confusion Matrix",
    normalize: bool = True
) -> Optional[plt.Figure]:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Optional path to save the plot
        figsize: Figure size
        title: Plot title
        normalize: Whether to normalize the confusion matrix
        
    Returns:
        Matplotlib figure if not saved, otherwise None
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return None
    
    return fig

def plot_precision_recall_curve(
    precision: List[float],
    recall: List[float],
    average_precision: float,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Precision-Recall Curve"
) -> Optional[plt.Figure]:
    """
    Plot precision-recall curve.
    
    Args:
        precision: List of precision values
        recall: List of recall values
        average_precision: Average precision score
        save_path: Optional path to save the plot
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure if not saved, otherwise None
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, lw=2, marker='.', markersize=3)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f"{title} (AP: {average_precision:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return None
    
    return fig

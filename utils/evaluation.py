"""
Evaluation metrics and utilities for SurgicalAI.

This module provides functions for evaluating model performance,
computing metrics, and visualizing results.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    average_precision_score,
    roc_auc_score
)

logger = logging.getLogger(__name__)


def compute_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    """
    Compute accuracy score.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted labels
    
    Returns:
        Accuracy score
    """
    return accuracy_score(targets, predictions)


def compute_precision(targets: np.ndarray, predictions: np.ndarray, average: str = 'weighted') -> float:
    """
    Compute precision score.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', 'samples', None)
    
    Returns:
        Precision score
    """
    return precision_score(targets, predictions, average=average, zero_division=0)


def compute_recall(targets: np.ndarray, predictions: np.ndarray, average: str = 'weighted') -> float:
    """
    Compute recall score.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', 'samples', None)
    
    Returns:
        Recall score
    """
    return recall_score(targets, predictions, average=average, zero_division=0)


def compute_f1_score(targets: np.ndarray, predictions: np.ndarray, average: str = 'weighted') -> float:
    """
    Compute F1 score.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted', 'samples', None)
    
    Returns:
        F1 score
    """
    return f1_score(targets, predictions, average=average, zero_division=0)


def compute_confusion_matrix(targets: np.ndarray, predictions: np.ndarray, 
                            normalize: Optional[str] = None) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted labels
        normalize: Normalization method ('true', 'pred', 'all', None)
    
    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(targets, predictions)
    
    if normalize is not None:
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()
    
    return cm


def compute_classification_report(targets: np.ndarray, predictions: np.ndarray,
                                 target_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute detailed classification report.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted labels
        target_names: List of class names
    
    Returns:
        Dictionary with classification metrics
    """
    return classification_report(targets, predictions, target_names=target_names, 
                               output_dict=True, zero_division=0)


def compute_mean_average_precision(targets: np.ndarray, predictions: np.ndarray, 
                                  average: str = 'weighted') -> float:
    """
    Compute mean average precision for multi-class classification.
    
    Args:
        targets: Ground truth labels (one-hot encoded)
        predictions: Predicted probabilities
        average: Averaging method ('micro', 'macro', 'weighted', 'samples', None)
    
    Returns:
        Mean average precision
    """
    # Handle binary classification case
    if targets.shape[1] == 1:
        return average_precision_score(targets, predictions)
    
    # Handle multi-class case
    return average_precision_score(targets, predictions, average=average)


def compute_roc_auc(targets: np.ndarray, predictions: np.ndarray, 
                   average: str = 'weighted', multi_class: str = 'ovr') -> float:
    """
    Compute ROC AUC score for classification.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted probabilities
        average: Averaging method ('micro', 'macro', 'weighted', 'samples', None)
        multi_class: Approach for multi-class classification ('ovr', 'ovo')
    
    Returns:
        ROC AUC score
    """
    try:
        return roc_auc_score(targets, predictions, average=average, multi_class=multi_class)
    except ValueError as e:
        logger.warning(f"Could not compute ROC AUC: {e}")
        return 0.0


def compute_iou(box1: Union[torch.Tensor, np.ndarray], box2: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
    
    Returns:
        IoU score
    """
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().numpy()
    
    # Get coordinates of intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Compute intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Compute IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou


def compute_map_for_object_detection(
    targets: List[Dict[str, torch.Tensor]],
    predictions: List[Dict[str, torch.Tensor]],
    iou_thresholds: Union[List[float], float] = 0.5,
    max_detections: int = 100
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP) for object detection.
    
    Args:
        targets: List of ground truth detections
        predictions: List of predicted detections
        iou_thresholds: IoU threshold(s) for matching
        max_detections: Maximum number of detections per image
    
    Returns:
        Dictionary with mAP scores at different IoU thresholds
    """
    if isinstance(iou_thresholds, float):
        iou_thresholds = [iou_thresholds]
    
    # Initialize counters for each IoU threshold
    ap_sums = {iou_threshold: 0.0 for iou_threshold in iou_thresholds}
    ap_counts = {iou_threshold: 0 for iou_threshold in iou_thresholds}
    
    # Process each image
    for target, prediction in zip(targets, predictions):
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        
        pred_boxes = prediction['boxes']
        pred_scores = prediction['scores']
        pred_labels = prediction['labels']
        
        # Sort predictions by score
        if len(pred_scores) > 0:
            sorted_indices = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[sorted_indices][:max_detections]
            pred_scores = pred_scores[sorted_indices][:max_detections]
            pred_labels = pred_labels[sorted_indices][:max_detections]
        
        # Process each class separately
        for class_id in torch.unique(gt_labels):
            # Get ground truth boxes for this class
            class_gt_indices = (gt_labels == class_id).nonzero(as_tuple=True)[0]
            class_gt_boxes = gt_boxes[class_gt_indices]
            
            # Get predicted boxes for this class
            class_pred_indices = (pred_labels == class_id).nonzero(as_tuple=True)[0]
            class_pred_boxes = pred_boxes[class_pred_indices]
            class_pred_scores = pred_scores[class_pred_indices]
            
            # Skip if no predictions or no ground truth for this class
            if len(class_gt_boxes) == 0 or len(class_pred_boxes) == 0:
                continue
            
            # Compute IoU matrix between all ground truth and predicted boxes
            iou_matrix = torch.zeros(len(class_gt_boxes), len(class_pred_boxes))
            for i, gt_box in enumerate(class_gt_boxes):
                for j, pred_box in enumerate(class_pred_boxes):
                    iou_matrix[i, j] = compute_iou(gt_box, pred_box)
            
            # For each IoU threshold, compute precision and recall
            for iou_threshold in iou_thresholds:
                # Initialize arrays for precision-recall curve
                tp = torch.zeros(len(class_pred_boxes))
                fp = torch.zeros(len(class_pred_boxes))
                gt_matched = torch.zeros(len(class_gt_boxes), dtype=torch.bool)
                
                # Assign predictions to ground truth
                for j in range(len(class_pred_boxes)):
                    # Find best matching ground truth
                    max_iou, max_idx = torch.max(iou_matrix[:, j], dim=0)
                    
                    if max_iou >= iou_threshold and not gt_matched[max_idx]:
                        tp[j] = 1
                        gt_matched[max_idx] = True
                    else:
                        fp[j] = 1
                
                # Compute precision and recall
                cumsum_tp = torch.cumsum(tp, dim=0)
                cumsum_fp = torch.cumsum(fp, dim=0)
                precision = cumsum_tp / (cumsum_tp + cumsum_fp)
                recall = cumsum_tp / len(class_gt_boxes)
                
                # Compute AP using 11-point interpolation
                ap = 0.0
                for r in torch.linspace(0, 1, 11):
                    if torch.sum(recall >= r) == 0:
                        p_interp = 0
                    else:
                        p_interp = torch.max(precision[recall >= r])
                    ap += p_interp / 11
                
                # Add to sum and count
                ap_sums[iou_threshold] += ap.item()
                ap_counts[iou_threshold] += 1
    
    # Compute mAP for each IoU threshold
    results = {}
    for iou_threshold in iou_thresholds:
        if ap_counts[iou_threshold] > 0:
            results[f'mAP@{iou_threshold:.2f}'] = ap_sums[iou_threshold] / ap_counts[iou_threshold]
        else:
            results[f'mAP@{iou_threshold:.2f}'] = 0.0
    
    # Compute mAP across all IoU thresholds
    if sum(ap_counts.values()) > 0:
        results['mAP'] = sum(ap_sums.values()) / sum(ap_counts.values())
    else:
        results['mAP'] = 0.0
    
    return results


def compute_temporal_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    sequence_lengths: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute metrics for sequence/temporal data.
    
    Args:
        targets: Ground truth labels [batch_size, seq_len]
        predictions: Predicted labels [batch_size, seq_len]
        sequence_lengths: Actual sequence lengths [batch_size]
    
    Returns:
        Dictionary with temporal metrics
    """
    batch_size, max_seq_len = targets.shape
    
    # Use maximum sequence length if lengths not provided
    if sequence_lengths is None:
        sequence_lengths = torch.full((batch_size,), max_seq_len, device=targets.device)
    
    # Convert to numpy
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(sequence_lengths, torch.Tensor):
        sequence_lengths = sequence_lengths.cpu().numpy()
    
    # Initialize metrics
    correct_transitions = 0
    total_transitions = 0
    
    # Compute edit distance
    edit_distances = []
    normalized_edit_distances = []
    
    # Process each sequence
    for i in range(batch_size):
        seq_len = sequence_lengths[i]
        target_seq = targets[i, :seq_len]
        pred_seq = predictions[i, :seq_len]
        
        # Count correct transitions
        for j in range(1, seq_len):
            if target_seq[j] != target_seq[j-1]:  # There's a transition
                total_transitions += 1
                if pred_seq[j] == target_seq[j] and pred_seq[j-1] == target_seq[j-1]:
                    correct_transitions += 1
        
        # Compute edit distance
        edit_dist = levenshtein_distance(target_seq, pred_seq)
        edit_distances.append(edit_dist)
        normalized_edit_distances.append(edit_dist / seq_len if seq_len > 0 else 0)
    
    # Compute transition accuracy
    transition_accuracy = correct_transitions / total_transitions if total_transitions > 0 else 0
    
    # Compute average edit distance
    avg_edit_distance = np.mean(edit_distances)
    avg_normalized_edit_distance = np.mean(normalized_edit_distances)
    
    return {
        'transition_accuracy': transition_accuracy,
        'edit_distance': avg_edit_distance,
        'normalized_edit_distance': avg_normalized_edit_distance
    }


def levenshtein_distance(s1: np.ndarray, s2: np.ndarray) -> int:
    """
    Compute Levenshtein (edit) distance between two sequences.
    
    Args:
        s1: First sequence
        s2: Second sequence
    
    Returns:
        Edit distance
    """
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    
    # Fill dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j],      # Deletion
                                  dp[i, j - 1],      # Insertion
                                  dp[i - 1, j - 1])  # Substitution
    
    return dp[m, n]


def evaluate_classification(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = None
) -> Dict[str, Any]:
    """
    Evaluate classification model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to use
        num_classes: Number of classes
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:  # (inputs, targets)
                    inputs, targets = batch
                elif len(batch) == 3:  # (sequences, lengths, targets)
                    sequences, lengths, targets = batch
                    inputs = sequences
                else:
                    raise ValueError(f"Unsupported batch format with {len(batch)} elements")
            else:
                # Single tensor input (assume autoencoder)
                inputs = batch
                targets = batch  # For autoencoder, target is the input
            
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Special handling for sequence data
            if len(batch) == 3 and lengths is not None:
                lengths = lengths.to(device)
                outputs = model(inputs, lengths)
            else:
                outputs = model(inputs)
            
            # Handle different output formats
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]  # Take the main output
            
            # Get predicted class
            _, predicted = outputs.max(1)
            
            # Store results
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            all_probabilities.append(torch.softmax(outputs, dim=1).cpu().numpy())
    
    # Concatenate results
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    all_probabilities = np.concatenate(all_probabilities)
    
    # Calculate metrics
    metrics = {
        'accuracy': compute_accuracy(all_targets, all_predictions),
        'precision': compute_precision(all_targets, all_predictions),
        'recall': compute_recall(all_targets, all_predictions),
        'f1_score': compute_f1_score(all_targets, all_predictions)
    }
    
    # Compute confusion matrix
    cm = compute_confusion_matrix(all_targets, all_predictions)
    metrics['confusion_matrix'] = cm
    
    # Try to compute mAP and ROC AUC (might fail for certain data)
    try:
        # One-hot encode targets
        if num_classes is None:
            num_classes = max(np.max(all_targets), np.max(all_predictions)) + 1
        
        one_hot_targets = np.zeros((len(all_targets), num_classes))
        for i, t in enumerate(all_targets):
            one_hot_targets[i, t] = 1
        
        metrics['mAP'] = compute_mean_average_precision(one_hot_targets, all_probabilities)
        metrics['roc_auc'] = compute_roc_auc(one_hot_targets, all_probabilities)
    except Exception as e:
        logger.warning(f"Could not compute mAP or ROC AUC: {e}")
    
    return metrics


def evaluate_object_detection(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    iou_thresholds: Union[List[float], float] = [0.5, 0.75]
) -> Dict[str, Any]:
    """
    Evaluate object detection model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to use
        iou_thresholds: IoU threshold(s) for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:  # (images, targets)
                    images, targets = batch
                else:
                    # Unsupported batch format
                    raise ValueError(f"Unsupported batch format with {len(batch)} elements")
            else:
                # Single tensor input (not likely for object detection)
                raise ValueError("Expected batch as (images, targets) tuple")
            
            # Move to device
            if isinstance(images, (list, tuple)):
                images = [img.to(device) for img in images]
            else:
                images = images.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Store results
            all_targets.extend(targets)
            all_predictions.extend(predictions)
    
    # Calculate mAP
    mAP_results = compute_map_for_object_detection(
        all_targets, all_predictions, iou_thresholds=iou_thresholds
    )
    
    return mAP_results

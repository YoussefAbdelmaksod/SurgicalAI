"""
COCO evaluation utilities for object detection models.

This module provides utilities for evaluating object detection models
using COCO metrics (mAP, etc.).
"""

import json
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict

class CocoEvaluator:
    """
    COCO evaluator for object detection models.
    
    This class provides utilities for evaluating object detection models
    using COCO metrics (mAP, etc.).
    """
    
    def __init__(self, dataset, iou_types=["bbox"]):
        """
        Initialize the evaluator.
        
        Args:
            dataset: Dataset to evaluate on
            iou_types: IoU types to evaluate (e.g., ["bbox"])
        """
        self.dataset = dataset
        self.iou_types = iou_types
        self.coco_eval = {}
        self.results = {}
        self.img_ids = []
        
        # Create dummy COCO GT
        self.coco_gt = COCO()
        self.coco_gt.dataset = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for i, cat in enumerate(dataset.get_categories()):
            self.coco_gt.dataset['categories'].append({
                'id': i + 1,
                'name': cat,
                'supercategory': 'none'
            })
        
        self.coco_gt.createIndex()
        
        # Initialize results
        for iou_type in iou_types:
            self.coco_eval[iou_type] = None
            self.results[iou_type] = []
    
    def update(self, predictions, targets):
        """
        Update the evaluator with predictions and targets.
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries
        """
        # Format predictions and targets into COCO format
        img_ids = list(range(len(predictions)))
        self.img_ids.extend(img_ids)
        
        results = self._prepare_for_coco_detection(predictions, targets, img_ids)
        
        for iou_type in self.iou_types:
            self.results[iou_type].extend(results)
    
    def _prepare_for_coco_detection(self, predictions, targets, img_ids):
        """
        Convert predictions and targets to COCO format.
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries
            img_ids: List of image IDs
            
        Returns:
            List of COCO-formatted results
        """
        results = []
        
        # Add ground truth
        for img_id, (pred, target) in enumerate(zip(predictions, targets)):
            # Add image
            self.coco_gt.dataset['images'].append({
                'id': img_id,
                'width': target.get('orig_size', [640, 480])[1],
                'height': target.get('orig_size', [640, 480])[0],
                'file_name': f'img_{img_id}.jpg'
            })
            
            # Add annotations
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                labels = target['labels']
                for i, (box, label) in enumerate(zip(boxes, labels)):
                    # Convert to [x, y, width, height] format
                    xmin, ymin, xmax, ymax = box.tolist()
                    width = xmax - xmin
                    height = ymax - ymin
                    
                    self.coco_gt.dataset['annotations'].append({
                        'id': len(self.coco_gt.dataset['annotations']) + 1,
                        'image_id': img_id,
                        'category_id': int(label),
                        'bbox': [xmin, ymin, width, height],
                        'area': width * height,
                        'iscrowd': 0
                    })
            
            # Add predictions
            if 'boxes' in pred and len(pred['boxes']) > 0:
                boxes = pred['boxes']
                scores = pred['scores']
                labels = pred['labels']
                
                for box, score, label in zip(boxes, scores, labels):
                    # Convert to [x, y, width, height] format
                    xmin, ymin, xmax, ymax = box.tolist()
                    width = xmax - xmin
                    height = ymax - ymin
                    
                    results.append({
                        'image_id': img_id,
                        'category_id': int(label),
                        'bbox': [xmin, ymin, width, height],
                        'score': float(score)
                    })
        
        # Create index after adding all ground truth
        self.coco_gt.createIndex()
        
        return results
    
    def compute(self):
        """
        Compute the evaluation metrics.
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Create COCO dt
        coco_dt = COCO()
        coco_dt.dataset = {
            'images': self.coco_gt.dataset['images'],
            'annotations': self.results["bbox"],
            'categories': self.coco_gt.dataset['categories']
        }
        coco_dt.createIndex()
        
        # Create COCO eval
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = self.img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        metrics = {
            "mAP": coco_eval.stats[0],  # AP@[.5:.95]
            "mAP_50": coco_eval.stats[1],  # AP@.50
            "mAP_75": coco_eval.stats[2],  # AP@.75
            "mAP_small": coco_eval.stats[3],  # AP small
            "mAP_medium": coco_eval.stats[4],  # AP medium
            "mAP_large": coco_eval.stats[5],  # AP large
            "AR_1": coco_eval.stats[6],  # AR max=1
            "AR_10": coco_eval.stats[7],  # AR max=10
            "AR_100": coco_eval.stats[8],  # AR max=100
            "AR_small": coco_eval.stats[9],  # AR small
            "AR_medium": coco_eval.stats[10],  # AR medium
            "AR_large": coco_eval.stats[11]  # AR large
        }
        
        return metrics 
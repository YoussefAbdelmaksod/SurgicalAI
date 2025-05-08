"""
Training script for tool detection model.
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tool_detection import ToolDetectionModel, AdvancedToolDetectionModel
from data.coco_dataset import COCOSurgicalToolDataset
from training.trainer import Trainer
from utils.helpers import load_config, setup_logging, get_device
from utils.lr_scheduler import get_scheduler
from data.augmentation import get_object_detection_augmentations, get_validation_transforms


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Tool Detection Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for model checkpoints')
    parser.add_argument('--config', type=str, default='config/training_config.yaml', help='Path to config file')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of tool classes (including background)')
    parser.add_argument('--architecture', type=str, default='faster_rcnn', 
                        choices=['faster_rcnn', 'mask_rcnn', 'retinanet'], help='Model architecture')
    parser.add_argument('--backbone', type=str, default='resnet50', 
                        help='Backbone network (resnet50, resnet101, mobilenet_v3_large)')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Debug mode with fewer images')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    return parser.parse_args()


def collate_fn(batch):
    """
    Custom collate function for object detection.
    This is required for batching variable size images and targets.
    """
    return tuple(zip(*batch))


def get_transforms(training=True, image_size=224, use_advanced=False, level='medium'):
    """Get transforms for image preprocessing."""
    if use_advanced:
        if training:
            return get_object_detection_augmentations(
                image_size=image_size,
                level=level,
                p=0.5
            )
        else:
            return get_validation_transforms(image_size=image_size)
    else:
        # Use basic transforms for backward compatibility
        if training:
            return T.Compose([
                T.RandomHorizontalFlip(0.5),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


def main():
    """Main training function."""
    args = parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting tool detection training...")
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Adjust num_workers for Colab
    if torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0).find('Tesla T4') != -1:
        logger.info("Detected Colab environment with T4 GPU")
        args.num_workers = 2  # Colab-friendly setting
    
    # Load dataset
    train_dataset, train_loader, val_dataset, val_loader = load_datasets(
        args.data_dir, args.batch_size, args.num_workers, args.debug
    )
    
    # Create model
    model = AdvancedToolDetectionModel(
        num_classes=args.num_classes,
        architecture=args.architecture,
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        use_fpn=True
    ).to(device)
    
    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=args.num_epochs,
        pct_start=0.1
    )
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.output_dir, 'tool_detection')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = ToolDetectionTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=10,
        use_mixed_precision=args.use_mixed_precision
    )
    
    # Train model
    logger.info(f"Training for {args.num_epochs} epochs...")
    history = trainer.train(num_epochs=args.num_epochs, resume=args.resume)
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'tool_detection.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Optionally evaluate on test set
    # TODO: Implement test set evaluation
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main() 
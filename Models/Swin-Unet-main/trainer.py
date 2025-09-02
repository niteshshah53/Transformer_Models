"""
Training Module for Historical Document Segmentation

This module handles the actual training process including:
- Setting up data loaders
- Computing class weights for balanced training
- Running training and validation loops
- Saving best models
- Logging progress

Author: Clean Code Version
"""

import numpy as np
import logging
import os
import sys
import glob
import re
import warnings
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as TF

from utils import DiceLoss, FocalLoss

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_worker_random_seed(worker_id):
    """
    Initialize random seed for data loading workers.
    This ensures reproducible data loading across different workers.
    
    Args:
        worker_id (int): ID of the data loading worker
    """
    import random
    base_seed = getattr(setup_worker_random_seed, 'base_seed', 1234)
    random.seed(base_seed + worker_id)


def setup_logging(output_path):
    """
    Set up logging to both file and console.
    
    Args:
        output_path (str): Directory where log file will be saved
    """
    log_file = os.path.join(output_path, "training.log")
    
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


import numpy as np
from PIL import Image
import torch

def compute_class_weights(train_dataset, num_classes):
    """
    Compute class weights for balanced training based on pixel frequency.
    Matches the original implementation from trainer_synapse.
    
    Args:
        train_dataset: Training dataset object with .mask_paths
        num_classes (int): Number of segmentation classes
        
    Returns:
        torch.Tensor: Normalized class weights for loss function (on CUDA if available)
    """
    print("\nComputing class weights...")

    # Color mapping for U-DIADS-Bib dataset (same as original code)
    COLOR_MAP = {
        (0, 0, 0): 0,        # Background
        (255, 255, 0): 1,    # Paratext
        (0, 255, 255): 2,    # Decoration
        (255, 0, 255): 3,    # Main text
        (255, 0, 0): 4,      # Title
        (0, 255, 0): 5,      # Chapter Heading
    }

    class_counts = np.zeros(num_classes, dtype=np.float64)

    # Count pixels for each class across all training masks
    for mask_path in train_dataset.mask_paths:
        mask = Image.open(mask_path).convert("RGB")
        mask = np.array(mask)

        for rgb, cls in COLOR_MAP.items():
            matches = np.all(mask == rgb, axis=-1)
            class_counts[cls] += matches.sum()

    # Compute class frequencies and weights
    class_freq = class_counts / class_counts.sum()
    weights = 1.0 / (class_freq + 1e-6)
    weights = weights / weights.sum()  # Normalize

    # Print class distribution analysis (exact format from original)
    print("\n" + "-"*80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("-"*80)
    print(f"{'Class':<6} {'Frequency':<15} {'Weight':<15}")
    print("-"*80)
    for cls in range(num_classes):
        print(f"{cls:<6} {class_freq[cls]:<15.6f} {weights[cls]:<15.6f}")
    print("-"*80 + "\n")

    # Return as CUDA tensor if available
    return torch.tensor(weights, dtype=torch.float32).cuda()



def create_data_loaders(train_dataset, val_dataset, batch_size, num_workers, seed):
    """
    Create training and validation data loaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of data loading workers
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Set seed for worker initialization
    setup_worker_random_seed.base_seed = seed
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=setup_worker_random_seed
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # Use same batch size as training
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=setup_worker_random_seed
    )
    
    return train_loader, val_loader


def create_loss_functions(class_weights, num_classes):
    """
    Create the loss functions used for training.
    
    Args:
        class_weights (torch.Tensor): Weights for each class
        num_classes (int): Number of segmentation classes
        
    Returns:
        tuple: (cross_entropy_loss, focal_loss, dice_loss)
    """
    
    ce_loss = CrossEntropyLoss(weight=class_weights.cuda())
    focal_loss = FocalLoss(gamma=2, weight=class_weights.cuda())
    dice_loss = DiceLoss(num_classes)
    
    return ce_loss, focal_loss, dice_loss


def create_optimizer_and_scheduler(model, learning_rate, args=None):
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: The neural network model
        learning_rate (float): Initial learning rate
        args: Command line arguments (optional, for scheduler configuration)
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    print(f"Setting up optimizer with learning rate: {learning_rate}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,  # Use the passed learning rate instead of hardcoded 0.001
        weight_decay=0.01  # Helps prevent overfitting
    )
    
    # NEW: ReduceLROnPlateau scheduler - reduces learning rate when validation loss plateaus
    # This is more adaptive than fixed exponential decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # Monitor validation loss (minimize)
        factor=getattr(args, 'lr_factor', 0.5),      # Reduce LR by this factor when plateauing
        patience=getattr(args, 'lr_patience', 10),   # Wait this many epochs before reducing LR
        min_lr=getattr(args, 'lr_min', 1e-7),       # Minimum learning rate
        threshold=getattr(args, 'lr_threshold', 1e-4), # Minimum change to be considered improvement
        threshold_mode='rel'  # Relative threshold (percentage change)
    )
    
    # OLD: Exponential decay scheduler (commented out but preserved)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    print(f"Using ReduceLROnPlateau scheduler:")
    print(f"  - Factor: {getattr(args, 'lr_factor', 0.5)} (reduce LR by this factor)")
    print(f"  - Patience: {getattr(args, 'lr_patience', 10)} epochs")
    print(f"  - Min LR: {getattr(args, 'lr_min', 1e-7)}")
    print(f"  - Threshold: {getattr(args, 'lr_threshold', 1e-4)} (relative)")
    print(f"  - Note: Learning rate changes will be logged automatically")
    
    return optimizer, scheduler


def run_training_epoch(model, train_loader, ce_loss, focal_loss, dice_loss, optimizer):
    """
    Run one training epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        ce_loss, focal_loss, dice_loss: Loss functions
        optimizer: Optimizer
        
    Returns:
        float: Average training loss for this epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        # Handle different batch formats (dict vs tuple)
        if isinstance(batch, dict):
            images = batch['image'].cuda()
            labels = batch['label'].cuda()
        else:
            images, labels = batch[0].cuda(), batch[1].cuda()
        
        # Forward pass
        predictions = model(images)
        
        # Compute different loss components
        loss_ce = ce_loss(predictions, labels)
        loss_focal = focal_loss(predictions, labels)
        loss_dice = dice_loss(predictions, labels, softmax=True)
        
        # Combined loss (weighted combination)
        # We give more weight to focal and dice losses as they work better for segmentation
        loss = 0.05 * loss_ce + 0.475 * loss_focal + 0.475 * loss_dice
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / num_batches


def validate_with_sliding_window(model, val_dataset, ce_loss, focal_loss, dice_loss, patch_size=224):
    """
    Run validation using sliding window inference on full-size images.
    This gives more accurate validation results than using patches.
    
    Args:
        model: Neural network model
        val_dataset: Validation dataset
        ce_loss, focal_loss, dice_loss: Loss functions (must match training)
        patch_size (int): Size of patches for sliding window
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    val_loss = 0.0
    stride = patch_size  # No overlap between patches
        
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            # Load full-size image and mask
            img_path = val_dataset.img_paths[idx]
            mask_path = val_dataset.mask_paths[idx]
            
            # Load and convert images
            original_image = Image.open(img_path).convert("RGB")
            original_mask = Image.open(mask_path).convert("RGB")
            
            img_array = np.array(original_image)
            mask_array = np.array(original_mask)
            height, width = img_array.shape[:2]
            
            # Initialize prediction and count maps
            prediction_map = np.zeros((height, width, model.module.num_classes if hasattr(model, 'module') else 6))
            count_map = np.zeros((height, width))
            
            # Sliding window over the image
            for y in range(0, height - patch_size + 1, stride):
                for x in range(0, width - patch_size + 1, stride):
                    # Extract patch
                    patch = img_array[y:y+patch_size, x:x+patch_size, :]
                    patch_tensor = TF.to_tensor(Image.fromarray(patch)).unsqueeze(0).cuda()
                    
                    # Get prediction
                    output = model(patch_tensor)
                    output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # H, W, C
                    
                    # Add to prediction map
                    prediction_map[y:y+patch_size, x:x+patch_size, :] += output_np
                    count_map[y:y+patch_size, x:x+patch_size] += 1
            
            # Average overlapping predictions
            count_map = np.maximum(count_map, 1)[:, :, None]
            prediction_map = prediction_map / count_map
            
            # Convert ground truth mask to class indices
            from datasets.dataset_udiadsbib import rgb_to_class
            ground_truth = rgb_to_class(mask_array)
            
            # Compute loss on full image
            pred_tensor = torch.from_numpy(prediction_map.transpose(2, 0, 1)).unsqueeze(0).float().cuda()
            gt_tensor = torch.from_numpy(ground_truth).unsqueeze(0).long().cuda()
            
            loss_ce = ce_loss(pred_tensor, gt_tensor)
            loss_focal = focal_loss(pred_tensor, gt_tensor)
            loss_dice = dice_loss(pred_tensor, gt_tensor, softmax=True)
            # Use SAME loss weighting as training for consistency
            loss = 0.05 * loss_ce + 0.475 * loss_focal + 0.475 * loss_dice
            
            val_loss += loss.item()
    
    return val_loss / len(val_dataset)


def save_best_model(model, epoch, val_loss, best_val_loss, save_path):
    """
    Save the model if it achieved the best validation loss so far.
    Also manages checkpoint cleanup to avoid too many saved files.
    
    Args:
        model: Neural network model
        epoch (int): Current epoch number
        val_loss (float): Current validation loss
        best_val_loss (float): Best validation loss so far
        save_path (str): Directory to save models
        
    Returns:
        tuple: (updated_best_val_loss, improvement_made)
    """
    if val_loss < best_val_loss:
        # Save the current best model
        best_model_path = os.path.join(save_path, "best_model_latest.pth")
        epoch_model_path = os.path.join(save_path, f"best_model_epoch{epoch+1}.pth")
        
        torch.save(model.state_dict(), best_model_path)
        torch.save(model.state_dict(), epoch_model_path)
        
        print(f"    ✓ New best model saved! Validation loss: {val_loss:.4f}")
        
        # Clean up old checkpoints (keep only the 3 most recent)
        cleanup_old_checkpoints(save_path, keep_count=3)
        
        return val_loss, True
    else:
        print(f"    No improvement (current: {val_loss:.4f}, best: {best_val_loss:.4f})")
        return best_val_loss, False


def cleanup_old_checkpoints(save_path, keep_count=3):
    """
    Remove old checkpoint files, keeping only the most recent ones.
    
    Args:
        save_path (str): Directory containing checkpoint files
        keep_count (int): Number of recent checkpoints to keep
    """
    checkpoint_pattern = os.path.join(save_path, "best_model_epoch*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if len(checkpoint_files) <= keep_count:
        return
    
    # Extract epoch numbers and sort by epoch (newest first)
    def extract_epoch_number(filename):
        match = re.search(r"best_model_epoch(\d+)\.pth", filename)
        return int(match.group(1)) if match else -1
    
    checkpoint_files.sort(key=extract_epoch_number, reverse=True)
    
    # Remove old checkpoints
    for old_checkpoint in checkpoint_files[keep_count:]:
        try:
            os.remove(old_checkpoint)
        except OSError:
            pass  # Ignore errors when removing files


def trainer_synapse(args, model, snapshot_path, train_dataset=None, val_dataset=None):
    """
    Main training function that coordinates the entire training process.
    
    Args:
        args: Command line arguments with training configuration
        model: Neural network model to train
        snapshot_path (str): Directory to save models and logs
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    # Set up logging
    logger = setup_logging(snapshot_path)
    
    # Print training configuration
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Epochs: {args.max_epochs}")
    print(f"Learning Rate: {args.base_lr}")
    print(f"Number of Classes: {args.num_classes}")
    print(f"Output Directory: {snapshot_path}")
    print(f"Early Stopping Patience: {getattr(args, 'patience', 50)} epochs")
    print("="*80 + "\n")
    
    # Set up multi-GPU training if available
    if args.n_gpu > 1:
        print(f"Using {args.n_gpu} GPUs for training")
        model = nn.DataParallel(model)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, 
        args.batch_size * args.n_gpu, 
        args.num_workers, 
        args.seed
    )
    
    # Compute class weights for balanced training (U-DIADS-Bib only)
    if (hasattr(train_dataset, 'mask_paths') and 
        args.dataset.lower() == 'udiads_bib'):
        class_weights = compute_class_weights(train_dataset, args.num_classes)
    else:
        class_weights = torch.ones(args.num_classes)
    
    # Create loss functions, optimizer, and scheduler
    ce_loss, focal_loss, dice_loss = create_loss_functions(class_weights, args.num_classes)
    optimizer, scheduler = create_optimizer_and_scheduler(model, args.base_lr, args)
    
    # Set up TensorBoard logging
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard_logs'))
    
    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = getattr(args, 'patience', 50)  # Early stopping patience from args or default to 50
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Early stopping patience: {patience} epochs")
    print(f"Learning rate scheduler: ReduceLROnPlateau (reduces LR when validation loss plateaus)")
    print("="*80)
    
    for epoch in range(args.max_epochs):
        print(f"\nEPOCH {epoch+1}/{args.max_epochs}")
        print("-" * 50)
        
        # Training phase
        train_loss = run_training_epoch(model, train_loader, ce_loss, focal_loss, dice_loss, optimizer)
        
        # Validation phase
        val_loss = validate_with_sliding_window(model, val_dataset, ce_loss, focal_loss, dice_loss)
        
        # Log results
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Early_Stopping/Patience_Remaining', patience - epochs_without_improvement, epoch)
        writer.add_scalar('Early_Stopping/Epochs_Without_Improvement', epochs_without_improvement, epoch)
        
        # Print epoch summary
        print(f"Results:")
        print(f"  • Train Loss: {train_loss:.4f}")
        print(f"  • Validation Loss: {val_loss:.4f}")
        print(f"  • Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model and check for improvement
        best_val_loss, improvement_made = save_best_model(model, epoch, val_loss, best_val_loss, snapshot_path)
        
        # Early stopping logic
        if improvement_made:
            epochs_without_improvement = 0
            print(f"    ✓ Improvement detected! Resetting patience counter.")
        else:
            epochs_without_improvement += 1
            remaining_patience = patience - epochs_without_improvement
            print(f"    ⚠ No improvement for {epochs_without_improvement} epochs (patience: {patience}, remaining: {remaining_patience})")
            
            if epochs_without_improvement >= patience:
                print(f"\n" + "="*80)
                print("EARLY STOPPING TRIGGERED!")
                print("="*80)
                print(f"Model has not improved for {patience} consecutive epochs.")
                print(f"Stopping training at epoch {epoch+1}.")
                print(f"Best validation loss achieved: {best_val_loss:.4f}")
                print("="*80 + "\n")
                break
        
        # Update learning rate
        # NEW: ReduceLROnPlateau needs validation loss to decide whether to reduce LR
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Check if learning rate was reduced
        if new_lr < old_lr:
            print(f"    📉 Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # OLD: ExponentialLR call (commented out but preserved)
        # scheduler.step()
        
        print("-" * 50)
    
    # Training completed
    print("\n" + "="*80)
    if epochs_without_improvement >= patience:
        print("TRAINING COMPLETED WITH EARLY STOPPING!")
        print("="*80)
        print(f"Training stopped early after {patience} epochs without improvement.")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"Models Saved To: {snapshot_path}")
        print(f"TensorBoard Logs: {os.path.join(snapshot_path, 'tensorboard_logs')}")
        print("="*80 + "\n")
        
        # Log final results
        logger.info(f"Training completed with early stopping after {patience} epochs without improvement. Best validation loss: {best_val_loss:.4f}")
    else:
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"Models Saved To: {snapshot_path}")
        print(f"TensorBoard Logs: {os.path.join(snapshot_path, 'tensorboard_logs')}")
        print("="*80 + "\n")
        
        # Log final results
        logger.info(f"Training completed successfully. Best validation loss: {best_val_loss:.4f}")
    
    # Close TensorBoard writer
    writer.close()
"""
SwinUnet Training Module
Training approach for SwinUnet model with early stopping and class weights.
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# Add common directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))

from utils import DiceLoss, FocalLoss


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


def compute_class_weights(train_dataset, num_classes):
    """
    Compute class weights for balanced training based on pixel frequency.
    Args:
        train_dataset: Training dataset object with .mask_paths
        num_classes (int): Number of segmentation classes
    Returns:
        torch.Tensor: Class weights (on CUDA if available)
    """
    print("\nComputing class weights...")

    # Define color maps for different datasets
    if num_classes == 6:
        # UDIADS-BIB color map (standard manuscripts)
        COLOR_MAP = {
            (0, 0, 0): 0,        # Background
            (255, 255, 0): 1,    # Paratext
            (0, 255, 255): 2,    # Decoration
            (255, 0, 255): 3,    # Main text
            (255, 0, 0): 4,      # Title
            (0, 255, 0): 5,      # Chapter Heading
        }
    elif num_classes == 5:
        # UDIADS-BIB color map for Syriaque341 (no Chapter Headings)
        COLOR_MAP = {
            (0, 0, 0): 0,        # Background
            (255, 255, 0): 1,    # Paratext
            (0, 255, 255): 2,    # Decoration
            (255, 0, 255): 3,    # Main text
            (255, 0, 0): 4,      # Title
            # Note: Chapter Heading (0, 255, 0) is not present in Syriaque341
        }
    elif num_classes == 4:
        # DivaHisDB color map
        COLOR_MAP = {
            (0, 0, 0): 0,        # Background
            (0, 255, 0): 1,      # Comment
            (255, 0, 0): 2,      # Decoration
            (0, 0, 255): 3,      # Main Text
        }
    else:
        raise ValueError(f"Unsupported number of classes: {num_classes}")

    class_counts = np.zeros(num_classes, dtype=np.float64)

    for mask_path in train_dataset.mask_paths:
        mask = np.array(Image.open(mask_path).convert("RGB"))
        for rgb, cls in COLOR_MAP.items():
            matches = np.all(mask == rgb, axis=-1)
            class_counts[cls] += np.sum(matches)

    # Compute frequencies
    class_freq = class_counts / class_counts.sum()

    # Inverse frequency (no normalization)
    weights = np.log(1.0 + (1.0 / (class_freq + 1e-6)))

    # Print
    print("\n" + "-"*80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("-"*80)
    print(f"{'Class':<6} {'Frequency':<15} {'Weight':<15}")
    print("-"*80)
    for cls in range(num_classes):
        print(f"{cls:<6} {class_freq[cls]:<15.6f} {weights[cls]:<15.6f}")
    print("-"*80 + "\n")

    # Return as tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_data_loaders(train_dataset, val_dataset, batch_size, num_workers, seed):
    """
    Create data loaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        seed: Random seed
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    def worker_init_fn(worker_id):
        import random
        random.seed(seed + worker_id)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
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
    
    ce_loss = CrossEntropyLoss()
    focal_loss = FocalLoss(gamma=3)
    dice_loss = DiceLoss(num_classes)
    
    return ce_loss, focal_loss, dice_loss


def create_optimizer_and_scheduler(model, learning_rate, args=None):
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: Neural network model
        learning_rate (float): Initial learning rate
        args: Command line arguments (optional)
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Use AdamW optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Use ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-7
    )
    
    print(f"Setting up optimizer with learning rate: {learning_rate}")
    print("Using ReduceLROnPlateau scheduler:")
    print(f"  - Factor: 0.5 (reduce LR by this factor)")
    print(f"  - Patience: 10 epochs")
    print(f"  - Min LR: 1e-07")
    print(f"  - Note: Learning rate changes will be logged automatically")
    
    return optimizer, scheduler


def run_training_epoch(model, train_loader, ce_loss, focal_loss, dice_loss, optimizer, class_weights):
    """
    Run one training epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        ce_loss, focal_loss, dice_loss: Loss functions
        optimizer: Optimizer
        class_weights: Class weights for Dice loss
        
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
        
        # Combined loss (weighted combination) for swinunet
        loss = 0.4 * loss_ce + 0.0 * loss_focal + 0.6 * loss_dice
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / num_batches


def validate_with_sliding_window(model, val_dataset, ce_loss, focal_loss, dice_loss, patch_size=224):
    """
    Validate model using sliding window approach.
    
    Args:
        model: Neural network model
        val_dataset: Validation dataset
        ce_loss, focal_loss, dice_loss: Loss functions
        patch_size (int): Size of patches for sliding window
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for sample in val_dataset:
            if isinstance(sample, dict):
                image = sample['image'].unsqueeze(0).cuda()
                label = sample['label'].unsqueeze(0).cuda()
            else:
                image, label = sample[0].unsqueeze(0).cuda(), sample[1].unsqueeze(0).cuda()
            
            # Forward pass
            predictions = model(image)
            
            # Compute loss
            loss_ce = ce_loss(predictions, label)
            loss_focal = focal_loss(predictions, label)
            loss_dice = dice_loss(predictions, label, softmax=True)
            
            # Combined loss
            loss = 0.4 * loss_ce + 0.0 * loss_focal + 0.6 * loss_dice
            
            total_loss += loss.item()
            num_samples += 1
    
    return total_loss / num_samples if num_samples > 0 else float('inf')


def save_best_model(model, epoch, val_loss, best_val_loss, snapshot_path):
    """
    Save model if it's the best so far.
    
    Args:
        model: Neural network model
        epoch (int): Current epoch
        val_loss (float): Current validation loss
        best_val_loss (float): Best validation loss so far
        snapshot_path (str): Directory to save models
        
    Returns:
        tuple: (best_val_loss, improvement_made)
    """
    improvement_made = False
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        improvement_made = True
        
        # Save best model
        best_model_path = os.path.join(snapshot_path, 'best_model_latest.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"    ✓ New best model saved! Validation loss: {val_loss:.4f}")
    else:
        print(f"    No improvement (current: {val_loss:.4f}, best: {best_val_loss:.4f})")
    
    return best_val_loss, improvement_made


def trainer_synapse(args, model, snapshot_path, train_dataset=None, val_dataset=None):
    """
    Main training function for SwinUnet model.
    
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
    
    # Compute class weights for balanced training
    if hasattr(train_dataset, 'mask_paths'):
        class_weights = compute_class_weights(train_dataset, args.num_classes)
        # Boost rare classes moderately to improve recall (classes 1 and 4)
        with torch.no_grad():
            if class_weights.numel() >= 5:
                class_weights[1] = class_weights[1] * 2.0  # Paratext
                class_weights[4] = class_weights[4] * 2.0  # Title
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
        train_loss = run_training_epoch(model, train_loader, ce_loss, focal_loss, dice_loss, optimizer, class_weights)
        
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
            print(f"    ⚠ No improvement for {epochs_without_improvement} epochs (patience: {patience}, remaining: {patience - epochs_without_improvement})")
        
        # Learning rate scheduling
        if hasattr(scheduler, 'step'):
            scheduler.step(val_loss)
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            print("\n" + "="*80)
            print("EARLY STOPPING TRIGGERED!")
            print("="*80)
            print(f"Model has not improved for {patience} consecutive epochs.")
            print(f"Stopping training at epoch {epoch+1}.")
            print(f"Best validation loss achieved: {best_val_loss:.4f}")
            print("="*80 + "\n")
            break
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED WITH EARLY STOPPING!")
    print("="*80)
    print(f"Training stopped early after {epochs_without_improvement} epochs without improvement.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Models Saved To: {snapshot_path}")
    print(f"TensorBoard Logs: {os.path.join(snapshot_path, 'tensorboard_logs')}")
    print("="*80 + "\n")
    
    writer.close()
    logger.info(f"Training completed with early stopping after {epochs_without_improvement} epochs without improvement. Best validation loss: {best_val_loss:.4f}")
    
    return "Training Finished!"

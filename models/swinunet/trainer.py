"""
SwinUnet Training Module
Training approach for SwinUnet model with early stopping and class weights.
Enhanced with features from Network/Hybrid models:
- Auto-resume checkpoint functionality
- Mixed Precision Training (AMP)
- Effective Number of Samples (ENS) class weighting
- Balanced sampler for rare classes
- Multiple LR scheduler types
- NaN/Inf detection
- Periodic checkpoints
- Enhanced error handling
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
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import warnings
from collections import defaultdict

# Add common directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))

from utils import DiceLoss, FocalLoss  # pyright: ignore[reportMissingImports]


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


def compute_class_weights(train_dataset, num_classes, method='ens'):
    """
    Compute class weights for balanced training based on pixel frequency.
    
    Args:
        train_dataset: Training dataset object with .mask_paths
        num_classes (int): Number of segmentation classes
        method (str): Weight computation method - 'ens' or 'log_smooth'
            - 'ens': Effective Number of Samples (ENS) - state-of-the-art for extreme imbalance (default)
            - 'log_smooth': Inverse frequency with logarithmic smoothing
    
    Returns:
        torch.Tensor: Class weights (on CUDA if available)
    """
    print(f"\nComputing class weights using method: {method}...")

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

    # Count pixels per class
    class_counts = np.zeros(num_classes, dtype=np.float64)

    for mask_path in train_dataset.mask_paths:
        mask = np.array(Image.open(mask_path).convert("RGB"))
        for rgb, cls in COLOR_MAP.items():
            matches = np.all(mask == rgb, axis=-1)
            class_counts[cls] += np.sum(matches)

    # Compute frequencies
    total_pixels = class_counts.sum()
    class_freq = class_counts / total_pixels if total_pixels > 0 else class_counts

    if method == 'ens':
        # Effective Number of Samples (ENS) method
        # Paper: "Class-Balanced Loss Based on Effective Number of Samples"
        # ENS = (1 - β^n) / (1 - β)
        # where n = number of samples per class, β = re-weighting factor
        beta = 0.9999  # Standard value for extreme imbalance
        
        # Compute ENS weights
        ens_weights = np.zeros(num_classes, dtype=np.float64)
        for cls in range(num_classes):
            n_samples = class_counts[cls]
            if n_samples > 0:
                ens = (1.0 - np.power(beta, n_samples)) / (1.0 - beta)
                ens_weights[cls] = 1.0 / (ens + 1e-8)  # Inverse ENS
            else:
                ens_weights[cls] = 1.0
        
        # Normalize to background = 1.0
        if ens_weights[0] > 0:
            weights = ens_weights / ens_weights[0]
        else:
            weights = ens_weights
        
        # Additional boosting for extreme cases (as per user's suggestion)
        weights[1] *= 8.0   # Paratext (was 6.0)
        weights[4] *= 10.0  # Title - rarest (was 8.0)
        if num_classes >= 6:
            weights[5] *= 7.0   # Chapter Heading (was 5.0)
        
        method_name = "ENS (Effective Number of Samples)"
    else:
        # Default: Inverse frequency with logarithmic smoothing
        weights = np.log(1.0 + (1.0 / (class_freq + 1e-6)))
        method_name = "Logarithmic Smoothing"

    # Print analysis
    print("\n" + "="*80)
    print(f"CLASS DISTRIBUTION ANALYSIS - {method_name}")
    print("="*80)
    print(f"{'Class':<6} {'Frequency':<15} {'Pixel Count':<15} {'Weight':<15}")
    print("-"*80)
    for cls in range(num_classes):
        print(f"{cls:<6} {class_freq[cls]:<15.6f} {class_counts[cls]:<15.0f} {weights[cls]:<15.6f}")
    print("="*80 + "\n")

    # Return as tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_balanced_sampler(train_dataset, num_classes, threshold=0.01, eps=1e-6):
    """
    Create a WeightedRandomSampler that oversamples images containing rare classes.
    
    Uses continuous rarity scores with square-root inverse frequency to prevent
    overly aggressive oversampling that can cause noisy gradients.
    
    Returns None if dataset is invalid.
    """
    if not hasattr(train_dataset, 'mask_paths') or len(train_dataset.mask_paths) == 0:
        return None

    # Build color map for dataset classes (same mapping as compute_class_weights)
    if num_classes == 6:
        COLOR_MAP = {
            (0, 0, 0): 0,
            (255, 255, 0): 1,
            (0, 255, 255): 2,
            (255, 0, 255): 3,
            (255, 0, 0): 4,
            (0, 255, 0): 5,
        }
    elif num_classes == 5:
        COLOR_MAP = {
            (0, 0, 0): 0,
            (255, 255, 0): 1,
            (0, 255, 255): 2,
            (255, 0, 255): 3,
            (255, 0, 0): 4,
        }
    elif num_classes == 4:
        COLOR_MAP = {
            (0, 0, 0): 0,
            (0, 255, 0): 1,
            (255, 0, 0): 2,
            (0, 0, 255): 3,
        }
    else:
        return None

    # compute per-class pixel counts
    class_counts = np.zeros(num_classes, dtype=np.int64)
    mapping = {k: v for k, v in COLOR_MAP.items()}
    map_int = { (r << 16) | (g << 8) | b: cls for (r, g, b), cls in mapping.items() }

    for mask_path in train_dataset.mask_paths:
        mask = np.array(Image.open(mask_path).convert('RGB'))
        rgb_int = (mask[:, :, 0].astype(np.uint32) << 16) | (mask[:, :, 1].astype(np.uint32) << 8) | mask[:, :, 2].astype(np.uint32)
        flat = rgb_int.ravel()
        label_flat = np.full(flat.shape, -1, dtype=np.int32)
        for rgb_val, cls in map_int.items():
            if np.any(flat == rgb_val):
                label_flat[flat == rgb_val] = int(cls)
        valid = label_flat >= 0
        if np.any(valid):
            counts = np.bincount(label_flat[valid].astype(np.int64), minlength=num_classes)
            class_counts += counts

    total = class_counts.sum()
    if total == 0:
        return None

    # Compute class frequencies
    class_freq = class_counts.astype(np.float64) / float(total)

    # Compute continuous rarity scores for all samples
    # Use square-root inverse frequency to prevent overly aggressive oversampling
    sample_weights = []
    for mask_path in train_dataset.mask_paths:
        mask = np.array(Image.open(mask_path).convert('RGB'))
        rgb_int = (mask[:, :, 0].astype(np.uint32) << 16) | (mask[:, :, 1].astype(np.uint32) << 8) | mask[:, :, 2].astype(np.uint32)
        present = set()
        for rgb_val, cls in map_int.items():
            if np.any(rgb_int == rgb_val):
                present.add(cls)
        
        if len(present) == 0:
            # No valid classes found, use uniform weight
            sample_weights.append(1.0)
        else:
            # Compute rarity score: sum of square-root inverse frequency for present classes
            # Square-root provides smoother interpolation than linear inverse frequency
            w = 0.0
            for cls in present:
                w += (1.0 / (class_freq[cls] + eps)) ** 0.5
            sample_weights.append(float(w))

    # Normalize weights to mean=1 for stable sampling probabilities
    sw = np.array(sample_weights, dtype=np.float64)
    sw = sw / (sw.mean() + eps)

    # Create PyTorch sampler
    weights_tensor = torch.DoubleTensor(sw)
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)
    print(f"Balanced sampler created (continuous rarity-based oversampling).")
    return sampler


def create_data_loaders(train_dataset, val_dataset, batch_size, num_workers, seed, sampler=None):
    """
    Create data loaders for training and validation.
    
    If `sampler` is provided, it will be used for the training loader and
    `shuffle` will be disabled (as required by PyTorch DataLoader).
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        seed: Random seed
        sampler: Optional sampler for training dataset (e.g., WeightedRandomSampler)
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    def worker_init_fn(worker_id):
        import random
        random.seed(seed + worker_id)
    
    if sampler is not None:
        # When providing a sampler, DataLoader requires shuffle=False
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
    else:
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


def get_rare_class_indices(class_weights, frequency_threshold=0.01, weight_threshold=2.0):
    """
    Dynamically determine rare class indices based on class weights.
    
    Rare classes are identified as those with weights significantly higher than background,
    indicating they have low frequency in the dataset.
    
    Args:
        class_weights (torch.Tensor): Class weights tensor [num_classes]
        frequency_threshold (float): Frequency threshold (not used if weights available, kept for API)
        weight_threshold (float): Weight threshold - classes with weight > weight_threshold * background_weight
                                  are considered rare (default: 2.0)
    
    Returns:
        set: Set of rare class indices (excluding background class 0)
    """
    if class_weights is None or len(class_weights) == 0:
        return set()
    
    # Convert to numpy for easier comparison
    weights = class_weights.cpu().numpy() if isinstance(class_weights, torch.Tensor) else class_weights
    
    # Background is typically class 0 with weight ~1.0 (normalized)
    background_weight = weights[0] if len(weights) > 0 else 1.0
    
    # Classes with weight significantly higher than background are rare
    # Exclude background (class 0) from rare classes
    rare_classes = set()
    for i in range(1, len(weights)):  # Skip background (class 0)
        if weights[i] > weight_threshold * background_weight:
            rare_classes.add(i)
    
    return rare_classes


def create_loss_functions(class_weights, num_classes):
    """
    Create the loss functions used for training.
    
    Args:
        class_weights (torch.Tensor): Weights for each class
        num_classes (int): Number of segmentation classes
        
    Returns:
        tuple: (cross_entropy_loss, focal_loss, dice_loss)
    """
    # CrossEntropyLoss with class weights
    ce_loss = CrossEntropyLoss(weight=class_weights)
    
    # FocalLoss with class weights and gamma=3 (for extreme imbalance)
    focal_loss = FocalLoss(gamma=5, weight=class_weights)
    
    # DiceLoss with class weights (CRITICAL: pass weights for proper class balancing)
    # NOTE: The DiceLoss implementation in common/utils/utils.py DOES use class weights correctly:
    #   - Multiplies each class's Dice loss by its weight (line 290: loss += dice * class_weights[i])
    #   - Normalizes by sum of weights, not n_classes (line 296: return loss / total_weight)
    # If you're using a different DiceLoss implementation, verify it supports weighted classes.
    # Many standard Dice implementations ignore per-class weights - check your implementation!
    dice_loss = DiceLoss(num_classes, weight=class_weights)
    
    # Verify DiceLoss weights are being used (print first time only)
    if hasattr(create_loss_functions, '_weights_printed'):
        pass  # Already printed
    else:
        create_loss_functions._weights_printed = True
        if class_weights is not None and isinstance(class_weights, torch.Tensor):
            weights_list = class_weights.cpu().tolist()
            print(f"Loss: CE (weighted) + Focal (γ=5, weighted) + Dice (weighted) | Weights: {[f'{w:.2f}' for w in weights_list]}")
        else:
            print(f"Loss: CE (weighted) + Focal (γ=5, weighted) + Dice (weighted)")
    
    return ce_loss, focal_loss, dice_loss


def create_optimizer_and_scheduler(model, learning_rate, args=None, train_loader=None):
    """
    Create optimizer and learning rate scheduler with support for multiple scheduler types.
    
    Args:
        model: Neural network model
        learning_rate (float): Initial learning rate
        args: Command line arguments (optional)
        train_loader: Training data loader (optional, needed for OneCycleLR)
        
    Returns:
        tuple: (optimizer, scheduler, scheduler_type)
    """
    # Use AdamW optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Choose scheduler type from args or default to CosineAnnealingWarmRestarts
    # CosineAnnealingWarmRestarts is recommended for extreme class imbalance as it gives
    # the model more time to learn rare classes with longer cycles
    scheduler_type = getattr(args, 'scheduler_type', 'CosineAnnealingWarmRestarts') if args else 'CosineAnnealingWarmRestarts'
    max_epochs = getattr(args, 'max_epochs', 300) if args else 300
    warmup_epochs = getattr(args, 'warmup_epochs', 10) if args else 10
    
    if scheduler_type == 'OneCycleLR' and train_loader is not None:
        steps_per_epoch = len(train_loader)
        total_steps = max_epochs * steps_per_epoch
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 4,  # Reduced from *10 to *4 for less aggressive LR peak
            total_steps=total_steps,
            pct_start=warmup_epochs / max_epochs,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100
        )
        scheduler_name = f"OneCycleLR (max_lr={learning_rate*4:.4f})"
        
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=1e-7
        )
        scheduler_name = f"CosineAnnealingLR (T_max={max_epochs - warmup_epochs})"
        
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        # Configurable T_0 for faster adaptation while still giving rare classes time to learn
        # Default T_0=50 epochs (reduced from 80 for faster adaptation)
        t_0 = getattr(args, 'scheduler_t0', 50) if args else 50
        t_mult = getattr(args, 'scheduler_t_mult', 2) if args else 2
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0,  # Initial cycle length (configurable, default 50 for faster adaptation)
            T_mult=t_mult,  # Cycle length multiplier (default 2, doubles each restart)
            eta_min=1e-7
        )
        scheduler_name = f"CosineAnnealingWarmRestarts (T_0={t_0}, T_mult={t_mult})"
        
    else:  # ReduceLROnPlateau (fallback)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        scheduler_name = "ReduceLROnPlateau (patience=10)"
    
    print(f"Optimizer: AdamW (lr={learning_rate}) | Scheduler: {scheduler_name}")
    
    return optimizer, scheduler, scheduler_type


def run_training_epoch(model, train_loader, ce_loss, focal_loss, dice_loss, optimizer, 
                       scheduler=None, scheduler_type='ReduceLROnPlateau',
                       use_amp=False, scaler=None, class_weights=None):
    """
    Run one training epoch with AMP support and NaN/Inf detection.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        ce_loss, focal_loss, dice_loss: Loss functions
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        scheduler_type: Type of scheduler - determines when to step
            - 'OneCycleLR': Step per batch
            - Others: Step per epoch (caller handles)
        use_amp: Whether to use automatic mixed precision (FP16) for faster training
        scaler: GradScaler instance for AMP (required if use_amp=True)
        class_weights: Class weights (kept for backward compatibility, not used)
        
    Returns:
        float: Average training loss for this epoch
    """
    import math
    
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    skipped_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Handle different batch formats (dict vs tuple)
        if isinstance(batch, dict):
            images = batch['image']
            labels = batch['label']
        else:
            images, labels = batch[0], batch[1]
        
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        
        # Forward pass with automatic mixed precision (AMP) for faster training
        if use_amp and scaler is not None:
            with autocast():
                predictions = model(images)
                
                # Compute different loss components
                loss_ce = ce_loss(predictions, labels)
                loss_focal = focal_loss(predictions, labels)
                loss_dice = dice_loss(predictions, labels, softmax=True)
                
                # Combined loss: 30% CE + 20% Focal + 50% Dice
                loss = 0.15 * loss_ce + 0.55 * loss_focal + 0.30 * loss_dice
        else:
            predictions = model(images)
            
            # Compute different loss components
            loss_ce = ce_loss(predictions, labels)
            loss_focal = focal_loss(predictions, labels)
            loss_dice = dice_loss(predictions, labels, softmax=True)
            
            # Combined loss: 30% CE + 20% Focal + 50% Dice
            loss = 0.15 * loss_ce + 0.55 * loss_focal + 0.30 * loss_dice
        
        # Check for NaN/Inf loss (like hybrid/network models)
        if torch.isnan(loss) or torch.isinf(loss):
            skipped_batches += 1
            if skipped_batches <= 5:  # Only print first few warnings
                print(f"  WARNING: Skipping batch {batch_idx + 1}: NaN/Inf loss detected")
            optimizer.zero_grad()
            continue
        
        # Backward pass with AMP support
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping (required for AMP)
            scaler.unscale_(optimizer)
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check for NaN/Inf gradients
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                skipped_batches += 1
                if skipped_batches <= 5:
                    print(f"  WARNING: Skipping batch {batch_idx + 1}: NaN/Inf gradients detected")
                scaler.update()
                continue
            
            # Optimizer step with AMP
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Step scheduler per batch (for OneCycleLR)
        if scheduler is not None and scheduler_type == 'OneCycleLR':
            scheduler.step()
        
        total_loss += loss.item()
    
    if skipped_batches > 0:
        print(f"  WARNING: Skipped {skipped_batches} batches due to NaN/Inf (out of {num_batches} total)")
    
    return total_loss / (num_batches - skipped_batches) if (num_batches - skipped_batches) > 0 else float('inf')


def compute_per_class_dice(predictions, labels, num_classes, smooth=1e-4):
    """
    Compute per-class Dice scores for segmentation predictions.
    
    Args:
        predictions: Model predictions [B, C, H, W] (logits or probabilities)
        labels: Ground truth labels [B, H, W] (class indices)
        num_classes: Number of classes
        smooth: Smoothing factor to prevent division by zero
        
    Returns:
        torch.Tensor: Per-class Dice scores [num_classes]
    """
    # Apply softmax if needed (predictions are logits)
    if predictions.size(1) == num_classes:
        probs = torch.softmax(predictions, dim=1)
    else:
        probs = predictions
    
    # Convert labels to one-hot encoding
    labels_one_hot = torch.zeros_like(probs)
    for i in range(num_classes):
        labels_one_hot[:, i] = (labels == i).float()
    
    # Compute Dice score per class
    dice_scores = torch.zeros(num_classes, device=predictions.device)
    
    for i in range(num_classes):
        pred_class = probs[:, i]
        target_class = labels_one_hot[:, i]
        
        # Compute intersection and union
        intersect = (pred_class * target_class).sum()
        pred_sum = pred_class.sum()
        target_sum = target_class.sum()
        union = pred_sum + target_sum
        
        # Compute Dice score with smoothing
        if union > smooth:
            dice = (2.0 * intersect + smooth) / (union + smooth)
        else:
            # Handle edge cases: if class doesn't appear in target, return NaN (exclude from metrics)
            # This prevents artificially inflating scores for absent classes
            if target_sum == 0:
                # Class not present in ground truth - return NaN to exclude from mean computation
                dice = torch.tensor(float('nan'), device=predictions.device)
            else:
                # Class present but prediction is empty - return 0 (incorrect prediction)
                dice = torch.tensor(0.0, device=predictions.device)
        
        dice_scores[i] = dice
    
    return dice_scores


def validate_with_sliding_window(model, val_dataset, ce_loss, focal_loss, dice_loss, num_classes, patch_size=224):
    """
    Validate model using sliding window approach with per-class metrics.
    
    Args:
        model: Neural network model
        val_dataset: Validation dataset
        ce_loss, focal_loss, dice_loss: Loss functions
        num_classes: Number of segmentation classes
        patch_size (int): Size of patches for sliding window
        
    Returns:
        tuple: (average_validation_loss, per_class_dice_scores)
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    # Accumulate per-class Dice scores (track counts for classes that appear)
    total_dice_scores = torch.zeros(num_classes, device='cuda' if torch.cuda.is_available() else 'cpu')
    class_counts = torch.zeros(num_classes, device='cuda' if torch.cuda.is_available() else 'cpu')
    
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
            
            # Combined loss: 30% CE + 20% Focal + 50% Dice
            loss = 0.15 * loss_ce + 0.55 * loss_focal + 0.30 * loss_dice
            
            # Compute per-class Dice scores
            per_class_dice = compute_per_class_dice(predictions, label, num_classes)
            
            # Accumulate only valid (non-NaN) Dice scores
            for i in range(num_classes):
                if not torch.isnan(per_class_dice[i]):
                    total_dice_scores[i] += per_class_dice[i]
                    class_counts[i] += 1.0
            
            total_loss += loss.item()
            num_samples += 1
    
    # Average per-class Dice scores (only for classes that appeared)
    # Classes that never appeared will have NaN
    avg_dice_scores = torch.zeros(num_classes, device=total_dice_scores.device)
    for i in range(num_classes):
        if class_counts[i] > 0:
            avg_dice_scores[i] = total_dice_scores[i] / class_counts[i]
        else:
            avg_dice_scores[i] = torch.tensor(float('nan'), device=total_dice_scores.device)
    
    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    return avg_loss, avg_dice_scores


def save_best_model(model, epoch, val_loss, best_val_loss, mean_dice, best_mean_dice, snapshot_path,
                    optimizer=None, scheduler=None, scaler=None, use_dice_for_early_stopping=True,
                    epochs_without_improvement=0):
    """
    Save model + optimizer/scheduler checkpoint based on mean per-class Dice score.
    
    Uses mean Dice (excluding background) for early stopping instead of validation loss,
    since overall loss is dominated by majority classes and doesn't reflect rare class performance.
    
    Args:
        model: Neural network model
        epoch (int): Current epoch
        val_loss (float): Current validation loss (kept for logging)
        best_val_loss (float): Best validation loss so far (kept for logging)
        mean_dice (float): Current mean per-class Dice score (foreground classes)
        best_mean_dice (float): Best mean Dice score so far
        snapshot_path (str): Directory to save models
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        scaler: GradScaler for AMP (optional)
        use_dice_for_early_stopping (bool): If True, use Dice for early stopping; else use val_loss
        
    Returns:
        tuple: (best_val_loss, best_mean_dice, improvement_made)
    """
    improvement_made = False
    
    # Use mean Dice for early stopping (better reflects rare class performance)
    if use_dice_for_early_stopping:
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            improvement_made = True
            metric_name = "Mean Dice"
            current_metric = mean_dice
            best_metric = best_mean_dice
        else:
            metric_name = "Mean Dice"
            current_metric = mean_dice
            best_metric = best_mean_dice
    else:
        # Fallback to validation loss (legacy behavior)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improvement_made = True
            metric_name = "Val Loss"
            current_metric = val_loss
            best_metric = best_val_loss
        else:
            metric_name = "Val Loss"
            current_metric = val_loss
            best_metric = best_val_loss
    
    if improvement_made:
        best_model_path = os.path.join(snapshot_path, 'best_model_latest.pth')
        
        # Build checkpoint dict (like network/hybrid models)
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state': model_state,
            'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
            'scaler_state': scaler.state_dict() if scaler is not None else None,
            'best_val_loss': best_val_loss,
            'best_mean_dice': best_mean_dice,
            'epochs_without_improvement': epochs_without_improvement,  # Save early stopping counter
        }
        
        # Save checkpoint
        torch.save(checkpoint, best_model_path)
        print(f"    New best model! {metric_name}: {current_metric:.4f} (Val Loss: {val_loss:.4f})")
    else:
        print(f"    No improvement - {metric_name}: {current_metric:.4f} (best: {best_metric:.4f}), Val Loss: {val_loss:.4f}")
    
    return best_val_loss, best_mean_dice, improvement_made


def trainer_synapse(args, model, snapshot_path, train_dataset=None, val_dataset=None):
    """
    Main training function for SwinUnet model with enhanced features.
    
    Enhanced features:
    - Auto-resume checkpoint functionality
    - Mixed Precision Training (AMP)
    - Effective Number of Samples (ENS) class weighting
    - Balanced sampler for rare classes
    - Multiple LR scheduler types
    - NaN/Inf detection
    - Periodic checkpoints
    - Enhanced error handling
    
    Args:
        args: Command line arguments with training configuration
        model: Neural network model to train
        snapshot_path (str): Directory to save models and logs
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    import random
    import math
    
    # Set random seeds for reproducibility
    seed = getattr(args, 'seed', 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
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
    print(f"Early Stopping Patience: {getattr(args, 'patience', 30)} epochs")
    print("="*80 + "\n")
    
    # Set up multi-GPU training if available
    if args.n_gpu > 1:
        print(f"Using {args.n_gpu} GPUs for training\n")
        model = nn.DataParallel(model)
    
    # Create balanced sampler if requested
    # IMPORTANT: Use EITHER balanced sampler OR aggressive class weights, not both
    # Using both causes over-correction, instability, and suboptimal convergence
    use_balanced_sampler = getattr(args, 'use_balanced_sampler', False)
    sampler = None
    if use_balanced_sampler:
        if hasattr(train_dataset, 'mask_paths'):
            sampler = create_balanced_sampler(train_dataset, args.num_classes)
            if sampler is not None:
                print("Balanced sampler enabled (oversampling rare classes)")
                print("  → Using uniform class weights to avoid double correction")
            else:
                print("WARNING: Balanced sampler requested but could not be created (using default shuffling)")
                use_balanced_sampler = False  # Fall back to class weights
        else:
            print("WARNING: Balanced sampler requested but dataset doesn't have mask_paths (using default shuffling)")
            use_balanced_sampler = False  # Fall back to class weights
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, 
        args.batch_size * args.n_gpu, 
        args.num_workers, 
        args.seed,
        sampler=sampler
    )
    
    print(f"Dataset Statistics:")
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Validation samples: {len(val_dataset)}")
    print(f"   - Batch size: {args.batch_size * args.n_gpu}")
    print(f"   - Steps per epoch: {len(train_loader)}\n")
    
    # Compute class weights: Option A (balanced sampler) vs Option B (aggressive weights)
    # IMPORTANT: When balanced sampler is enabled, we skip compute_class_weights entirely
    # to avoid wasting computation time, since we'll set weights to uniform anyway.
    if hasattr(train_dataset, 'mask_paths'):
        if use_balanced_sampler and sampler is not None:
            # Option A: Balanced sampler + uniform weights (avoid double correction)
            # Skip compute_class_weights call - no need to compute weights we won't use
            print("Class Weight Strategy: Uniform (balanced sampler handles class imbalance)")
            print("  → Skipping class weight computation (using uniform weights)")
            class_weights = torch.ones(args.num_classes)
            if torch.cuda.is_available():
                class_weights = class_weights.cuda()
        else:
            # Option B: Standard sampling + class weights (recommended for extreme imbalance)
            print("Class Weight Strategy: Weighted loss (standard sampling)")
            
            # Choose weight computation method (ENS is default for extreme imbalance)
            weight_method = getattr(args, 'weight_method', 'ens')  # 'ens' or 'log_smooth'
            class_weights = compute_class_weights(train_dataset, args.num_classes, method=weight_method)
            
            # Additional boosting only if using log_smooth method (ENS already has boosting built-in)
            if weight_method == 'log_smooth':
                with torch.no_grad():
                    # Apply aggressive boosting to match ENS levels for consistency
                    class_weights[1] = class_weights[1] * 8.0   # Paratext: 8.0x boost (matches ENS)
                    class_weights[4] = class_weights[4] * 10.0  # Title: 10.0x boost (matches ENS, rarest)
                    if class_weights.numel() >= 6:
                        class_weights[5] = class_weights[5] * 7.0   # Chapter Heading: 7.0x boost (matches ENS)
                print(f"  → Rare class boosting: Paratext (8x), Title (10x), Chapter Heading (7x)")
            else:
                print(f"  → Rare class boosting: Paratext (8x), Title (10x), Chapter Heading (7x)")
    else:
        class_weights = torch.ones(args.num_classes)
        if torch.cuda.is_available():
            class_weights = class_weights.cuda()
    
    # Create loss functions, optimizer, and scheduler
    ce_loss, focal_loss, dice_loss = create_loss_functions(class_weights, args.num_classes)
    optimizer, scheduler, scheduler_type = create_optimizer_and_scheduler(
        model, args.base_lr, args, train_loader=train_loader
    )
    
    # Store base learning rate and warmup epochs for warmup implementation
    base_lr = args.base_lr
    warmup_epochs = getattr(args, 'warmup_epochs', 10)
    
    # Initialize automatic mixed precision (AMP) for faster training
    use_amp = getattr(args, 'use_amp', True) and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("AMP: Enabled\n")
    else:
        print("AMP: Disabled (FP32)\n")
    
    # Set up TensorBoard logging
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard_logs'))
    
    # Checkpoint resume functionality (like network model)
    start_epoch = 0
    best_val_loss = float('inf')
    best_mean_dice = 0.0  # Track best mean per-class Dice (foreground classes)
    epochs_without_improvement = 0
    patience = getattr(args, 'patience', 30)
    use_dice_for_early_stopping = getattr(args, 'use_dice_for_early_stopping', True)  # Use Dice by default
    
    # Auto-resume: check for existing checkpoint
    resume_path = getattr(args, 'resume', None)
    no_auto_resume = getattr(args, 'no_auto_resume', False)
    
    checkpoint_path = None
    if resume_path:
        # User specified a checkpoint path
        checkpoint_path = resume_path
    elif not no_auto_resume:
        # Auto-detect checkpoint in output directory
        auto_checkpoint = os.path.join(snapshot_path, 'best_model_latest.pth')
        if os.path.exists(auto_checkpoint):
            checkpoint_path = auto_checkpoint
    
    # Load checkpoint if found
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"   📂 Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state
            if 'model_state' in checkpoint:
                try:
                    if isinstance(model, nn.DataParallel):
                        model.module.load_state_dict(checkpoint['model_state'], strict=False)
                    else:
                        model.load_state_dict(checkpoint['model_state'], strict=False)
                    print("   Loaded model state")
                except Exception as e:
                    print(f"   WARNING: Failed to load model state: {e}")
                    raise
            else:
                # Backward compatibility: checkpoint might be just state_dict
                try:
                    if isinstance(model, nn.DataParallel):
                        model.module.load_state_dict(checkpoint, strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                    print("   Loaded model state (legacy format)")
                except Exception as e:
                    print(f"   WARNING: Failed to load model state: {e}")
                    raise
            
            # Load optimizer state
            if optimizer is not None and 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                    print("   Loaded optimizer state")
                except Exception as e:
                    print(f"   WARNING: Could not load optimizer state: {e}")
                    print("   Starting with fresh optimizer state (this is OK if architecture changed)")
            
            # Load scheduler state (handle mismatches gracefully)
            if scheduler is not None and 'scheduler_state' in checkpoint and checkpoint['scheduler_state'] is not None:
                scheduler_state = checkpoint['scheduler_state']
                current_scheduler_type = type(scheduler).__name__
                
                # Check if scheduler state matches current scheduler type
                state_keys = set(scheduler_state.keys())
                is_onecycle = 'total_steps' in state_keys
                is_cosine_warm = 'T_0' in state_keys and 'T_mult' in state_keys
                is_cosine_simple = 'T_max' in state_keys and 'T_0' not in state_keys
                is_reduce_on_plateau = 'mode' in state_keys and 'factor' in state_keys
                
                scheduler_type_match = False
                if current_scheduler_type == 'OneCycleLR' and is_onecycle:
                    scheduler_type_match = True
                elif current_scheduler_type == 'CosineAnnealingWarmRestarts' and is_cosine_warm:
                    scheduler_type_match = True
                elif current_scheduler_type == 'CosineAnnealingLR' and is_cosine_simple:
                    scheduler_type_match = True
                elif current_scheduler_type == 'ReduceLROnPlateau' and is_reduce_on_plateau:
                    scheduler_type_match = True
                
                if scheduler_type_match:
                    if current_scheduler_type == 'OneCycleLR':
                        print(f"   WARNING: OneCycleLR scheduler cannot be resumed directly")
                        print(f"   Will recreate scheduler and fast-forward to current step")
                        checkpoint_epoch = checkpoint.get('epoch', 0)
                        if checkpoint_epoch > 0:
                            steps_per_epoch = len(train_loader)
                            current_step = checkpoint_epoch * steps_per_epoch
                            print(f"   Note: Will resume OneCycleLR from step {current_step}")
                    else:
                        try:
                            scheduler.load_state_dict(scheduler_state)
                            print("   Loaded scheduler state")
                        except Exception as e:
                            print(f"   WARNING: Could not load scheduler state: {e}")
                            print("   Starting with fresh scheduler state")
                            # Fast-forward scheduler to current epoch if epoch-based
                            if current_scheduler_type in ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR']:
                                for _ in range(checkpoint.get('epoch', 0)):
                                    scheduler.step()
                                print(f"   Fast-forwarded scheduler to epoch {checkpoint.get('epoch', 0)}")
                else:
                    print(f"   WARNING: Scheduler type mismatch (checkpoint has different scheduler type)")
                    print(f"   Starting with fresh scheduler state")
                    # Fast-forward scheduler to current epoch if epoch-based
                    if current_scheduler_type in ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR']:
                        for _ in range(checkpoint.get('epoch', 0)):
                            scheduler.step()
                        print(f"   Fast-forwarded scheduler to epoch {checkpoint.get('epoch', 0)}")
            
            # Load scaler state (for AMP)
            if scaler is not None and 'scaler_state' in checkpoint and checkpoint['scaler_state'] is not None:
                try:
                    scaler.load_state_dict(checkpoint['scaler_state'])
                    print(f"   Loaded scaler state (AMP)")
                except Exception as e:
                    print(f"   WARNING: Failed to load scaler state: {e}, starting with fresh scaler")
            
            # Load training state
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_mean_dice = checkpoint.get('best_mean_dice', 0.0)
            epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)  # Restore early stopping counter
            
            print(f"   Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
            print(f"   Best validation loss: {best_val_loss:.4f}")
            print(f"   Best mean Dice (foreground): {best_mean_dice:.4f}")
            print(f"   Epochs without improvement: {epochs_without_improvement} (patience: {patience}, remaining: {patience - epochs_without_improvement})")
            print(f"   Resuming from epoch {start_epoch}\n")
        except Exception as e:
            import traceback
            print(f"   WARNING: Failed to load checkpoint: {e}")
            print(f"   Error type: {type(e).__name__}")
            print("   Traceback:")
            traceback.print_exc()
            print("   Starting training from scratch\n")
    elif resume_path:
        # User specified a checkpoint but it doesn't exist
        print(f"   ERROR: Specified checkpoint not found: {checkpoint_path}")
        print(f"   Please check the path and try again.")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        print("   Starting from scratch\n")
    
    # Training loop
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Loss: 0.15*CE + 0.55*Focal + 0.30*Dice")
    print(f"Early stopping: {patience} epochs patience")
    if scheduler_type != 'OneCycleLR' and warmup_epochs > 0:
        if start_epoch < warmup_epochs:
            remaining_warmup = warmup_epochs - start_epoch
            print(f"Learning rate warmup: {remaining_warmup} epochs remaining (epochs {start_epoch+1}-{warmup_epochs})")
        else:
            print(f"Learning rate warmup: Already completed (resuming after warmup)")
    if start_epoch > 0:
        print(f"Resuming from epoch: {start_epoch}")
    print("="*80 + "\n")
    
    for epoch in range(start_epoch, args.max_epochs):
        try:
            print(f"\nEPOCH {epoch+1}/{args.max_epochs}")
            print("-" * 50)
            sys.stdout.flush()  # Force flush for background jobs
            
            # Training phase with AMP support
            train_loss = run_training_epoch(
                model, train_loader, ce_loss, focal_loss, dice_loss, 
                optimizer, scheduler, scheduler_type=scheduler_type,
                use_amp=use_amp, scaler=scaler
            )
            
            # Check for NaN/Inf in training loss
            if isinstance(train_loss, torch.Tensor):
                if torch.isnan(train_loss).any() or torch.isinf(train_loss).any():
                    raise ValueError(f"NaN/Inf detected in training loss: {train_loss}")
                train_loss_val = train_loss.item()
            elif isinstance(train_loss, (int, float)):
                if math.isnan(train_loss) or math.isinf(train_loss):
                    raise ValueError(f"NaN/Inf detected in training loss: {train_loss}")
                train_loss_val = float(train_loss)
            else:
                train_loss_val = float(train_loss)
            
            # Sanity check for unreasonably high training loss (indicates potential weight/LR issues)
            if train_loss_val > 5.0:
                if epoch == start_epoch:
                    # First epoch - more detailed warning
                    print(f"\n  WARNING: Training loss is very high ({train_loss_val:.4f}) on first epoch!")
                    print(f"  This may indicate:")
                    print(f"    - Class weights are too aggressive (check ENS/log_smooth method)")
                    print(f"    - Learning rate is too high (current: {optimizer.param_groups[0]['lr']:.6f})")
                    print(f"    - Loss function weights may need adjustment")
                    print(f"    - Data preprocessing issues")
                    print(f"  If loss doesn't decrease in next few epochs, consider:")
                    print(f"    - Reducing learning rate (try --base_lr 0.0001)")
                    print(f"    - Checking class weight computation")
                    print(f"    - Verifying data normalization\n")
                else:
                    # Subsequent epochs - brief warning
                    print(f"  WARNING: Training loss is very high ({train_loss_val:.4f}) - check class weights or learning rate")
            
            # Validation phase with per-class metrics
            val_loss, per_class_dice = validate_with_sliding_window(
                model, val_dataset, ce_loss, focal_loss, dice_loss, args.num_classes
            )
            
            # Check for NaN/Inf in validation loss
            if isinstance(val_loss, torch.Tensor):
                if torch.isnan(val_loss).any() or torch.isinf(val_loss).any():
                    raise ValueError(f"NaN/Inf detected in validation loss: {val_loss}")
            elif isinstance(val_loss, (int, float)):
                if math.isnan(val_loss) or math.isinf(val_loss):
                    raise ValueError(f"NaN/Inf detected in validation loss: {val_loss}")
            
            # Define class names for logging
            CLASS_NAMES = {
                6: ['Background', 'Paratext', 'Decoration', 'Main Text', 'Title', 'Chapter Heading'],
                5: ['Background', 'Paratext', 'Decoration', 'Main Text', 'Title'],
                4: ['Background', 'Comment', 'Decoration', 'Main Text']
            }
            class_names = CLASS_NAMES.get(args.num_classes, [f'Class {i}' for i in range(args.num_classes)])
            
            # Log results
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Early_Stopping/Patience_Remaining', patience - epochs_without_improvement, epoch)
            writer.add_scalar('Early_Stopping/Epochs_Without_Improvement', epochs_without_improvement, epoch)
            
            # Log per-class Dice scores to TensorBoard (skip NaN values for absent classes)
            per_class_dice_dict = {}
            for i in range(args.num_classes):
                if not torch.isnan(per_class_dice[i]):
                    per_class_dice_dict[f'Class_{i}_{class_names[i]}'] = per_class_dice[i].item()
            if per_class_dice_dict:  # Only log if there are valid classes
                writer.add_scalars('Dice/Per_Class', per_class_dice_dict, epoch)
            
            # Compute mean Dice score (excluding background and NaN values for absent classes)
            # NaN values represent classes that don't appear in validation - exclude them from mean
            foreground_dice = per_class_dice[1:] if args.num_classes > 1 else per_class_dice
            valid_dice = foreground_dice[~torch.isnan(foreground_dice)]
            if len(valid_dice) > 0:
                mean_dice_foreground = valid_dice.mean().item()
            else:
                # All foreground classes are absent - set to 0
                mean_dice_foreground = 0.0
            writer.add_scalar('Dice/Mean_Foreground', mean_dice_foreground, epoch)
            
            # Print epoch summary
            print(f"Results:")
            print(f"  • Train Loss: {train_loss:.4f}")
            print(f"  • Validation Loss: {val_loss:.4f}")
            print(f"  • Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"  • Mean Dice (Foreground): {mean_dice_foreground:.4f} {'(BEST!)' if mean_dice_foreground > best_mean_dice else ''}")
            
            # Per-class Dice scores are calculated and logged to TensorBoard but not printed to console
            # (calculation is needed for mean_dice_foreground, but printing is removed for cleaner output)
            
            # Save periodic checkpoint (every 100 epochs) - useful for recovery and evaluation
            if (epoch + 1) % 100 == 0:
                periodic_checkpoint_path = os.path.join(snapshot_path, f"epoch_{epoch + 1}.pth")
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                periodic_checkpoint = {
                    'epoch': epoch,
                    'model_state': model_state,
                    'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
                    'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                    'scaler_state': scaler.state_dict() if scaler is not None else None,
                    'best_val_loss': best_val_loss,
                    'best_mean_dice': best_mean_dice,
                    'epochs_without_improvement': epochs_without_improvement,  # Save early stopping counter
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }
                torch.save(periodic_checkpoint, periodic_checkpoint_path)
                print(f"   Periodic checkpoint saved: epoch_{epoch + 1}.pth")
            
            # Save best model based on mean Dice score (better reflects rare class performance)
            best_val_loss, best_mean_dice, improvement_made = save_best_model(
                model, epoch, val_loss, best_val_loss, mean_dice_foreground, best_mean_dice, snapshot_path,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                use_dice_for_early_stopping=use_dice_for_early_stopping,
                epochs_without_improvement=epochs_without_improvement  # Save early stopping counter
            )
            
            # Early stopping logic
            if improvement_made:
                epochs_without_improvement = 0
                print(f"    Improvement detected! Resetting patience counter.")
            else:
                epochs_without_improvement += 1
                remaining_patience = patience - epochs_without_improvement
                print(f"    WARNING: No improvement for {epochs_without_improvement} epochs (patience: {patience}, remaining: {remaining_patience})")
            
            # Learning rate warmup (gradually increase LR from small value to base_lr over warmup_epochs)
            # OneCycleLR has built-in warmup, so skip manual warmup for it
            # Check if we're still in warmup period (relative to absolute epoch, accounting for resume)
            is_in_warmup = (epoch < warmup_epochs) and (scheduler_type != 'OneCycleLR')
            
            if is_in_warmup:
                # Linear warmup: lr = base_lr * (epoch + 1) / warmup_epochs
                # Gradually increase from base_lr * (1/warmup_epochs) to base_lr
                warmup_lr = base_lr * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                if epoch == start_epoch or (epoch - start_epoch) % 5 == 0:  # Print every 5 epochs during warmup
                    print(f"  Warmup: LR = {warmup_lr:.6f} (target: {base_lr:.6f}, epoch {epoch+1}/{warmup_epochs})")
            
            # Learning rate scheduling
            if scheduler_type == 'ReduceLROnPlateau':
                # ReduceLROnPlateau adjusts based on validation - can be used during/after warmup
                scheduler_val_loss = val_loss if not math.isinf(val_loss) else 1e6
                scheduler.step(scheduler_val_loss)
            elif scheduler_type == 'OneCycleLR':
                # OneCycleLR is stepped per batch (already handled in training loop)
                pass
            elif not is_in_warmup:
                # Only step epoch-based schedulers after warmup is complete
                scheduler.step()
            
            sys.stdout.flush()  # Force flush for background jobs
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                print("\n" + "="*80)
                print("EARLY STOPPING TRIGGERED!")
                print("="*80)
                print(f"Model has not improved for {patience} consecutive epochs.")
                print(f"Stopping training at epoch {epoch+1}.")
                if use_dice_for_early_stopping:
                    print(f"Best mean Dice (foreground) achieved: {best_mean_dice:.4f}")
                    print(f"Final validation loss: {val_loss:.4f}")
                else:
                    print(f"Best validation loss achieved: {best_val_loss:.4f}")
                print("="*80 + "\n")
                break
                
        except Exception as e:
            import traceback
            print(f"\nERROR during epoch {epoch+1}: {e}")
            print("Traceback:")
            traceback.print_exc()
            print("\nContinuing to next epoch...")
            sys.stdout.flush()
            continue
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    if start_epoch >= args.max_epochs:
        print(f"Total Epochs:   {start_epoch}")
    else:
        print(f"Total Epochs:   {epoch + 1}")
    if use_dice_for_early_stopping:
        print(f"Best Mean Dice (Foreground): {best_mean_dice:.4f}")
        print(f"Best Val Loss:  {best_val_loss:.4f}")
    else:
        print(f"Best Val Loss:  {best_val_loss:.4f}")
    print(f"Models Saved:   {snapshot_path}")
    print(f"TensorBoard:    {os.path.join(snapshot_path, 'tensorboard_logs')}")
    print("="*80 + "\n")
    
    writer.close()
    if use_dice_for_early_stopping:
        logger.info(f"Training completed. Best mean Dice: {best_mean_dice:.4f}, Best val loss: {best_val_loss:.4f}")
    else:
        logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")
    
    return "Training Finished!"

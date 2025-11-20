"""
Hybrid2 Model Training Module
Training approach for Hybrid2 model with early stopping and class weights.

Hybrid2: SwinUnet encoder + EfficientNet/ResNet50 decoder

Uses training approach with all three losses (CE + Focal + Dice).
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import warnings
from collections import defaultdict

# Add common directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))

from utils import utils
from utils.utils import FocalLoss, DiceLoss

# Global worker initialization function for DataLoader (needed for Windows multiprocessing)
def worker_init_fn(worker_id, seed=1234):
    import random
    import numpy as np
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


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


def compute_class_weights(train_dataset, num_classes, smoothing=0.05):
    """
    Compute balanced class weights using effective number of samples.
    
    Uses the formula from "Class-Balanced Loss Based on Effective Number of Samples"
    (Cui et al., 2019): weight = (1 - beta) / (1 - beta^n)
    
    Args:
        train_dataset: Training dataset with .mask_paths attribute
        num_classes (int): Number of segmentation classes
        smoothing (float): Smoothing factor to prevent extreme weights (default: 0.1)
        
    Returns:
        torch.Tensor: Normalized class weights on GPU if available
    """
    print("\n" + "="*80)
    print("COMPUTING CLASS WEIGHTS")
    print("="*80)

    # Define color maps
    COLOR_MAPS = {
        6: {  # UDIADS-BIB (standard manuscripts)
            (0, 0, 0): 0,        # Background
            (255, 255, 0): 1,    # Paratext
            (0, 255, 255): 2,    # Decoration
            (255, 0, 255): 3,    # Main text
            (255, 0, 0): 4,      # Title
            (0, 255, 0): 5,      # Chapter Heading
        },
        5: {  # UDIADS-BIB Syriaque341 (no Chapter Headings)
            (0, 0, 0): 0,        # Background
            (255, 255, 0): 1,    # Paratext
            (0, 255, 255): 2,    # Decoration
            (255, 0, 255): 3,    # Main text
            (255, 0, 0): 4,      # Title
        },
        4: {  # DivaHisDB
            (0, 0, 0): 0,        # Background
            (0, 255, 0): 1,      # Comment
            (255, 0, 0): 2,      # Decoration
            (0, 0, 255): 3,      # Main Text
        }
    }
    
    if num_classes not in COLOR_MAPS:
        raise ValueError(f"Unsupported number of classes: {num_classes}")
    
    COLOR_MAP = COLOR_MAPS[num_classes]
    
    # Count pixels per class
    class_counts = np.zeros(num_classes, dtype=np.float64)
    unmapped_count = 0
    chapter_heading_count = 0
    
    # Count pixels silently (no progress messages)
    for mask_path in train_dataset.mask_paths:
        try:
            mask = np.array(Image.open(mask_path).convert("RGB"))
        except Exception as e:
            warnings.warn(f"Failed to load mask {mask_path}: {e}")
            continue
        
        mapped_mask = np.zeros(mask.shape[:2], dtype=bool)
        
        # Count pixels for each class
        for rgb, cls in COLOR_MAP.items():
            matches = np.all(mask == rgb, axis=-1)
            class_counts[cls] += np.sum(matches)
            mapped_mask[matches] = True
        
        # Track unmapped pixels (including Chapter Headings in 5-class mode)
        if num_classes == 5:
            chapter_matches = np.all(mask == (0, 255, 0), axis=-1)
            if np.any(chapter_matches):
                chapter_heading_count += np.sum(chapter_matches)
                mapped_mask[chapter_matches] = True
        
        unmapped_pixels = ~mapped_mask
        if np.any(unmapped_pixels):
            unmapped_count += np.sum(unmapped_pixels)

    # Report findings
    total_pixels = class_counts.sum()
    if total_pixels == 0:
        raise ValueError("No valid pixels found in training masks!")
    
    if num_classes == 5 and chapter_heading_count > 0:
        print(f"\n⚠️  WARNING: Found {chapter_heading_count:,} Chapter Heading pixels")
        print(f"   These will be mapped to Background (class 0)")
    
    if unmapped_count > 0:
        print(f"⚠️  WARNING: {unmapped_count:,} unmapped pixels (mapped to Background)")
    
    # Compute class frequencies
    class_freq = class_counts / total_pixels
    
    # Effective Number of Samples (ENS) weighting
    # Force beta = 0.9999 for extreme imbalance in all cases
    beta = 0.9999
    
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    
    # Debug: Print raw ENS weights before smoothing
    print("Raw ENS weights:", weights)
    
    # Apply smoothing to prevent extreme weights
    if smoothing > 0:
        # Linear interpolation between computed weights and uniform weights
        uniform_weights = np.ones(num_classes)
        weights = (1 - smoothing) * weights + smoothing * uniform_weights
    
    # Normalize to sum to num_classes (maintains balanced loss scale)
    weights = weights / weights.sum() * num_classes
    
    # Cap maximum weight to prevent dominance (max 10x the minimum)
    max_weight_ratio = 10.0
    min_weight = weights.min()
    weights = np.minimum(weights, min_weight * max_weight_ratio)
    
    # Re-normalize after capping
    weights = weights / weights.sum() * num_classes
    
    # Print class statistics
    print(f"\nClass Distribution:")
    print(f"{'Class':<20} {'Pixels':<15} {'Frequency':<15} {'Weight':<15}")
    print("-" * 65)
    for cls in range(num_classes):
        freq_pct = class_freq[cls] * 100
        weight_val = weights[cls]
        print(f"{cls:<20} {class_counts[cls]:>14,.0f} {freq_pct:>14.4f}% {weight_val:>14.4f}")
    
    print(f"\nTotal pixels: {total_pixels:,}")
    print(f"Beta: {beta}")
    print(f"Smoothing: {smoothing}")
    print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    print("="*80)
    
    # Convert to torch tensor and move to GPU if available
    weights_tensor = torch.from_numpy(weights).float()
    if torch.cuda.is_available():
        weights_tensor = weights_tensor.cuda()
    
    return weights_tensor


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
    # Validate that datasets are not empty
    if len(train_dataset) == 0:
        raise ValueError(
            f"Training dataset is empty! Found 0 samples.\n"
            f"This usually means the dataset directories don't exist or are empty.\n"
            f"Please check that the dataset path is correct and contains the expected files."
        )
    
    if len(val_dataset) == 0:
        raise ValueError(
            f"Validation dataset is empty! Found 0 samples.\n"
            f"This usually means the dataset directories don't exist or are empty.\n"
            f"Please check that the dataset path is correct and contains the expected files."
        )
    
    # On Windows, reduce num_workers to avoid multiprocessing issues
    if os.name == 'nt':  # Windows
        num_workers = min(num_workers, 2)
        if num_workers > 0:
            print(f"Windows detected: reducing num_workers to {num_workers}")
    
    # Create a partial function for worker initialization
    import functools
    worker_fn = functools.partial(worker_init_fn, seed=seed)
    
    # Use sampler if provided, otherwise use shuffle
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),  # Don't shuffle if using sampler
        sampler=sampler,  # Use sampler if provided
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_fn if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_fn if num_workers > 0 else None
    )
    
    return train_loader, val_loader


def create_loss_functions(class_weights, num_classes, focal_gamma=2.0):
    """
    Create properly weighted loss functions.
    Matches Network model's pattern.
    
    Args:
        class_weights (torch.Tensor): Per-class weights
        num_classes (int): Number of classes
        focal_gamma (float): Focal loss focusing parameter (default: 2.0)
        
    Returns:
        tuple: (ce_loss, focal_loss, dice_loss)
    """
    # CrossEntropyLoss with class weights and label smoothing (matching Network model)
    ce_loss = CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Focal loss with gamma=2.0 (matching Network model)
    focal_loss = FocalLoss(gamma=focal_gamma, weight=class_weights)
    
    # Dice loss (handles class imbalance internally, matching Network model)
    dice_loss = DiceLoss(num_classes)
    
    print(f"✓ Loss functions created: CE (weighted, label_smoothing=0.1), Focal (γ={focal_gamma}), Dice")
    
    return ce_loss, focal_loss, dice_loss


def compute_combined_loss(predictions, labels, ce_loss, focal_loss, dice_loss, 
                          ce_weight=0.25, focal_weight=0.35, dice_weight=0.4):
    """
    Compute combined loss with support for deep supervision.
    Matches Network model's pattern.
    
    Args:
        predictions: Model output (logits or tuple with auxiliary outputs)
        labels: Ground truth labels
        ce_loss, focal_loss, dice_loss: Loss functions
        ce_weight, focal_weight, dice_weight: Loss combination weights (default: 0.3, 0.2, 0.5)
        
    Returns:
        tuple: (total_loss, loss_dict) where loss_dict contains individual losses
    """
    loss_dict = {}
    
    # Handle deep supervision
    if isinstance(predictions, tuple):
        logits, aux_outputs = predictions
        
        # Main branch losses
        loss_ce = ce_loss(logits, labels)
        loss_focal = focal_loss(logits, labels)
        loss_dice = dice_loss(logits, labels, softmax=True)
        
        main_loss = ce_weight * loss_ce + focal_weight * loss_focal + dice_weight * loss_dice
        loss_dict['main'] = main_loss.item()
        loss_dict['ce'] = loss_ce.item()
        loss_dict['focal'] = loss_focal.item()
        loss_dict['dice'] = loss_dice.item()
        
        # Auxiliary losses with multi-resolution support (MSAGHNet style - matching Network model)
        # Auxiliary outputs are at native resolutions: [H/16, H/8, H/4]
        # Scale factors: [16, 8, 4] (downsample GT by these factors)
        aux_weights = [0.05 * (0.8 ** i) for i in range(len(aux_outputs))]  # Network model weights
        aux_loss = 0.0
        
        # Prepare labels for multi-resolution loss computation
        # Ensure labels are in [B, 1, H, W] format for interpolation
        if labels.dim() == 3:
            labels_4d = labels.unsqueeze(1)  # [B, 1, H, W]
        else:
            labels_4d = labels
        
        for i, (weight, aux_output) in enumerate(zip(aux_weights, aux_outputs)):
            # Get target resolution from auxiliary output
            _, _, aux_h, aux_w = aux_output.shape
            
            # Downsample ground truth to match auxiliary output resolution
            labels_downsampled = F.interpolate(
                labels_4d.float(), 
                size=(aux_h, aux_w), 
                mode='nearest'  # Use nearest neighbor for label downsampling
            ).long()
            
            # Remove channel dimension if needed for loss functions
            if labels_downsampled.shape[1] == 1:
                labels_downsampled = labels_downsampled.squeeze(1)  # [B, H, W]
            
            # Compute losses at native resolution
            aux_ce = ce_loss(aux_output, labels_downsampled)
            aux_focal = focal_loss(aux_output, labels_downsampled)
            aux_dice = dice_loss(aux_output, labels_downsampled, softmax=True)
            aux_combined = ce_weight * aux_ce + focal_weight * aux_focal + dice_weight * aux_dice
            aux_loss += weight * aux_combined
            loss_dict[f'aux_{i}'] = (weight * aux_combined).item()
            loss_dict[f'aux_{i}_res'] = f'{aux_h}x{aux_w}'
            loss_dict[f'aux_{i}_ce'] = aux_ce.item()
            loss_dict[f'aux_{i}_focal'] = aux_focal.item()
            loss_dict[f'aux_{i}_dice'] = aux_dice.item()
        
        # Total loss: main + weighted auxiliary
        total_loss = main_loss + aux_loss
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
    else:
        # Standard single output
        loss_ce = ce_loss(predictions, labels)
        loss_focal = focal_loss(predictions, labels)
        loss_dice = dice_loss(predictions, labels, softmax=True)
        
        # Balanced combination - matching Network model weights
        total_loss = ce_weight * loss_ce + focal_weight * loss_focal + dice_weight * loss_dice
        loss_dict['ce'] = loss_ce.item()
        loss_dict['focal'] = loss_focal.item()
        loss_dict['dice'] = loss_dice.item()
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


def create_optimizer_and_scheduler(model, learning_rate, args=None, train_loader=None):
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: Neural network model
        learning_rate (float): Initial learning rate
        args: Command line arguments (optional)
        train_loader: Training data loader (optional, needed for OneCycleLR)
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Hybrid2 Best Practice: Differential Learning Rates
    # Pretrained encoder gets 10x smaller LR to preserve learned features
    # Bottleneck (2 Swin blocks) gets 5x smaller LR for gradual adaptation
    # Decoder gets base LR for faster convergence
    
    encoder_params = []
    bottleneck_params = []
    decoder_params = []
    
    # Separate parameters by module
    for name, param in model.named_parameters():
        if 'encoder' in name.lower() and 'decoder' not in name.lower():
            # Encoder parameters (not decoder.encoder_projections)
            encoder_params.append(param)
        elif 'bottleneck' in name.lower():
            # Bottleneck: 2 Swin Transformer blocks
            bottleneck_params.append(param)
        else:
            # Decoder parameters (including encoder_projections, decoder blocks, etc.)
            decoder_params.append(param)
    
    # Parameter groups with differential LR and weight decay
    param_groups = [
        {
            'params': encoder_params,
            'lr': learning_rate * 0.1,  # 10x smaller for pretrained
            'weight_decay': 1e-3,  # Light regularization for pretrained
            'name': 'encoder'
        },
        {
            'params': bottleneck_params,
            'lr': learning_rate * 0.5,  # 5x smaller for bottleneck (2 Swin blocks)
            'weight_decay': 5e-3,  # Medium regularization
            'name': 'bottleneck'
        },
        {
            'params': decoder_params,
            'lr': learning_rate,  # Full LR for new modules
            'weight_decay': 1e-2,  # Strong regularization for new modules
            'name': 'decoder'
        }
    ]
    
    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    # Use AdamW optimizer with differential LR
    optimizer = optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Select scheduler based on args
    scheduler_type = getattr(args, 'scheduler_type', 'CosineAnnealingWarmRestarts')
    max_epochs = getattr(args, 'max_epochs', 300)
    
    if scheduler_type == 'OneCycleLR':
        # OneCycleLR scheduler - optimal for hybrid CNN-transformer models
        if train_loader is not None:
            # Calculate actual steps per epoch from data loader
            steps_per_epoch = len(train_loader)
            total_steps = max_epochs * steps_per_epoch
            print(f"  📊 OneCycleLR: {steps_per_epoch} steps/epoch × {max_epochs} epochs = {total_steps} total steps")
        else:
            # Fallback: estimate steps per epoch
            total_steps = max_epochs * 1000  # Estimate: ~1000 steps per epoch
            print(f"  📊 OneCycleLR: Estimated {total_steps} total steps (1000 steps/epoch)")
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[learning_rate * 10, learning_rate * 5, learning_rate * 10],  # Peak LRs for each group
            total_steps=total_steps,
            pct_start=0.3,  # 30% warmup
            anneal_strategy='cos',
            div_factor=10,  # Initial LR = max_lr/10
            final_div_factor=100  # Final LR = max_lr/1000
        )
        scheduler_name = "OneCycleLR (Peak: 10x, Warmup: 30%, Cosine)"
        
    elif scheduler_type == 'ReduceLROnPlateau':
        # ReduceLROnPlateau scheduler - adaptive based on validation loss
        # Improved: Less aggressive reduction (factor=0.7) and more patience (20 epochs)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,  # Less aggressive reduction (was 0.5)
            patience=20,  # More patience before reducing (was 15)
            min_lr=1e-6,  # Minimum learning rate
            verbose=True  # Print LR reduction messages
        )
        scheduler_name = "ReduceLROnPlateau (factor=0.7, patience=20)"
        
    elif scheduler_type == 'CosineAnnealingLR':
        # CosineAnnealingLR scheduler - smooth decay without restarts
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=1e-6
        )
        scheduler_name = f"CosineAnnealingLR (T_max={max_epochs})"
        
    else:
        # CosineAnnealingWarmRestarts scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,  # Restart every 50 epochs
            T_mult=2,  # Double period after each restart
            eta_min=1e-7
        )
        scheduler_name = "CosineAnnealingWarmRestarts (T_0=50, T_mult=2)"
    
    print("🚀 Hybrid2 Best Practice: Differential Learning Rates")
    print(f"  📊 Encoder LR:     {learning_rate * 0.1:.6f} (10x smaller, {len(encoder_params)} params)")
    print(f"  📊 Bottleneck LR:  {learning_rate * 0.5:.6f} (5x smaller, {len(bottleneck_params)} params)")
    print(f"  📊 Decoder LR:     {learning_rate:.6f} (base LR, {len(decoder_params)} params)")
    print(f"  ⚙️  Scheduler: {scheduler_name}")
    print(f"  ⚙️  Weight decay: Encoder=1e-3, Bottleneck=5e-3, Decoder=1e-2")
    
    return optimizer, scheduler


def run_training_epoch(model, train_loader, ce_loss, focal_loss, dice_loss, optimizer, class_weights, scheduler=None, scheduler_type='CosineAnnealingWarmRestarts'):
    """
    Run one training epoch.
    Matches Network model's pattern: returns loss_dict.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        ce_loss, focal_loss, dice_loss: Loss functions
        optimizer: Optimizer
        class_weights: Class weights for Dice loss (unused but kept for compatibility)
        scheduler: Learning rate scheduler (optional)
        scheduler_type: Type of scheduler (for OneCycleLR step handling)
        
    Returns:
        dict: Dictionary with 'total', 'ce', 'focal', 'dice' losses
    """
    import math
    
    model.train()
    
    epoch_losses = defaultdict(float)
    num_batches = len(train_loader)
    skipped_loss_nan = 0
    skipped_grad_nan = 0
    scheduler_warning_printed = False
    
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
        
        # Forward pass
        predictions = model(images)
        
        # Use unified loss computation
        loss, loss_dict = compute_combined_loss(
            predictions, labels, ce_loss, focal_loss, dice_loss,
            ce_weight=0.25, focal_weight=0.35, dice_weight=0.4
        )
        
        # Check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            skipped_loss_nan += 1
            continue
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        # Check for NaN/Inf gradients before clipping
        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan_grad = True
                    break
        
        if has_nan_grad:
            skipped_grad_nan += 1
            optimizer.zero_grad()
            continue
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Step scheduler for OneCycleLR (step-based scheduler)
        if scheduler is not None and scheduler_type == 'OneCycleLR':
            if hasattr(scheduler, 'total_steps') and scheduler.last_epoch + 1 < scheduler.total_steps:
                scheduler.step()
            elif hasattr(scheduler, 'total_steps') and not scheduler_warning_printed:
                if scheduler.last_epoch + 1 >= scheduler.total_steps:
                    print("⚠️  Scheduler reached max steps, stopping LR updates.")
                    scheduler_warning_printed = True
        
        # Accumulate losses from loss_dict (skip non-numeric values like resolution strings)
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                epoch_losses[key] += value
            elif isinstance(value, torch.Tensor):
                epoch_losses[key] += value.item()
    
    # Print summary only if batches were skipped
    if skipped_loss_nan > 0 or skipped_grad_nan > 0:
        total_skipped = skipped_loss_nan + skipped_grad_nan
        print(f"  ⚠️  Skipped {total_skipped} batches ({skipped_loss_nan} NaN/Inf loss, {skipped_grad_nan} NaN/Inf gradients)")
    
    # Average losses
    if num_batches > 0:
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
    else:
        # No valid batches - return default dict with inf
        epoch_losses['total'] = float('inf')
        epoch_losses['ce'] = float('inf')
        epoch_losses['focal'] = float('inf')
        epoch_losses['dice'] = float('inf')
    
    return epoch_losses


def validate_model(model, val_loader, ce_loss, focal_loss, dice_loss, max_batches=None):
    """
    Validate model on validation set.
    Matches Network model's pattern: uses DataLoader instead of sliding window.
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        ce_loss, focal_loss, dice_loss: Loss functions
        max_batches: Optional limit on number of batches to validate
        
    Returns:
        dict: Dictionary with 'total', 'ce', 'focal', 'dice' losses
    """
    import math
    
    model.eval()
    
    epoch_losses = defaultdict(float)
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            # Extract data
            if isinstance(batch, dict):
                images = batch['image']
                labels = batch['label']
            else:
                images, labels = batch[0], batch[1]
            
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            
            # Forward pass
            predictions = model(images)
            loss, loss_dict = compute_combined_loss(
                predictions, labels, ce_loss, focal_loss, dice_loss,
                ce_weight=0.25, focal_weight=0.35, dice_weight=0.4
            )
            
            # Check for NaN/Inf in loss_dict before accumulating
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        logging.warning(f"NaN/Inf detected in validation {key} loss: {value}")
                        # Skip this batch if NaN/Inf detected
                        continue
                elif isinstance(value, (int, float)):
                    if math.isnan(value) or math.isinf(value):
                        logging.warning(f"NaN/Inf detected in validation {key} loss: {value}")
                        # Skip this batch if NaN/Inf detected
                        continue
            
            # Accumulate losses (only if no NaN/Inf detected)
            for key, value in loss_dict.items():
                # Skip non-numeric values like resolution strings
                if isinstance(value, (int, float)):
                    epoch_losses[key] += value
                elif isinstance(value, torch.Tensor):
                    epoch_losses[key] += value.item()
            
            num_batches += 1
    
    # Average losses
    if num_batches > 0:
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
    else:
        # No valid batches - return default dict with inf
        epoch_losses['total'] = float('inf')
        epoch_losses['ce'] = float('inf')
        epoch_losses['focal'] = float('inf')
        epoch_losses['dice'] = float('inf')
    
    return epoch_losses


def save_best_model(model, epoch, val_loss, best_val_loss, snapshot_path,
                    optimizer=None, scheduler=None):
    """
    Save model + optimizer/scheduler checkpoint if validation loss improved.
    
    Args:
        model: Neural network model
        epoch (int): Current epoch
        val_loss (float): Current validation loss
        best_val_loss (float): Best validation loss so far
        snapshot_path (str): Directory to save models
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        tuple: (best_val_loss, improvement_made)
    """
    improvement_made = False
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        improvement_made = True
        
        best_model_path = os.path.join(snapshot_path, 'best_model_latest.pth')
        
        # Build checkpoint dict
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state': model_state,
            'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
            'best_val_loss': best_val_loss,
        }
        
        # Save checkpoint
        torch.save(checkpoint, best_model_path)
        
        print(f"    ✓ New best checkpoint saved! Val loss: {val_loss:.4f}")
    else:
        print(f"    No improvement (current: {val_loss:.4f}, best: {best_val_loss:.4f})")
    
    return best_val_loss, improvement_made


def trainer_hybrid(args, model, snapshot_path, train_dataset=None, val_dataset=None):
    """
    Main training function for Hybrid models (hybrid1 and hybrid2).
    
    Args:
        args: Command line arguments with training configuration
        model: Neural network model to train (hybrid1 or hybrid2)
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
    print(f"Model: Hybrid2")  # Hardcoded instead of args.model
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
    
    # Print dataset statistics (matching Network model)
    print(f"📊 Dataset Statistics:")
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Validation samples: {len(val_dataset)}")
    print(f"   - Batch size: {args.batch_size * args.n_gpu}")
    print(f"   - Steps per epoch: {len(train_loader)}\n")
    
    # Compute class weights for balanced training
    if hasattr(train_dataset, 'mask_paths'):
        class_weights = compute_class_weights(train_dataset, args.num_classes, smoothing=0.0)
        # Print class weights (matching Network model)
        print(f"📈 Class weights computed with ENS method (smoothing=0.1)")
        print(f"   Final weights: {class_weights.cpu().numpy()}\n")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = torch.ones(args.num_classes, device=device)
    
    # Create loss functions, optimizer, and scheduler
    focal_gamma = getattr(args, 'focal_gamma', 2.0)  # Default 2.0, configurable via --focal_gamma
    ce_loss, focal_loss, dice_loss = create_loss_functions(class_weights, args.num_classes, focal_gamma=focal_gamma)
    optimizer, scheduler = create_optimizer_and_scheduler(model, args.base_lr, args, train_loader)
    
    # Get scheduler type for training loop
    scheduler_type = getattr(args, 'scheduler_type', 'CosineAnnealingWarmRestarts')
    
    # No AMP scaler - Network model doesn't use it
    scaler = None
    
    # Set up TensorBoard logging
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard_logs'))
    
    # Resume from checkpoint if available
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    checkpoint_path = os.path.join(snapshot_path, 'best_model_latest.pth')
    print(f"\n🔍 Checking for checkpoint at: {checkpoint_path}")
    print(f"   Absolute path: {os.path.abspath(checkpoint_path)}")
    print(f"   File exists: {os.path.exists(checkpoint_path)}")
    
    if os.path.exists(checkpoint_path):
        print(f"\n📂 Found checkpoint: {checkpoint_path}")
        print("   Attempting to resume training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model state (critical - must succeed)
            try:
                if isinstance(model, nn.DataParallel):
                    model.module.load_state_dict(checkpoint['model_state'], strict=False)
                else:
                    model.load_state_dict(checkpoint['model_state'], strict=False)
                print("   ✓ Loaded model state")
            except Exception as e:
                print(f"   ⚠️  Failed to load model state: {e}")
                raise  # Re-raise if model loading fails - cannot continue without model
            
            # Load optimizer state (handle parameter group mismatches gracefully)
            if optimizer is not None and 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                    print("   ✓ Loaded optimizer state")
                except Exception as e:
                    print(f"   ⚠️  Could not load optimizer state: {e}")
                    print("   Starting with fresh optimizer state (this is OK if architecture changed)")
            
            # Load scheduler state (handle mismatches gracefully, especially for OneCycleLR)
            # Check if scheduler type matches before loading state
            if scheduler is not None and 'scheduler_state' in checkpoint and checkpoint['scheduler_state'] is not None:
                scheduler_state = checkpoint['scheduler_state']
                current_scheduler_type = type(scheduler).__name__
                
                # Check if scheduler state matches current scheduler type
                # OneCycleLR has 'total_steps' key, CosineAnnealingWarmRestarts has 'T_0', etc.
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
                    try:
                        scheduler.load_state_dict(scheduler_state)
                        print("   ✓ Loaded scheduler state")
                    except Exception as e:
                        print(f"   ⚠️  Could not load scheduler state: {e}")
                        print("   Starting with fresh scheduler state")
                        # Fast-forward scheduler to current epoch if epoch-based
                        if current_scheduler_type in ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR']:
                            for _ in range(checkpoint.get('epoch', 0)):
                                scheduler.step()
                            print(f"   ✓ Fast-forwarded scheduler to epoch {checkpoint.get('epoch', 0)}")
                else:
                    print(f"   ⚠️  Scheduler type mismatch (checkpoint has different scheduler type)")
                    print(f"   Starting with fresh scheduler state")
                    # Fast-forward scheduler to current epoch if epoch-based
                    if current_scheduler_type in ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR']:
                        for _ in range(checkpoint.get('epoch', 0)):
                            scheduler.step()
                        print(f"   ✓ Fast-forwarded scheduler to epoch {checkpoint.get('epoch', 0)}")
            
            # Load training state
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            print(f"   ✓ Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
            print(f"   ✓ Best validation loss: {best_val_loss:.4f}")
            print(f"   ✓ Resuming from epoch {start_epoch}\n")
        except Exception as e:
            print(f"   ⚠️  Failed to load checkpoint: {e}")
            print("   Starting training from scratch\n")
    
    # Training loop
    patience = getattr(args, 'patience', 25)  # Default 25 (matching Network model)
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Early stopping patience: {patience} epochs")
    print(f"Learning rate scheduler: {scheduler_type} (better convergence for transformers)")
    if start_epoch > 0:
        print(f"Resuming from epoch: {start_epoch}")
    print("="*80)
    
    for epoch in range(start_epoch, args.max_epochs):
        print(f"\nEPOCH {epoch+1}/{args.max_epochs}")
        print("-" * 50)
        
        # Training phase
        train_losses = run_training_epoch(model, train_loader, ce_loss, focal_loss, dice_loss, optimizer, class_weights, scheduler, scheduler_type)
        
        # Validation phase
        val_losses = validate_model(model, val_loader, ce_loss, focal_loss, dice_loss)
        
        # Extract total losses for logging
        train_loss = train_losses.get('total', float('inf'))
        val_loss = val_losses.get('total', float('inf'))
        
        # Log results (matching network model pattern)
        for key, value in train_losses.items():
            writer.add_scalar(f'Train/{key}', value, epoch)
        for key, value in val_losses.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Early_Stopping/Patience_Remaining', patience - epochs_without_improvement, epoch)
        writer.add_scalar('Early_Stopping/Epochs_Without_Improvement', epochs_without_improvement, epoch)
        
        # Print epoch summary
        print("Results:")
        print(f"  • Train Loss: {train_loss:.4f}")
        print(f"  • Validation Loss: {val_loss:.4f}")
        print(f"  • Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save periodic checkpoint (every 100 epochs) - useful for recovery and evaluation
        if (epoch + 1) % 100 == 0:
            periodic_checkpoint_path = os.path.join(snapshot_path, f"epoch_{epoch + 1}.pth")
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            periodic_checkpoint = {
                'epoch': epoch,
                'model_state': model_state,
                'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
                'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            if scaler is not None:
                try:
                    periodic_checkpoint['scaler_state'] = scaler.state_dict()
                except Exception:
                    pass
            torch.save(periodic_checkpoint, periodic_checkpoint_path)
            print(f"   💾 Periodic checkpoint saved: epoch_{epoch + 1}.pth")
        
        # Save best model and check for improvement
        best_val_loss, improvement_made = save_best_model(
            model, epoch, val_loss, best_val_loss, snapshot_path,
            optimizer=optimizer, scheduler=scheduler
        )
        
        # Early stopping logic
        if improvement_made:
            epochs_without_improvement = 0
            print(f"    ✓ Improvement detected! Resetting patience counter.")
        else:
            epochs_without_improvement += 1
            print(f"    ⚠ No improvement for {epochs_without_improvement} epochs (patience: {patience}, remaining: {patience - epochs_without_improvement})")
        
        # Step scheduler per epoch (for schedulers that step per epoch, not per batch)
        # OneCycleLR is stepped per batch in run_training_epoch
        if scheduler_type != 'OneCycleLR':
            if scheduler_type == 'ReduceLROnPlateau':
                # ReduceLROnPlateau steps based on validation loss
                import math
                scheduler_val_loss = val_loss if not math.isinf(val_loss) else 1e6
                scheduler.step(scheduler_val_loss)
            else:
                # Other epoch-based schedulers
                scheduler.step()
        
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
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best Val Loss:  {best_val_loss:.4f}")
    # Handle case where training loop didn't execute (resumed from final epoch)
    if start_epoch >= args.max_epochs:
        print(f"Total Epochs:   {start_epoch}")
    else:
        print(f"Total Epochs:   {epoch + 1}")
    print(f"Models Saved:   {snapshot_path}")
    print(f"TensorBoard:    {os.path.join(snapshot_path, 'tensorboard_logs')}")
    print("="*80 + "\n")
    
    writer.close()
    logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")
    
    return "Training Finished!"
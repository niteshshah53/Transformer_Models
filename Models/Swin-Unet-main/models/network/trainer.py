import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from collections import defaultdict
import warnings

# Add common directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))

from utils import DiceLoss, FocalLoss


def setup_logging(output_path):
    """Set up logging to both file and console."""
    log_file = os.path.join(output_path, "training.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File and console handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def compute_class_weights(train_dataset, num_classes, smoothing=0.1):
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
    # beta = 0.9999 for highly imbalanced datasets, 0.99 for moderate imbalance
    beta = 0.9999 if np.min(class_freq[class_freq > 0]) < 0.001 else 0.99
    
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    
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
    
    # Define class names for display
    CLASS_NAMES = {
        6: ['Background', 'Paratext', 'Decoration', 'Main Text', 'Title', 'Chapter Heading'],
        5: ['Background', 'Paratext', 'Decoration', 'Main Text', 'Title'],
        4: ['Background', 'Comment', 'Decoration', 'Main Text']
    }
    
    class_names = CLASS_NAMES.get(num_classes, [f'Class {i}' for i in range(num_classes)])
    
    # Print simplified analysis
    print("\n" + "-"*60)
    print(f"{'Class Name':<20} {'Percentage':<15} {'Weight':<15}")
    print("-"*60)
    for cls in range(num_classes):
        percentage = class_freq[cls] * 100
        weight = weights[cls]
        print(f"{class_names[cls]:<20} {percentage:>6.2f}%       {weight:>6.4f}")
    print("-"*60)
    print(f"Total pixels: {total_pixels:,}")
    print(f"Weight ratio (max/min): {weights.max()/weights.min():.2f}")
    print("="*80 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_data_loaders(train_dataset, val_dataset, batch_size, num_workers, seed, sampler=None):
    """Create training and validation data loaders.

    If `sampler` is provided, it will be used for the training loader and
    `shuffle` will be disabled (as required by PyTorch DataLoader).
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


def create_loss_functions(class_weights, num_classes, focal_gamma=2.0):
    """
    Create properly weighted loss functions.
    Matches hybrid2's pattern.
    
    Args:
        class_weights (torch.Tensor): Per-class weights
        num_classes (int): Number of classes
        focal_gamma (float): Focal loss focusing parameter (default: 2.0)
        
    Returns:
        tuple: (ce_loss, focal_loss, dice_loss)
    """
    # CrossEntropyLoss with class weights (matching hybrid2: label_smoothing=0.1)
    ce_loss = CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Focal loss with moderate gamma (2.0 is standard, matching hybrid2)
    focal_loss = FocalLoss(gamma=focal_gamma, weight=class_weights)
    
    # Dice loss (handles class imbalance internally, matching hybrid2: no weight, no smooth)
    dice_loss = DiceLoss(num_classes)
    
    print(f"✓ Loss functions created: CE (weighted), Focal (γ={focal_gamma}), Dice")
    
    return ce_loss, focal_loss, dice_loss


def compute_combined_loss(predictions, labels, ce_loss, focal_loss, dice_loss, 
                          ce_weight=0.3, focal_weight=0.2, dice_weight=0.5):
    """
    Compute combined loss with support for deep supervision.
    Matches hybrid2's pattern but keeps network model's weight values.
    
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
        
        # Auxiliary losses with exponentially decaying weights (matching hybrid2 pattern)
        aux_weights = [0.4 * (0.8 ** i) for i in range(len(aux_outputs))]
        aux_loss = 0.0
        
        for i, (weight, aux_output) in enumerate(zip(aux_weights, aux_outputs)):
            aux_ce = ce_loss(aux_output, labels)
            aux_focal = focal_loss(aux_output, labels)
            aux_dice = dice_loss(aux_output, labels, softmax=True)
            aux_combined = ce_weight * aux_ce + focal_weight * aux_focal + dice_weight * aux_dice
            aux_loss += weight * aux_combined
            loss_dict[f'aux_{i}'] = (weight * aux_combined).item()
        
        total_loss = main_loss + aux_loss
        loss_dict['total'] = total_loss.item()
        
    else:
        # Single output
        loss_ce = ce_loss(predictions, labels)
        loss_focal = focal_loss(predictions, labels)
        loss_dice = dice_loss(predictions, labels, softmax=True)
        
        total_loss = ce_weight * loss_ce + focal_weight * loss_focal + dice_weight * loss_dice
        
        loss_dict['ce'] = loss_ce.item()
        loss_dict['focal'] = loss_focal.item()
        loss_dict['dice'] = loss_dice.item()
        loss_dict['total'] = total_loss.item()
    
    return total_loss, loss_dict



def create_optimizer_and_scheduler(model, learning_rate, args=None, train_loader=None):
    """
    Create optimizer with differential learning rates and appropriate scheduler.
    
    For baseline Hybrid2 model:
    - Pretrained encoder: 10x smaller LR
    - Bottleneck (2 Swin blocks): 5x smaller LR  
    - Decoder (including encoder_projections): full LR
    - Weight decay for regularization
    """
    encoder_params = []
    bottleneck_params = []
    decoder_params = []
    
    # Separate parameters by component
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'encoder' in name.lower() and 'decoder' not in name.lower():
            # Encoder parameters (not decoder.encoder_projections)
            encoder_params.append(param)
        elif 'bottleneck' in name.lower():
            # Bottleneck: 2 Swin Transformer blocks
            bottleneck_params.append(param)
        else:
            # Decoder parameters (including encoder_projections, decoder blocks, etc.)
            decoder_params.append(param)
    
    # Parameter groups with differential learning rates
    param_groups = [
        {
            'params': encoder_params,
            'lr': learning_rate * 0.1,
            'weight_decay': 1e-4,
            'name': 'encoder'
        },
        {
            'params': bottleneck_params,
            'lr': learning_rate * 0.5,
            'weight_decay': 5e-4,
            'name': 'bottleneck'
        },
        {
            'params': decoder_params,
            'lr': learning_rate,
            'weight_decay': 1e-3,
            'name': 'decoder'
        }
    ]
    
    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    # AdamW optimizer
    optimizer = optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Choose scheduler
    scheduler_type = getattr(args, 'scheduler_type', 'CosineAnnealingWarmRestarts')
    max_epochs = getattr(args, 'max_epochs', 300)
    warmup_epochs = getattr(args, 'warmup_epochs', 10)
    
    if scheduler_type == 'OneCycleLR' and train_loader is not None:
        steps_per_epoch = len(train_loader)
        total_steps = max_epochs * steps_per_epoch
        
        # Get max LRs for each group
        max_lrs = [g['lr'] * 10 for g in param_groups]
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=warmup_epochs / max_epochs,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100
        )
        scheduler_name = f"OneCycleLR (warmup: {warmup_epochs} epochs)"
        
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=1e-7
        )
        scheduler_name = f"CosineAnnealingLR (T_max={max_epochs - warmup_epochs})"
        
    else:  # Default: CosineAnnealingWarmRestarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,
            T_mult=2,
            eta_min=1e-7
        )
        scheduler_name = "CosineAnnealingWarmRestarts (T_0=50)"
    
    # Print configuration
    print("\n" + "="*80)
    print("OPTIMIZER CONFIGURATION")
    print("="*80)
    for group in param_groups:
        num_params = sum(p.numel() for p in group['params'])
        print(f"{group['name'].capitalize():12}: LR={group['lr']:.6f}, "
              f"WD={group['weight_decay']:.6f}, Params={num_params:,}")
    print(f"Scheduler:   {scheduler_name}")
    print("="*80 + "\n")
    
    return optimizer, scheduler


def run_training_epoch(model, train_loader, ce_loss, focal_loss, dice_loss, 
                       optimizer, scheduler, scheduler_type='OneCycleLR'):
    """Run one training epoch with gradient clipping.
    Matches hybrid2's pattern: returns loss_dict.

    scheduler_type: Type of scheduler - determines when to step
        - 'OneCycleLR': Step per batch
        - Others: Step per epoch (caller handles)
    """
    import math
    
    model.train()
    
    epoch_losses = defaultdict(float)
    num_batches = len(train_loader)
    skipped_loss_nan = 0
    skipped_grad_nan = 0
    scheduler_warning_printed = False
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(train_loader):
        # Get data
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
            ce_weight=0.3, focal_weight=0.2, dice_weight=0.5
        )
        
        # Check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            skipped_loss_nan += 1
            continue
        
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
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # Scheduler step per batch (OneCycleLR expects step every batch)
        if scheduler_type == 'OneCycleLR':
            if hasattr(scheduler, 'total_steps') and scheduler.last_epoch + 1 < scheduler.total_steps:
                scheduler.step()
            elif hasattr(scheduler, 'total_steps') and not scheduler_warning_printed:
                if scheduler.last_epoch + 1 >= scheduler.total_steps:
                    print("⚠️  Scheduler reached max steps, stopping LR updates.")
                    scheduler_warning_printed = True
        
        # Accumulate losses
        for key, value in loss_dict.items():
            epoch_losses[key] += value
    
    # Print summary only if batches were skipped
    if skipped_loss_nan > 0 or skipped_grad_nan > 0:
        total_skipped = skipped_loss_nan + skipped_grad_nan
        print(f"  ⚠️  Skipped {total_skipped} batches ({skipped_loss_nan} NaN/Inf loss, {skipped_grad_nan} NaN/Inf gradients)")
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


def validate_model(model, val_loader, ce_loss, focal_loss, dice_loss, max_batches=None):
    """
    Validate model on validation set.
    Matches hybrid2's pattern: returns loss_dict with NaN checking.
    
    Args:
        max_batches: Optional limit on number of batches to validate
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
            _, loss_dict = compute_combined_loss(
                predictions, labels, ce_loss, focal_loss, dice_loss,
                ce_weight=0.3, focal_weight=0.2, dice_weight=0.5
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
                epoch_losses[key] += value
            
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
    """Save model + optimizer/scheduler checkpoint if validation loss improved.

    Returns (best_val_loss, improvement_made)
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


def trainer_synapse(args, model, snapshot_path, train_dataset=None, val_dataset=None):
    """
    Main training function optimized based on ablation study.
    
    Best configuration (F1=0.6919, IoU=0.5831):
    - BL (Baseline encoder-decoder)
    - ASH (Alternative Segmentation Head: Conv3x3-ReLU-Conv1x1)
    - DS (Deep Supervision with auxiliary outputs)
    - AFF (Attention Feature Fusion)
    - Bo (Bottleneck with 2 Swin blocks)
    - FL (Focal Loss in combination)
    """
    # Set random seeds for reproducibility
    import random
    seed = getattr(args, 'seed', 1234)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Setup
    logger = setup_logging(snapshot_path)
    patience = getattr(args, 'patience', 25)
    
    # Get actual model configuration
    if isinstance(model, nn.DataParallel):
        model_obj = model.module.model
    else:
        model_obj = model.model
    
    use_deep_supervision = getattr(model_obj, 'use_deep_supervision', False)
    fusion_method = getattr(model_obj, 'fusion_method', 'simple')
    use_bottleneck = getattr(model_obj, 'use_bottleneck', False)
    use_multiscale_agg = getattr(model_obj, 'use_multiscale_agg', False)
    use_baseline = getattr(args, 'use_baseline', False)
    
    # Build configuration string dynamically
    config_parts = []
    if use_baseline:
        config_parts.append("BL")  # Baseline
    if use_bottleneck:
        config_parts.append("Bo")  # Bottleneck
    if use_deep_supervision:
        config_parts.append("DS")  # Deep Supervision
    if fusion_method == 'fourier':
        config_parts.append("FF")  # Fourier Fusion
    elif fusion_method == 'smart':
        config_parts.append("AFF")  # Attention Feature Fusion (Smart Skip)
    if use_multiscale_agg:
        config_parts.append("MSA")  # Multi-Scale Aggregation
    config_parts.append("FL")  # Focal Loss (always used)
    
    config_str = " + ".join(config_parts) if config_parts else "BASELINE"
    
    # Print configuration
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Manuscript: {getattr(args, 'manuscript', 'N/A')}")
    print(f"Model: CNN-Transformer (EfficientNet-B4 + Swin-UNet Decoder)")
    print(f"Configuration: {config_str}")
    print(f"  • Baseline: {'✓' if use_baseline else '✗'}")
    print(f"  • Bottleneck: {'✓' if use_bottleneck else '✗'}")
    print(f"  • Deep Supervision: {'✓' if use_deep_supervision else '✗'}")
    print(f"  • Fusion Method: {fusion_method.upper()}")
    print(f"  • Multi-Scale Aggregation: {'✓' if use_multiscale_agg else '✗'}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Epochs: {args.max_epochs}")
    print(f"Learning Rate: {args.base_lr}")
    print(f"Number of Classes: {args.num_classes}")
    print(f"Early Stopping Patience: {patience} epochs")
    print(f"Output Directory: {snapshot_path}")
    print("="*80 + "\n")
    
    # Multi-GPU setup
    if args.n_gpu > 1:
        print(f"🖥️  Using {args.n_gpu} GPUs for training\n")
        model = nn.DataParallel(model)
    
    # Freeze encoder if requested
    if getattr(args, 'freeze_encoder', False):
        print(f"🔒 Freezing encoder for training")
        if isinstance(model, nn.DataParallel):
            model.module.model.freeze_encoder()
        else:
            model.model.freeze_encoder()
        
        freeze_epochs = getattr(args, 'freeze_epochs', 0)
        if freeze_epochs > 0:
            print(f"   Will unfreeze after {freeze_epochs} epochs")
        else:
            print(f"   Encoder will remain frozen for entire training")
        print()
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset,
        args.batch_size * args.n_gpu,
        args.num_workers,
        args.seed
    )
    
    print(f"📊 Dataset Statistics:")
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Validation samples: {len(val_dataset)}")
    print(f"   - Batch size: {args.batch_size * args.n_gpu}")
    print(f"   - Steps per epoch: {len(train_loader)}\n")
    
    # Compute class weights
    if hasattr(train_dataset, 'mask_paths'):
        class_weights = compute_class_weights(train_dataset, args.num_classes)
        # compute_class_weights already applies a bounded rarity-based boost
        print(f"📈 Class weights computed with rarity-based boosting (mean scaled)")
        print(f"   Final weights: {class_weights.cpu().numpy()}\n")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = torch.ones(args.num_classes, device=device)
    
    # Create loss functions, optimizer, scheduler
    ce_loss, focal_loss, dice_loss = create_loss_functions(class_weights, args.num_classes)
    
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, args.base_lr, args, train_loader
    )
    scheduler_type = getattr(args, 'scheduler_type', 'CosineAnnealingWarmRestarts')
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard_logs'))
    
    # Resume from checkpoint if available
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    encoder_unfrozen_epoch = None
    
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
                    # Special handling for OneCycleLR - cannot resume directly, need to recreate
                    if current_scheduler_type == 'OneCycleLR':
                        print(f"   ⚠️  OneCycleLR scheduler cannot be resumed directly")
                        print(f"   Will recreate scheduler and fast-forward to current step")
                        # OneCycleLR will be recreated later in the training loop if needed
                        # For now, just note that we need to fast-forward
                        checkpoint_epoch = checkpoint.get('epoch', 0)
                        if checkpoint_epoch > 0:
                            steps_per_epoch = len(train_loader)
                            current_step = checkpoint_epoch * steps_per_epoch
                            print(f"   Note: Will resume OneCycleLR from step {current_step}")
                    else:
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
            
            # Determine encoder unfrozen epoch if encoder was unfrozen
            if getattr(args, 'freeze_encoder', False) and getattr(args, 'freeze_epochs', 0) > 0:
                if start_epoch > args.freeze_epochs:
                    encoder_unfrozen_epoch = args.freeze_epochs
            
            print(f"   ✓ Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
            print(f"   ✓ Best validation loss: {best_val_loss:.4f}")
            print(f"   ✓ Resuming from epoch {start_epoch}\n")
        except Exception as e:
            import traceback
            print(f"   ⚠️  Failed to load checkpoint: {e}")
            print(f"   Error type: {type(e).__name__}")
            print("   Traceback:")
            traceback.print_exc()
            print("   Starting training from scratch\n")
    else:
        print("   No checkpoint found - starting training from scratch\n")
    
    # Training loop
    print("="*80)
    print("🚀 STARTING TRAINING")
    print("="*80)
    ds_text = " (with Deep Supervision)" if use_deep_supervision else ""
    print(f"Loss: 0.3*CE + 0.2*Focal + 0.5*Dice{ds_text}")
    print(f"Early stopping: {patience} epochs patience")
    if start_epoch > 0:
        print(f"Resuming from epoch: {start_epoch}")
    print("="*80 + "\n")
    
    for epoch in range(start_epoch, args.max_epochs):
        try:
            print(f"\nEPOCH {epoch + 1}/{args.max_epochs}")
            print("-" * 50)
            sys.stdout.flush()  # Force flush for background jobs
            
            # Train
            train_losses = run_training_epoch(
                model, train_loader, ce_loss, focal_loss, dice_loss,
                optimizer, scheduler, scheduler_type=scheduler_type
            )
            
            # Check for NaN/Inf in training losses (matching hybrid2 pattern)
            import math
            for key, value in train_losses.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        raise ValueError(f"NaN/Inf detected in training {key} loss: {value}")
                elif isinstance(value, (int, float)):
                    if math.isnan(value) or math.isinf(value):
                        raise ValueError(f"NaN/Inf detected in training {key} loss: {value}")
            
            # Validate
            val_losses = validate_model(model, val_loader, ce_loss, focal_loss, dice_loss)
            
            # Check for NaN/Inf in validation losses (matching hybrid2 pattern)
            for key, value in val_losses.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        raise ValueError(f"NaN/Inf detected in validation {key} loss: {value}")
                elif isinstance(value, (int, float)):
                    if math.isnan(value) or math.isinf(value):
                        raise ValueError(f"NaN/Inf detected in validation {key} loss: {value}")
            
            val_loss = val_losses.get('total', float('inf'))
            train_loss = train_losses.get('total', float('inf'))
            
            # Logging (matching hybrid2 pattern)
            for key, value in train_losses.items():
                writer.add_scalar(f'Train/{key}', value, epoch)
            for key, value in val_losses.items():
                writer.add_scalar(f'Val/{key}', value, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            # Step scheduler per epoch (for schedulers that step per epoch, not per batch)
            # OneCycleLR is stepped per batch in run_training_epoch
            if scheduler_type != 'OneCycleLR':
                if scheduler_type == 'ReduceLROnPlateau':
                    # ReduceLROnPlateau steps based on validation loss
                    # Use a large value if val_loss is inf to avoid issues
                    scheduler_val_loss = val_loss if not math.isinf(val_loss) else 1e6
                    scheduler.step(scheduler_val_loss)
                else:
                    # Other epoch-based schedulers
                    scheduler.step()
            
            # Print summary in clean format (matching hybrid2)
            print("Results:")
            print(f"  • Train Loss: {train_loss:.4f}")
            print(f"  • Validation Loss: {val_loss:.4f}")
            print(f"  • Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Unfreeze encoder if requested and epoch reached
            if getattr(args, 'freeze_encoder', False) and getattr(args, 'freeze_epochs', 0) > 0:
                if epoch + 1 == args.freeze_epochs:
                    print(f"\n🔓 Unfreezing encoder at epoch {epoch + 1}")
                    if isinstance(model, nn.DataParallel):
                        model.module.model.unfreeze_encoder()
                    else:
                        model.model.unfreeze_encoder()
                    
                    # CRITICAL: Reconfigure optimizer with differential learning rates
                    # Encoder needs much lower LR to prevent gradient explosion
                    # When encoder is frozen, its params aren't in optimizer, so we need to recreate it
                    encoder_lr_factor = getattr(args, 'encoder_lr_factor', 0.1)
                    encoder_unfrozen_epoch = epoch + 1  # Track when encoder was unfrozen
                    print(f"🔄 Reconfiguring optimizer with differential learning rates")
                    print(f"   - Encoder LR factor: {encoder_lr_factor:.4f}x (encoder LR will be {args.base_lr * encoder_lr_factor:.6f})")
                    print(f"   - Encoder LR will decay: 0.95^epochs_since_unfreeze")
                    
                    # Save current optimizer state (for decoder params)
                    optimizer_state = optimizer.state_dict()
                    
                    # Recreate optimizer with all parameters (now including encoder)
                    encoder_params = []
                    bottleneck_params = []
                    decoder_params = []

                    for name, param in model.named_parameters():
                        if not param.requires_grad:
                            continue
                        lname = name.lower()
                        is_encoder = 'encoder' in lname or 'adapter' in lname or 'streaming_proj' in lname or 'feature_adapters' in lname
                        is_bottleneck = 'bottleneck' in lname
                        
                        if is_encoder:
                            encoder_params.append(param)
                        elif is_bottleneck:
                            bottleneck_params.append(param)
                        else:
                            decoder_params.append(param)

                    param_groups = []
                    if encoder_params:
                        param_groups.append({
                            'params': encoder_params, 
                            'lr': args.base_lr * encoder_lr_factor,
                            'weight_decay': 1e-4,
                            'name': 'encoder'
                        })
                    if bottleneck_params:
                        param_groups.append({
                            'params': bottleneck_params, 
                            'lr': args.base_lr * encoder_lr_factor, # Bottleneck uses encoder LR
                            'weight_decay': 5e-4,
                            'name': 'bottleneck'
                        })
                    if decoder_params:
                        param_groups.append({
                            'params': decoder_params, 
                            'lr': args.base_lr,
                            'weight_decay': 1e-3,
                            'name': 'decoder'
                        })

                    # Create new optimizer
                    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))
                    
                    # Try to load optimizer state (decoder params should match)
                    try:
                        optimizer.load_state_dict(optimizer_state)
                        print("   ✓ Preserved optimizer state for decoder parameters")
                    except Exception as e:
                        print(f"   ⚠️  Could not fully load optimizer state: {e}")
                        print("   Starting fresh optimizer state (this is OK for encoder)")
                    
                    # Recreate scheduler for new optimizer structure (matching hybrid2 pattern)
                    # OneCycleLR needs to be recreated when parameter groups change
                    scheduler_type = getattr(args, 'scheduler_type', 'CosineAnnealingWarmRestarts')
                    warmup_epochs = getattr(args, 'warmup_epochs', 10)
                    steps_per_epoch = len(train_loader)
                    
                    if scheduler_type == 'OneCycleLR':
                        # Track current scheduler step to maintain continuity
                        # OneCycleLR.last_epoch tracks the number of steps taken (since it steps per batch)
                        # At epoch 30, we've completed 30 epochs, so steps = 30 * steps_per_epoch
                        current_step = scheduler.last_epoch if hasattr(scheduler, 'last_epoch') else (epoch * steps_per_epoch)
                        total_steps = args.max_epochs * steps_per_epoch
                        
                        # Get max LRs for each group (matching hybrid2: max_lr = base_lr * 10)
                        max_lrs = [g['lr'] * 10 for g in param_groups]
                        
                        scheduler = optim.lr_scheduler.OneCycleLR(
                            optimizer,
                            max_lr=max_lrs,
                            total_steps=total_steps,
                            pct_start=warmup_epochs / args.max_epochs,
                            anneal_strategy='cos',
                            div_factor=10,
                            final_div_factor=100
                        )
                        
                        # Fast-forward to current step
                        if current_step > 0:
                            scheduler.last_epoch = current_step - 1
                            scheduler.step()
                        print(f"   ✓ Recreated OneCycleLR scheduler (resumed from step {current_step}/{total_steps})")
                    elif scheduler_type == 'CosineAnnealingLR':
                        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=args.max_epochs - warmup_epochs,
                            eta_min=1e-7
                        )
                        # Fast-forward to current epoch
                        for _ in range(epoch):
                            scheduler.step()
                        print(f"   ✓ Recreated CosineAnnealingLR scheduler (fast-forwarded to epoch {epoch})")
                    else:  # Default: CosineAnnealingWarmRestarts
                        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                            optimizer,
                            T_0=50,
                            T_mult=2,
                            eta_min=1e-7
                        )
                        # Fast-forward to current epoch
                        for _ in range(epoch):
                            scheduler.step()
                        print(f"   ✓ Recreated CosineAnnealingWarmRestarts scheduler (fast-forwarded to epoch {epoch})")
                    
                    print()
                elif encoder_unfrozen_epoch is not None:
                    # Encoder was unfrozen in a previous epoch - decay encoder LR factor
                    epochs_since_unfreeze = (epoch + 1) - encoder_unfrozen_epoch
                    base_encoder_lr_factor = getattr(args, 'encoder_lr_factor', 0.1)
                    encoder_lr_factor = base_encoder_lr_factor * (0.95 ** epochs_since_unfreeze)
                    
                    # Update encoder parameter groups with decayed learning rate
                    encoder_lr = args.base_lr * encoder_lr_factor
                    for param_group in optimizer.param_groups:
                        if 'encoder' in param_group.get('name', ''):
                            param_group['lr'] = encoder_lr
                    
                    # Log encoder LR decay (only every 10 epochs to avoid clutter)
                    if epochs_since_unfreeze % 10 == 0 or epochs_since_unfreeze == 1:
                        print(f"   🔄 Encoder LR decay: {encoder_lr_factor:.4f}x (epochs since unfreeze: {epochs_since_unfreeze})")
            
            # Save periodic checkpoint (every 100 epochs) - useful for recovery and evaluation
            # Sometimes the best model by loss isn't best by Dice/IoU
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
                torch.save(periodic_checkpoint, periodic_checkpoint_path)
                print(f"   💾 Periodic checkpoint: epoch_{epoch + 1}.pth")
                sys.stdout.flush()  # Force flush for background jobs
            
            # Save checkpoint (matching hybrid2 pattern)
            # Call save_best_model BEFORE updating best_val_loss to ensure checkpoint is saved
            best_val_loss, improvement_made = save_best_model(
                model, epoch, val_loss, best_val_loss, snapshot_path,
                optimizer=optimizer, scheduler=scheduler
            )
            
            # Update tracking variables based on improvement
            if improvement_made:
                epochs_without_improvement = 0
                print(f"    ✓ Improvement detected! Resetting patience counter.")
            else:
                epochs_without_improvement += 1
                remaining_patience = patience - epochs_without_improvement
                print(f"    ⚠ No improvement for {epochs_without_improvement} epochs (patience: {patience}, remaining: {remaining_patience})")
            
            sys.stdout.flush()  # Force flush for background jobs
            
            # Check early stopping
            if epochs_without_improvement >= patience:
                print("\n" + "="*80)
                print("EARLY STOPPING TRIGGERED")
                print("="*80)
                print(f"No improvement for {patience} consecutive epochs")
                print(f"Best validation loss: {best_val_loss:.4f}")
                print("="*80 + "\n")
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n❌ CUDA OUT OF MEMORY at epoch {epoch + 1}")
                print("Attempting to recover...")
                torch.cuda.empty_cache()
                # Save emergency checkpoint
                emergency_path = os.path.join(snapshot_path, f"emergency_epoch_{epoch + 1}.pth")
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state': model_state,
                    'best_val_loss': best_val_loss,
                }, emergency_path)
                print(f"Emergency checkpoint saved: {emergency_path}")
                logger.error(f"CUDA OOM at epoch {epoch + 1}: {e}")
                raise
            else:
                print(f"\n❌ RuntimeError at epoch {epoch + 1}: {e}")
                logger.error(f"RuntimeError at epoch {epoch + 1}: {e}", exc_info=True)
                raise
        except Exception as e:
            print(f"\n❌ ERROR at epoch {epoch + 1}: {type(e).__name__}: {e}")
            logger.error(f"Error at epoch {epoch + 1}: {e}", exc_info=True)
            # Save emergency checkpoint before crashing
            emergency_path = os.path.join(snapshot_path, f"emergency_epoch_{epoch + 1}.pth")
            try:
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state': model_state,
                    'best_val_loss': best_val_loss,
                }, emergency_path)
                print(f"Emergency checkpoint saved: {emergency_path}")
            except:
                print("Failed to save emergency checkpoint")
            raise
    
    # Training complete (matching hybrid2 format)
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
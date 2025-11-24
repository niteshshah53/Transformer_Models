#!/usr/bin/env python3
"""
Hybrid Model Training Script
Historical Document Segmentation using Hybrid EfficientNet-Swin Transformer
"""

import argparse
import os
import random
import sys
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Add common directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import training modules
from trainer import trainer_hybrid

# Import dataset classes
from datasets.dataset_udiadsbib import UDiadsBibDataset
try:
    from datasets.dataset_divahisdb import DivaHisDBDataset
    DIVAHISDB_AVAILABLE = True
except ImportError:
    DivaHisDBDataset = None
    DIVAHISDB_AVAILABLE = False


def get_model(args, config=None):
    """
    Create and initialize the Hybrid2 model.
    
    Args:
        args: Command line arguments containing model parameters
        config: Configuration object (not used for Hybrid2)
        
    Returns:
        torch.nn.Module: Initialized Hybrid2 model ready for training
    """
    use_baseline = getattr(args, 'use_baseline', False)
    decoder_type = getattr(args, 'decoder', 'simple')
    
    # Simplified logic:
    # - If --use_baseline is used without --decoder → decoder='simple' (default)
    # - If --use_baseline is used with --decoder → use specified decoder
    # - If --decoder is used without --use_baseline → error (caught in validate_arguments)
    if use_baseline:
        # use_baseline is set, use specified decoder (or 'simple' if not specified)
        decoder_type = decoder_type  # Already set from args
    else:
        # No use_baseline flag → error (should be caught in validate, but handle gracefully)
        print("ERROR: --use_baseline flag is required")
        print("Usage: --use_baseline [--decoder simple|EfficientNet-B4|ResNet50]")
        raise ValueError("--use_baseline flag is required")
    
    # Handle batch norm flag (if use_batchnorm is set, disable groupnorm)
    use_groupnorm_value = getattr(args, 'use_groupnorm', True)
    if getattr(args, 'use_batchnorm', False):
        use_groupnorm_value = False
    
    # Always use baseline with configurable decoder (flags control enhancements)
    print("=" * 80)
    print(f"🚀 Loading Hybrid2 with {decoder_type} Decoder")
    print("=" * 80)
    from hybrid2.model import create_hybrid2_baseline
    model = create_hybrid2_baseline(
        num_classes=args.num_classes,
        img_size=args.img_size,
        decoder=decoder_type,
        efficientnet_variant=getattr(args, 'efficientnet_variant', 'b4'),
        use_deep_supervision=getattr(args, 'use_deep_supervision', False),
        use_cbam=getattr(args, 'use_cbam', False),
        use_smart_skip=getattr(args, 'use_smart_skip', False),
        use_cross_attn=getattr(args, 'use_cross_attn', False),
        use_multiscale_agg=getattr(args, 'use_multiscale_agg', False),
        use_groupnorm=use_groupnorm_value,
        use_pos_embed=getattr(args, 'use_pos_embed', True)
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to CUDA")
    else:
        print("CUDA not available, using CPU")
    
    return model


def setup_datasets(args):
    """
    Create training and validation datasets based on the specified dataset type.
    
    Args:
        args: Command line arguments containing dataset configuration
        
    Returns:
        tuple: (train_dataset, val_dataset) - PyTorch datasets for training and validation
    """
    if args.dataset == 'UDIADS_BIB':
        print("Setting up U-DIADS-Bib dataset...")
        
        # Determine number of classes based on manuscript
        if args.manuscript in ['Syr341FS', 'Syr341']:
            num_classes = 5
            print("Detected Syriaque341 manuscript: using 5 classes (no Chapter Headings)")
        else:
            num_classes = 6
            print(f"Using 6 classes for manuscript: {args.manuscript}")
        
        # Update args with correct number of classes
        args.num_classes = num_classes
        
        # Create datasets
        use_class_aware_aug = getattr(args, 'use_class_aware_aug', False)
        if use_class_aware_aug:
            print("✓ Class-aware augmentation enabled (stronger augmentation for rare classes)")
            if num_classes == 5:
                print("  Rare classes: Paratext, Decoration, Title")
            else:  # num_classes == 6
                print("  Rare classes: Paratext, Decoration, Title, Chapter Headings")
        
        train_dataset = UDiadsBibDataset(
            root_dir=args.udiadsbib_root,
            split='training',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            manuscript=args.manuscript,
            model_type='hybrid2',  # Hardcoded since we only support hybrid2
            num_classes=num_classes,
            use_class_aware_aug=use_class_aware_aug
        )
        
        val_dataset = UDiadsBibDataset(
            root_dir=args.udiadsbib_root,
            split='validation',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            manuscript=args.manuscript,
            model_type='hybrid2',  # Hardcoded since we only support hybrid2
            num_classes=num_classes,
            use_class_aware_aug=False  # Never use augmentation for validation
        )
        
    elif args.dataset == 'DIVAHISDB' and DIVAHISDB_AVAILABLE:
        print("Setting up DivaHisDB dataset...")
        args.num_classes = 4
        
        train_dataset = DivaHisDBDataset(
            root_dir=args.divahisdb_root,
            split='training',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            manuscript=args.manuscript,
            model_type='hybrid2'  # Hardcoded since we only support hybrid2
        )
        
        val_dataset = DivaHisDBDataset(
            root_dir=args.divahisdb_root,
            split='validation',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            manuscript=args.manuscript,
            model_type='hybrid2'  # Hardcoded since we only support hybrid2
        )
        
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    return train_dataset, val_dataset


def validate_arguments(args):
    """
    Validate command line arguments and check for required files.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        SystemExit: If validation fails
    """
    # Check if dataset root exists
    if args.dataset == 'UDIADS_BIB' and not os.path.exists(args.udiadsbib_root):
        print(f"ERROR: UDIADS_BIB root directory not found: {args.udiadsbib_root}")
        sys.exit(1)
    
    if args.dataset == 'DIVAHISDB' and not os.path.exists(args.divahisdb_root):
        print(f"ERROR: DIVAHISDB root directory not found: {args.divahisdb_root}")
        sys.exit(1)
    
    # Check if output directory is writable
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate decoder and use_baseline flag usage
    use_baseline = getattr(args, 'use_baseline', False)
    decoder_type = getattr(args, 'decoder', 'simple')
    
    # Rule 1: If decoder is specified (and not 'simple'), must use --use_baseline
    if decoder_type != 'simple' and not use_baseline:
        print(f"ERROR: --decoder {decoder_type} requires --use_baseline flag")
        print(f"Usage: --use_baseline --decoder {decoder_type}")
        sys.exit(1)
    
    # Rule 2: Enhancement flags can only be used with --use_baseline
    enhancement_flags = [
        ('use_deep_supervision', getattr(args, 'use_deep_supervision', False)),
        ('use_cbam', getattr(args, 'use_cbam', False)),
        ('use_smart_skip', getattr(args, 'use_smart_skip', False)),
        ('use_cross_attn', getattr(args, 'use_cross_attn', False)),
        ('use_multiscale_agg', getattr(args, 'use_multiscale_agg', False)),
    ]
    
    used_enhancement_flags = [name for name, used in enhancement_flags if used]
    if used_enhancement_flags and not use_baseline:
        print(f"ERROR: Enhancement flags {used_enhancement_flags} require --use_baseline flag")
        print(f"Usage: --use_baseline {' '.join([f'--{flag}' for flag in used_enhancement_flags])}")
        sys.exit(1)
    
    # Rule 3: If --use_baseline is used without --decoder, set decoder='simple' automatically
    if use_baseline and decoder_type == 'simple':
        # This is the default case - decoder='simple' is already set
        pass
    
    print("All arguments validated successfully!")


def parse_arguments():
    """
    Parse command line arguments for Hybrid model training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Hybrid2 Model Training for Historical Document Segmentation')
    
    # Model configuration
    # Remove this line:
    # parser.add_argument('--model', type=str, default='hybrid2', choices=['hybrid2'],
    #                    help='Model type: hybrid2 (Swin-EfficientNet)')
    
    # Hybrid2 model variants
    parser.add_argument('--use_baseline', action='store_true', default=False,
                       help='Use baseline Hybrid2 (required). If used alone, defaults to simple decoder. Must be used with --decoder for other decoders.')
    parser.add_argument('--decoder', type=str, default='simple',
                       choices=['simple', 'EfficientNet-B4', 'ResNet50'],
                       help='Decoder type: simple (default when --use_baseline is used alone), EfficientNet-B4, or ResNet50 (requires --use_baseline)')
    parser.add_argument('--efficientnet_variant', type=str, default='b4', choices=['b0', 'b4'],
                       help='EfficientNet variant for simple decoder (b0, b4). Only used when --decoder simple')
    
    # Remove these lines:
    # parser.add_argument('--use_transunet', action='store_true', default=False, ...)
    # parser.add_argument('--use_efficientnet', action='store_true', default=False, ...)
    # parser.add_argument('--use_enhanced', action='store_true', default=False, ...)
    
    # Hybrid2 baseline enhancement flags (only used with --use_baseline)
    parser.add_argument('--use_deep_supervision', action='store_true', default=False,
                       help='Enable deep supervision (auxiliary outputs)')
    parser.add_argument('--use_cbam', action='store_true', default=False,
                       help='Enable CBAM attention modules')
    parser.add_argument('--use_smart_skip', action='store_true', default=False,
                       help='Use smart skip connections (attention-based) instead of simple concatenation')
    parser.add_argument('--use_cross_attn', action='store_true', default=False,
                       help='Enable cross-attention bottleneck')
    parser.add_argument('--use_multiscale_agg', action='store_true', default=False,
                       help='Enable multi-scale aggregation')
    parser.add_argument('--use_groupnorm', action='store_true', default=True,
                       help='Use GroupNorm instead of BatchNorm (default: True)')
    parser.add_argument('--use_batchnorm', action='store_true', default=False,
                       help='Use BatchNorm instead of GroupNorm (overrides --use_groupnorm)')
    parser.add_argument('--use_pos_embed', action='store_true', default=True,
                       help='Enable positional embeddings (default: True, matching SwinUnet pattern)')
    parser.add_argument('--no_pos_embed', dest='use_pos_embed', action='store_false',
                       help='Disable positional embeddings')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    
    # Training enhancements (matching Network model)
    parser.add_argument('--use_balanced_sampler', action='store_true', default=False,
                       help='Use balanced sampler to oversample images containing rare classes')
    parser.add_argument('--use_class_aware_aug', action='store_true', default=False,
                       help='Use class-aware augmentation (stronger augmentation for rare classes)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter (default: 2.0, Network model uses 3.0)')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='UDIADS_BIB', 
                       choices=['UDIADS_BIB', 'DIVAHISDB'], help='Dataset to use')
    parser.add_argument('--udiadsbib_root', type=str, default='U-DIADS-Bib-FS_patched',
                       help='Root directory for UDIADS_BIB dataset')
    parser.add_argument('--divahisdb_root', type=str, default='DIVAHISDB',
                       help='Root directory for DIVAHISDB dataset')
    parser.add_argument('--manuscript', type=str, default='Syr341FS',
                       help='Manuscript to train on')
    parser.add_argument('--use_patched_data', action='store_true', default=True,
                       help='Use pre-generated patches')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=300, help='Maximum number of epochs')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='Base learning rate')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--scheduler_type', type=str, default='OneCycleLR',
                       choices=['CosineAnnealingWarmRestarts', 'OneCycleLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'],
                       help='Learning rate scheduler type')
    parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    
    # Additional options for config override
    parser.add_argument('--opts', nargs='*', default=None, help='Additional options to override config')
    parser.add_argument('--zip', action='store_true', help='Use zipped dataset instead of folder dataset')
    parser.add_argument('--cache_mode', type=str, default='part', choices=['no', 'full', 'part'], help='Cache mode')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--accumulation_steps', type=int, help='Gradient accumulation steps')
    parser.add_argument('--use_checkpoint', action='store_true', help='Whether to use gradient checkpointing to save memory')
    parser.add_argument('--amp_opt_level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='Mixed precision opt level')
    parser.add_argument('--tag', type=str, help='Tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    
    return parser.parse_args()


def main():
    """
    Main training function for Hybrid model.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    validate_arguments(args)
    
    # Set random seed for reproducibility
    print(f"Setting random seed to {args.seed} for reproducible training...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    # Set up datasets
    train_dataset, val_dataset = setup_datasets(args)
    
    # Create balanced sampler if requested
    if getattr(args, 'use_balanced_sampler', False):
        from trainer import create_balanced_sampler
        balanced_sampler = create_balanced_sampler(train_dataset, args.num_classes)
        args.balanced_sampler = balanced_sampler
        if balanced_sampler is not None:
            print("✓ Balanced sampler enabled (oversampling rare classes)")
        else:
            print("⚠️  Balanced sampler requested but could not be created (using default shuffling)")
            args.balanced_sampler = None
    else:
        args.balanced_sampler = None
    
    # Create model (no config needed for Hybrid)
    model = get_model(args, config=None)
    print("Model created successfully with {} classes".format(args.num_classes))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Debug: Verify output_dir is correct (should not include dataset name)
    print(f"DEBUG: args.output_dir = {args.output_dir}")
    
    # Start training
    print("\n=== Starting Training ===")
    print("Dataset: {}".format(args.dataset))
    print("Model: Hybrid2")  # Hardcoded
    print("Batch size: {}".format(args.batch_size))
    print("Max epochs: {}".format(args.max_epochs))
    print("Learning rate: {}".format(args.base_lr))
    print()
    
    # Run training with SwinUnet approach
    # Use args.output_dir directly - it should already be the correct path
    result = trainer_hybrid(args, model, args.output_dir, train_dataset, val_dataset)
    
    print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
    print(result)


if __name__ == '__main__':
    main()
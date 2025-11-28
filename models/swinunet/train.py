#!/usr/bin/env python3
"""
SwinUnet Training Script
Historical Document Segmentation using Swin Transformer U-Net
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

# Import configuration and training modules
from configs.config import get_config
from trainer import trainer_synapse

# Import dataset classes
from datasets.dataset_udiadsbib_2 import UDiadsBibDataset
try:
    from datasets.dataset_divahisdb import DivaHisDBDataset
    DIVAHISDB_AVAILABLE = True
except ImportError:
    DivaHisDBDataset = None
    DIVAHISDB_AVAILABLE = False


def get_model(args, config):
    """
    Create and initialize the SwinUnet model.
    
    Args:
        args: Command line arguments containing model type and parameters
        config: Configuration object with model settings
        
    Returns:
        torch.nn.Module: Initialized SwinUnet model ready for training
    """
    print("Loading SwinUnet model...")
    from vision_transformer import SwinUnet as ViT_seg
    model = ViT_seg(
        config, 
        img_size=args.img_size, 
        num_classes=args.num_classes
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to CUDA")
    else:
        print("CUDA not available, using CPU")
    
    # Load pretrained weights if available
    try:
        model.load_from(config)
        print("Pretrained weights loaded successfully")
    except FileNotFoundError as e:
        print(f"Warning: Pretrained checkpoint not found: {e}")
        print("Continuing training without pretrained weights (random initialization)")
    except Exception as e:
        print(f"Warning: Failed to load pretrained weights: {e}")
        print("Continuing training without pretrained weights (random initialization)")
    
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
        
        # Check if class-aware augmentation is enabled
        use_class_aware_aug = getattr(args, 'use_class_aware_aug', False)
        if use_class_aware_aug:
            print("✓ Class-aware augmentation enabled (stronger augmentation for rare classes)")
            if num_classes == 5:
                print("  Rare classes: Paratext, Decoration, Title")
            else:  # num_classes == 6
                print("  Rare classes: Paratext, Decoration, Title, Chapter Headings")
        
        # Create datasets
        train_dataset = UDiadsBibDataset(
            root_dir=args.udiadsbib_root,
            split='training',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            manuscript=args.manuscript,
            model_type='swinunet',
            num_classes=num_classes,
            use_class_aware_aug=use_class_aware_aug
        )
        
        val_dataset = UDiadsBibDataset(
            root_dir=args.udiadsbib_root,
            split='validation',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            manuscript=args.manuscript,
            model_type='swinunet',
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
            model_type='swinunet'
        )
        
        val_dataset = DivaHisDBDataset(
            root_dir=args.divahisdb_root,
            split='validation',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            manuscript=args.manuscript,
            model_type='swinunet'
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
    # Check if config file exists
    if not os.path.exists(args.cfg):
        print(f"ERROR: Config file not found: {args.cfg}")
        sys.exit(1)
    
    # Check if dataset root exists
    if args.dataset == 'UDIADS_BIB' and not os.path.exists(args.udiadsbib_root):
        print(f"ERROR: UDIADS_BIB root directory not found: {args.udiadsbib_root}")
        sys.exit(1)
    
    if args.dataset == 'DIVAHISDB' and not os.path.exists(args.divahisdb_root):
        print(f"ERROR: DIVAHISDB root directory not found: {args.divahisdb_root}")
        sys.exit(1)
    
    # Check if output directory is writable
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("All arguments validated successfully!")


def parse_arguments():
    """
    Parse command line arguments for SwinUnet training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='SwinUnet Training for Historical Document Segmentation')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='swinunet', help='Model type (swinunet)')
    parser.add_argument('--cfg', type=str, required=False, help='Path to config file (optional when using --yaml)')
    parser.add_argument('--yaml', type=str, default='swintiny',
                        choices=['swintiny', 'simmim'],
                        help="Choose which preset YAML to use from common/configs: 'swintiny' or 'simmim'. If provided, --cfg is optional and will be overridden.")
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    
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
    parser.add_argument('--base_lr', type=float, default=0.0003, help='Base learning rate')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--scheduler_type', type=str, default='CosineAnnealingWarmRestarts',
                       choices=['ReduceLROnPlateau', 'OneCycleLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts'],
                       help='Learning rate scheduler type (default: CosineAnnealingWarmRestarts for imbalanced data)')
    parser.add_argument('--scheduler_t0', type=int, default=50,
                       help='Initial cycle length for CosineAnnealingWarmRestarts (default: 50 epochs for faster adaptation)')
    parser.add_argument('--scheduler_t_mult', type=int, default=2,
                       help='Cycle length multiplier for CosineAnnealingWarmRestarts (default: 2, doubles each restart)')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Number of warmup epochs for learning rate scheduler')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision (AMP) for faster training (2-3x speedup)')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                       help='Disable AMP (use FP32 training)')
    parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    
    # Checkpoint resume configuration
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (default: auto-detect best_model_latest.pth in output_dir)')
    parser.add_argument('--no_auto_resume', action='store_true', default=False,
                       help='Disable automatic resume from best_model_latest.pth (start fresh training)')
    
    # Data sampling configuration
    parser.add_argument('--use_balanced_sampler', action='store_true', default=False,
                       help='Use balanced sampler to oversample rare classes (helps with class imbalance)')
    
    # Data augmentation configuration
    parser.add_argument('--use_class_aware_aug', action='store_true', default=False,
                       help='Use class-aware augmentation (stronger augmentation for rare classes like Title, Paratext, Decoration)')
    
    # Additional options for config override
    parser.add_argument('--opts', nargs='*', default=None, help='Additional options to override config')
    parser.add_argument('--zip', action='store_true', help='Use zipped dataset instead of folder dataset')
    parser.add_argument('--cache_mode', type=str, default='part', choices=['no', 'full', 'part'], help='Cache mode')
    parser.add_argument('--accumulation_steps', type=int, help='Gradient accumulation steps')
    parser.add_argument('--use_checkpoint', action='store_true', help='Whether to use gradient checkpointing to save memory')
    parser.add_argument('--amp_opt_level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='Mixed precision opt level')
    parser.add_argument('--tag', type=str, help='Tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    
    return parser.parse_args()


def main():
    """
    Main training function for SwinUnet model.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Map the --yaml shortcut to an actual config file path in the repo.
    # This must be set BEFORE validate_arguments(args) which checks args.cfg exists.
    base_config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../common/configs'))
    
    if args.yaml == 'swintiny':
        # keep existing swintiny yaml name (adjust name if your repo uses different file)
        default_cfg = os.path.join(base_config_dir, 'swin_tiny_patch4_window7_224_lite.yaml')
    elif args.yaml == 'simmim':
        default_cfg = os.path.join(base_config_dir, 'simmim_swin_base_patch4_window7_224.yaml')
    else:
        default_cfg = None
    
    # If user passed explicit --cfg, prefer that; otherwise use shorthand mapping
    if not args.cfg:
        if default_cfg and os.path.exists(default_cfg):
            args.cfg = default_cfg
        else:
            # If the chosen config file is missing, raise a helpful error
            raise FileNotFoundError(f"Config for --yaml {args.yaml} not found at {default_cfg}. Make sure the file exists.")
    
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
    
    # Load configuration
    print("Loading configuration for SwinUnet...")
    config = get_config(args)
    
    # Set up datasets
    train_dataset, val_dataset = setup_datasets(args)
    
    # Create model
    model = get_model(args, config)
    print("Model created successfully with {} classes".format(args.num_classes))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    print("\n=== Starting Training ===")
    print("Dataset: {}".format(args.dataset))
    print("Model: {}".format(args.model))
    print("Batch size: {}".format(args.batch_size))
    print("Max epochs: {}".format(args.max_epochs))
    print("Learning rate: {}".format(args.base_lr))
    print()
    
    # Run training
    result = trainer_synapse(args, model, args.output_dir, train_dataset, val_dataset)
    print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
    print(result)


if __name__ == '__main__':
    main()
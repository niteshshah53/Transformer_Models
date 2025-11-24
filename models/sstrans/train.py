#!/usr/bin/env python3
"""
SSTrans Training Script
Historical Document Segmentation using Smart Swin Transformer
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
from trainer import trainer_sstrans

# Import dataset classes
from datasets.dataset_udiadsbib import UDiadsBibDataset
try:
    from datasets.dataset_divahisdb import DivaHisDBDataset
    DIVAHISDB_AVAILABLE = True
except ImportError:
    DivaHisDBDataset = None
    DIVAHISDB_AVAILABLE = False


def get_model(args, config):
    """
    Create and initialize the SSTrans model.
    
    Args:
        args: Command line arguments containing model type and parameters
        config: Configuration object with model settings
        
    Returns:
        torch.nn.Module: Initialized SSTrans model ready for training
    """
    print("Loading SSTrans model...")
    from vision_transformer import SwinUnet as SSTrans_seg
    model = SSTrans_seg(
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
    
    model.load_from(config)
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
        train_dataset = UDiadsBibDataset(
            root_dir=args.udiadsbib_root,
            split='training',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            manuscript=args.manuscript,
            model_type='sstrans',
            num_classes=num_classes
        )
        
        val_dataset = UDiadsBibDataset(
            root_dir=args.udiadsbib_root,
            split='validation',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            manuscript=args.manuscript,
            model_type='sstrans',
            num_classes=num_classes
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
            model_type='sstrans'
        )
        
        val_dataset = DivaHisDBDataset(
            root_dir=args.divahisdb_root,
            split='validation',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            manuscript=args.manuscript,
            model_type='sstrans'
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
    Parse command line arguments for SSTrans training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='SSTrans Training for Historical Document Segmentation')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='sstrans', help='Model type (sstrans)')
    parser.add_argument('--cfg', type=str, required=True, help='Path to config file')
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
    Main training function for SSTrans model.
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
    
    # Load configuration
    print("Loading configuration for SSTrans...")
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
    result = trainer_sstrans(args, model, args.output_dir, train_dataset, val_dataset)
    print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
    print(result)


if __name__ == '__main__':
    main()

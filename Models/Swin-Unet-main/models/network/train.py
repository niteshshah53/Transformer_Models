#!/usr/bin/env python3
"""
CNN-Transformer Training Script
Historical Document Segmentation using EfficientNet-Swin Transformer U-Net
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
from trainer import trainer_synapse

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
    Create and initialize the CNN-Transformer model.
    
    Args:
        args: Command line arguments containing model type and parameters
        config: Configuration object with model settings (not used for CNN-Transformer)
        
    Returns:
        torch.nn.Module: Initialized CNN-Transformer model ready for training
    """
    print("Loading CNN-Transformer model...")
    from vision_transformer_cnn import CNNTransformerUnet as ViT_seg
    
    use_baseline = getattr(args, 'use_baseline', False)
    
    if use_baseline:
        print("=" * 80)
        print("🚀 Loading CNN-Transformer BASELINE")
        print("=" * 80)
        print("Baseline Configuration:")
        print("  ✓ EfficientNet-B4 Encoder")
        print("  ✓ Bottleneck: 2 Swin Transformer blocks")
        print("  ✓ Swin Transformer Decoder")
        print("  ✓ Simple concatenation skip connections")
        print("  ✓ Adapter mode: streaming (default)")
        print("  ✓ GroupNorm: {}".format(getattr(args, 'use_groupnorm', True)))
        print("=" * 80)
        
        # Baseline defaults
        adapter_mode = 'streaming'  # Default for baseline
        use_bottleneck = True  # Always enabled for baseline
        fusion_method = getattr(args, 'fusion_method', 'simple')
        
        model = ViT_seg(
            None, 
            img_size=args.img_size, 
            num_classes=args.num_classes,
            use_deep_supervision=getattr(args, 'deep_supervision', False),
            fusion_method=fusion_method,
            use_bottleneck=use_bottleneck,
            adapter_mode=adapter_mode,
            use_multiscale_agg=getattr(args, 'use_multiscale_agg', False),
            use_groupnorm=getattr(args, 'use_groupnorm', True)  # Default True for baseline
        )
    else:
        # Original non-baseline mode (backward compatibility)
        model = ViT_seg(
            None, 
            img_size=args.img_size, 
            num_classes=args.num_classes,
            use_deep_supervision=getattr(args, 'deep_supervision', False),
            fusion_method=getattr(args, 'fusion_method', 'simple'),
            use_bottleneck=getattr(args, 'bottleneck', False),
            adapter_mode=getattr(args, 'adapter_mode', 'external'),
            use_multiscale_agg=getattr(args, 'use_multiscale_agg', False)
        )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to CUDA")
    else:
        print("CUDA not available, using CPU")
    
    model.load_from(None)
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
            model_type='swinunet',
            num_classes=num_classes
        )
        
        val_dataset = UDiadsBibDataset(
            root_dir=args.udiadsbib_root,
            split='validation',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            manuscript=args.manuscript,
            model_type='swinunet',
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
    # Check if dataset root exists
    if args.dataset == 'UDIADS_BIB' and not os.path.exists(args.udiadsbib_root):
        print(f"ERROR: UDIADS_BIB root directory not found: {args.udiadsbib_root}")
        sys.exit(1)
    
    if args.dataset == 'DIVAHISDB' and not os.path.exists(args.divahisdb_root):
        print(f"ERROR: DIVAHISDB root directory not found: {args.divahisdb_root}")
        sys.exit(1)
    
    # Check if output directory is writable
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate baseline flag usage
    use_baseline = getattr(args, 'use_baseline', False)
    
    # Component flags that should only work with --use_baseline
    component_flags = [
        ('use_deep_supervision', getattr(args, 'deep_supervision', False)),
        ('use_fourier_fusion', getattr(args, 'fusion_method', 'simple') == 'fourier'),
        ('use_smart_fusion', getattr(args, 'fusion_method', 'simple') == 'smart'),
        ('use_multiscale_agg', getattr(args, 'use_multiscale_agg', False)),
        ('use_groupnorm', getattr(args, 'use_groupnorm', False)),
    ]
    
    # Check if component flags are used without --use_baseline
    if not use_baseline:
        used_flags = [name for name, used in component_flags if used]
        if used_flags:
            print(f"ERROR: Component flags {used_flags} can only be used with --use_baseline flag")
            print("Please add --use_baseline to enable these features")
            sys.exit(1)
    
    print("All arguments validated successfully!")


def parse_arguments():
    """
    Parse command line arguments for CNN-Transformer training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='CNN-Transformer Training for Historical Document Segmentation')
    
    # Model configuration
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    
    # Baseline flag (similar to hybrid2)
    parser.add_argument('--use_baseline', action='store_true', default=False,
                       help='Use baseline CNN-Transformer (EfficientNet-B4 encoder + bottleneck + Swin decoder)')
    
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
    parser.add_argument('--scheduler_type', type=str, default='CosineAnnealingWarmRestarts',
                       choices=['CosineAnnealingWarmRestarts', 'OneCycleLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'],
                       help='Learning rate scheduler type')
    parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    
    # Baseline enhancement flags (only used with --use_baseline)
    parser.add_argument('--deep_supervision', action='store_true', default=False, 
                       help='Enable deep supervision with 3 auxiliary outputs (requires --use_baseline)')
    parser.add_argument('--fusion_method', type=str, default='simple',
                       choices=['simple', 'fourier', 'smart'],
                       help='Feature fusion: simple (concat), fourier (FFT-based), smart (attention-based smart skip connections) (requires --use_baseline)')
    parser.add_argument('--use_multiscale_agg', action='store_true', default=False,
                       help='Enable multi-scale aggregation in bottleneck (requires --use_baseline)')
    parser.add_argument('--use_groupnorm', action='store_true', default=True,
                       help='Use GroupNorm instead of LayerNorm (default: True for baseline, requires --use_baseline)')
    parser.add_argument('--no_groupnorm', dest='use_groupnorm', action='store_false',
                       help='Disable GroupNorm (use LayerNorm instead)')
    
    # Freezing configuration
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                       help='Freeze encoder during training (train decoder only)')
    parser.add_argument('--freeze_epochs', type=int, default=0,
                       help='Number of epochs to freeze encoder (0 = freeze for entire training)')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    
    return parser.parse_args()


def main():
    """
    Main training function for CNN-Transformer model.
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
    
    # Create model (no config needed for CNN-Transformer)
    model = get_model(args, None)
    print("Model created successfully with {} classes".format(args.num_classes))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    print("\n=== Starting Training ===")
    print("Dataset: {}".format(args.dataset))
    print("Model: CNN-Transformer")
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
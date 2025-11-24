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
    
    All architecture flags are independent and can be combined freely.
    
    Args:
        args: Command line arguments containing model type and parameters
        config: Configuration object with model settings (not used for CNN-Transformer)
        
    Returns:
        torch.nn.Module: Initialized CNN-Transformer model ready for training
    """
    print("Loading CNN-Transformer model...")
    from vision_transformer_cnn import CNNTransformerUnet as ViT_seg
    
    # Get all model configuration from args (all flags are independent)
    use_bottleneck = getattr(args, 'bottleneck', True)
    adapter_mode = getattr(args, 'adapter_mode', 'streaming')
    fusion_method = getattr(args, 'fusion_method', 'simple')
    use_deep_supervision = getattr(args, 'deep_supervision', False)
    use_multiscale_agg = getattr(args, 'use_multiscale_agg', False)
    use_groupnorm = getattr(args, 'use_groupnorm', True)
    use_se_msfe = getattr(args, 'use_se_msfe', False)
    use_msfa_mct_bottleneck = getattr(args, 'use_msfa_mct_bottleneck', False)
    encoder_type = getattr(args, 'encoder_type', 'efficientnet')  # 'efficientnet' or 'resnet50'
    
    # Print configuration
    print("=" * 80)
    print("🚀 Loading CNN-Transformer Model")
    print("=" * 80)
    print("Model Configuration:")
    if encoder_type == 'resnet50':
        print("  ✓ ResNet-50 Encoder (official)")
    else:
        print("  ✓ EfficientNet-B4 Encoder")
        if use_se_msfe:
            print("    - SE-MSFE: Enabled (replaces MBConv conv operations)")
    print(f"  ✓ Bottleneck: {'Enabled' if use_bottleneck else 'Disabled'}")
    if use_bottleneck:
        if use_msfa_mct_bottleneck:
            print("    - Type: MSFA + MCT Hybrid (from MSAGHNet)")
        else:
            print("    - Type: 2 Swin Transformer blocks")
    print("  ✓ Swin Transformer Decoder")
    print(f"  ✓ Fusion Method: {fusion_method}")
    print(f"  ✓ Adapter Mode: {adapter_mode}")
    print(f"  ✓ Deep Supervision: {'Enabled' if use_deep_supervision else 'Disabled'}")
    print(f"  ✓ Multi-Scale Aggregation: {'Enabled' if use_multiscale_agg else 'Disabled'}")
    print(f"  ✓ Normalization: {'GroupNorm' if use_groupnorm else 'LayerNorm'}")
    print("=" * 80)
    
    # Create model with all flags (all independent and compatible)
    model = ViT_seg(
        None, 
        img_size=args.img_size, 
        num_classes=args.num_classes,
        use_deep_supervision=use_deep_supervision,
        fusion_method=fusion_method,
        use_bottleneck=use_bottleneck,
        adapter_mode=adapter_mode,
        use_multiscale_agg=use_multiscale_agg,
        use_groupnorm=use_groupnorm,
        encoder_type=encoder_type,
        use_se_msfe=use_se_msfe,
        use_msfa_mct_bottleneck=use_msfa_mct_bottleneck
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
        
        # Handle patched data path: add _patched suffix if use_patched_data=True and path doesn't already have it
        root_dir = args.udiadsbib_root
        if args.use_patched_data and not root_dir.endswith('_patched'):
            root_dir = root_dir + '_patched'
            print(f"Using patched data: adjusted root directory to {root_dir}")
        elif not args.use_patched_data and root_dir.endswith('_patched'):
            print(f"Warning: root directory ends with '_patched' but use_patched_data=False")
        
        # Create datasets
        use_class_aware_aug = getattr(args, 'use_class_aware_aug', False)
        if use_class_aware_aug:
            print("✓ Class-aware augmentation enabled (stronger augmentation for rare classes)")
            if num_classes == 5:
                print("  Rare classes: Paratext, Decoration, Title")
            else:  # num_classes == 6
                print("  Rare classes: Paratext, Decoration, Title, Chapter Headings")
        
        train_dataset = UDiadsBibDataset(
            root_dir=root_dir,
            split='training',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            manuscript=args.manuscript,
            model_type='network',  # Network model (previously called 'hybrid1')
            num_classes=num_classes,
            use_class_aware_aug=use_class_aware_aug
        )
        
        val_dataset = UDiadsBibDataset(
            root_dir=root_dir,
            split='validation',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            manuscript=args.manuscript,
            model_type='network',  # Network model (previously called 'hybrid1')
            num_classes=num_classes,
            use_class_aware_aug=False  # Never use augmentation for validation
        )
        
    elif args.dataset == 'DIVAHISDB' and DIVAHISDB_AVAILABLE:
        print("Setting up DivaHisDB dataset...")
        args.num_classes = 4
        
        # Handle patched data path: add _patched suffix if use_patched_data=True and path doesn't already have it
        root_dir = args.divahisdb_root
        if args.use_patched_data and not root_dir.endswith('_patched'):
            root_dir = root_dir + '_patched'
            print(f"Using patched data: adjusted root directory to {root_dir}")
        elif not args.use_patched_data and root_dir.endswith('_patched'):
            print(f"Warning: root directory ends with '_patched' but use_patched_data=False")
        
        train_dataset = DivaHisDBDataset(
            root_dir=root_dir,
            split='training',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            manuscript=args.manuscript,
            model_type='network'  # Network model (previously called 'hybrid1')
        )
        
        val_dataset = DivaHisDBDataset(
            root_dir=root_dir,
            split='validation',
            patch_size=args.img_size,
            use_patched_data=args.use_patched_data,
            manuscript=args.manuscript,
            model_type='network'  # Network model (previously called 'hybrid1')
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
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='UDIADS_BIB', 
                       choices=['UDIADS_BIB', 'DIVAHISDB'], help='Dataset to use')
    parser.add_argument('--udiadsbib_root', type=str, default='U-DIADS-Bib-FS',
                       help='Root directory for UDIADS_BIB dataset (if using patched data, use path ending with _patched, e.g., U-DIADS-Bib-FS_patched)')
    parser.add_argument('--divahisdb_root', type=str, default='DIVAHISDB',
                       help='Root directory for DIVAHISDB dataset (if using patched data, use path ending with _patched)')
    parser.add_argument('--manuscript', type=str, default='Syr341FS',
                       help='Manuscript to train on')
    parser.add_argument('--use_patched_data', action='store_true', default=True,
                       help='Use pre-generated patches (expects root directory path ending with _patched)')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size per GPU (default: 8, increase if memory allows. Small batches (4) cause unstable gradients and poor GPU utilization)')
    parser.add_argument('--max_epochs', type=int, default=300, help='Maximum number of epochs')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='Base learning rate')
    parser.add_argument('--encoder_lr_factor', type=float, default=0.05,
                       help='Learning rate multiplier for pretrained encoder (default: 0.05x base_lr, recommended: 0.05-0.2 for pretrained layers)')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--scheduler_type', type=str, default='CosineAnnealingWarmRestarts',
                       choices=['CosineAnnealingWarmRestarts', 'OneCycleLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'],
                       help='Learning rate scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Number of warmup epochs for learning rate scheduler (critical for transformer training stability)')
    parser.add_argument('--val_interval', type=int, default=1,
                       help='Validation interval in epochs (default: 1 = every epoch, set to 5 to validate every 5 epochs for faster training)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision (AMP) for faster training (2-3x speedup on modern GPUs)')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                       help='Disable AMP (use FP32 training)')
    parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    
    # Model architecture flags (all flags are independent and can be combined freely)
    parser.add_argument('--bottleneck', action='store_true', default=True,
                       help='Enable bottleneck with 2 Swin Transformer blocks (default: True)')
    parser.add_argument('--no_bottleneck', dest='bottleneck', action='store_false',
                       help='Disable bottleneck')
    parser.add_argument('--adapter_mode', type=str, default='streaming',
                       choices=['external', 'streaming'],
                       help='Adapter placement mode: external (separate adapters) or streaming (integrated) (default: streaming)')
    parser.add_argument('--deep_supervision', action='store_true', default=False, 
                       help='Enable deep supervision with 3 auxiliary outputs')
    parser.add_argument('--fusion_method', type=str, default='simple',
                       choices=['simple', 'fourier', 'smart', 'gcff'],
                       help='Feature fusion method: simple (concat), fourier (FFT-based), smart (attention-based smart skip connections), gcff (Global Context Feature Fusion from MSAGHNet)')
    parser.add_argument('--use_multiscale_agg', action='store_true', default=False,
                       help='Enable multi-scale aggregation in bottleneck')
    parser.add_argument('--use_groupnorm', action='store_true', default=True,
                       help='Use GroupNorm instead of LayerNorm (default: True)')
    parser.add_argument('--no_groupnorm', dest='use_groupnorm', action='store_false',
                       help='Disable GroupNorm (use LayerNorm instead)')
    parser.add_argument('--use_se_msfe', action='store_true', default=False,
                       help='Use SE-MSFE (Squeeze-and-Excitation Multi-Scale Feature Extraction) to replace MBConv conv operations in encoder (default: False)')
    parser.add_argument('--use_msfa_mct_bottleneck', action='store_true', default=False,
                       help='Use MSFA + MCT Hybrid Bottleneck (from MSAGHNet) instead of 2 Swin Transformer blocks (default: False)')
    
    # Encoder type configuration
    parser.add_argument('--encoder_type', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet50'],
                       help='Encoder type: efficientnet (EfficientNet-B4) or resnet50 (ResNet-50) (default: efficientnet)')
    
    # Freezing configuration
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                       help='Freeze encoder during training (train decoder only)')
    parser.add_argument('--freeze_epochs', type=int, default=0,
                       help='Number of epochs to freeze encoder (0 = freeze for entire training)')
    
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
    
    # Loss function configuration
    parser.add_argument('--use_cb_loss', action='store_true', default=False,
                       help='Use Class-Balanced Loss instead of standard CE (best for extreme imbalance >100:1)')
    parser.add_argument('--cb_beta', type=float, default=0.9999,
                       help='Beta hyperparameter for Class-Balanced Loss (default: 0.9999 for extreme imbalance)')
    parser.add_argument('--focal_gamma', type=float, default=3.0,
                       help='Focal loss gamma parameter (default: 3.0 for extreme imbalance with CB Loss, 4.0 for standalone use, 2.0 for moderate imbalance)')
    
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
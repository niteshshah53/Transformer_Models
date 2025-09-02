import argparse
import os
import random
import sys
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import configuration and training modules
from config import get_config
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
    Create and initialize the specified model.
    
    Args:
        args: Command line arguments containing model type and parameters
        config: Configuration object with model settings (None for MissFormer)
        
    Returns:
        torch.nn.Module: Initialized model ready for training
        
    Raises:
        SystemExit: If unsupported model type is specified
    """
    model_name = args.model.lower()
    
    if model_name == 'swinunet':
        if config is None:
            raise ValueError("Config is required for SwinUnet model")
        print("Loading SwinUnet model...")
        from networks.vision_transformer import SwinUnet as ViT_seg
        model = ViT_seg(
            config, 
            img_size=args.img_size, 
            num_classes=args.num_classes
        ).cuda()
        model.load_from(config)
        return model
        
    elif model_name == 'missformer':
        print("Loading MissFormer model...")
        from networks.MissFormer.MISSFormer import MISSFormer
        model = MISSFormer(num_classes=args.num_classes)
        model = model.cuda()
        return model
        
    else:
        print(f"ERROR: Unknown model '{args.model}'")
        print("Supported models: swinunet, missformer")
        sys.exit(1)


def setup_datasets(args):
    """
    Create training and validation datasets based on the specified dataset type.
    
    Args:
        args: Command line arguments containing dataset configuration
        
    Returns:
        tuple: (train_dataset, val_dataset) - PyTorch datasets for training and validation
        
    Raises:
        SystemExit: If unsupported dataset is specified or dataset is unavailable
    """
    dataset_name = args.dataset.lower()
    
    if dataset_name == "udiads_bib":
        print("Setting up U-DIADS-Bib dataset...")
        args.num_classes = 6  # U-DIADS-Bib has 6 classes
        
        train_dataset = UDiadsBibDataset(
            root_dir=args.udiadsbib_root,
            split="training",
            patch_size=args.patch_size,
            stride=args.patch_stride,
            use_patched_data=args.use_patched_data
        )
        
        val_dataset = UDiadsBibDataset(
            root_dir=args.udiadsbib_root,
            split="validation",
            patch_size=None,  # Use full images for validation
            stride=None,
            use_patched_data=args.use_patched_data
        )
        
        return train_dataset, val_dataset
        
    elif dataset_name == "divahisdb":
        if not DIVAHISDB_AVAILABLE:
            print("ERROR: DIVAHISDB dataset support not available")
            print("Make sure the dataset_divahisdb.py file exists")
            sys.exit(1)
            
        print("Setting up DIVAHISDB dataset...")
        args.num_classes = 4  # DIVAHISDB has 4 classes
        
        train_dataset = DivaHisDBDataset(
            root_dir=args.divahisdb_root,  # Use correct root path
            split="training",
            patch_size=args.patch_size,
            stride=args.patch_stride,
            use_patched_data=args.use_patched_data
        )
        
        val_dataset = DivaHisDBDataset(
            root_dir=args.divahisdb_root,
            split="validation",
            patch_size=None,  # Use full images for validation
            stride=None,
            use_patched_data=args.use_patched_data
        )
        
        return train_dataset, val_dataset
        
    else:
        print(f"ERROR: Unsupported dataset '{args.dataset}'")
        print("Supported datasets: UDIADS_BIB, DIVAHISDB")
        sys.exit(1)


def setup_reproducible_training(seed):
    """
    Set up reproducible training by fixing random seeds.
    
    Args:
        seed (int): Random seed to use for all random number generators
    """
    print(f"Setting random seed to {seed} for reproducible training...")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Set CUDNN settings for reproducibility
    cudnn.benchmark = False
    cudnn.deterministic = True


def setup_fast_training():
    """
    Set up training for maximum speed (non-reproducible).
    """
    print("Setting up fast training mode (non-reproducible)...")
    
    # Enable CUDNN optimizations for speed
    cudnn.benchmark = True
    cudnn.deterministic = False


def validate_arguments(args):
    """
    Validate command line arguments and fix common issues.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        ValueError: If required arguments are missing or invalid
    """
    # Check for accidental paste fragments in command line
    bad_tokens = [token for token in sys.argv[1:] 
                  if token.lstrip('-').startswith('mg_') or token.lstrip('-').startswith('mg')]
    if bad_tokens:
        print(f"WARNING: Suspicious command line tokens detected: {bad_tokens}")
        print("Did you accidentally paste a partial command? Please check your command.")
    
    # Ensure num_classes is properly set
    if not hasattr(args, 'num_classes') or args.num_classes is None:
        if hasattr(args, 'n_class') and args.n_class is not None:
            args.num_classes = args.n_class
        else:
            raise ValueError("Please provide --num_classes (or --n_class)")
    
    # Also set n_class for backwards compatibility
    args.n_class = args.num_classes
    
    # Validate required paths
    if args.dataset.lower() == "udiads_bib" and not os.path.exists(args.udiadsbib_root):
        raise ValueError(f"U-DIADS-Bib dataset path does not exist: {args.udiadsbib_root}")
    
    if args.dataset.lower() == "divahisdb" and not os.path.exists(args.divahisdb_root):
        raise ValueError(f"DIVAHISDB dataset path does not exist: {args.divahisdb_root}")
    
    # Validate config file
    if args.model.lower() == 'swinunet':
        if not args.cfg:
            raise ValueError("--cfg argument is required for SwinUnet model")
        if not os.path.exists(args.cfg):
            raise ValueError(f"Config file does not exist: {args.cfg}")
    elif args.model.lower() == 'missformer' and args.cfg:
        # For MissFormer, config is optional but if provided, validate it exists
        if not os.path.exists(args.cfg):
            raise ValueError(f"Config file does not exist: {args.cfg}")
    
    print("All arguments validated successfully!")


def parse_arguments():
    """
    Parse and return command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Train segmentation models on historical document datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train SwinUnet on U-DIADS-Bib dataset
  python train.py --cfg configs/swin_tiny.yaml --dataset UDIADS_BIB --output_dir ./models/
  
  # Train MissFormer on DIVAHISDB dataset with custom parameters (no config needed)
  python train.py --model missformer --dataset DIVAHISDB \\
                  --divahisdb_root ./DIVAHISDB --batch_size 16 --output_dir ./models/
  
  # Train with custom early stopping patience (stop after 30 epochs without improvement)
  python train.py --cfg configs/swin_tiny.yaml --dataset UDIADS_BIB --output_dir ./models/ \\
                  --patience 30
  
  # Train with custom learning rate scheduler parameters
  python train.py --cfg configs/swin_tiny.yaml --dataset UDIADS_BIB --output_dir ./models/ \\
                  --lr_factor 0.3 --lr_patience 15 --lr_min 1e-6
        """
    )
    
    # Model selection
    parser.add_argument('--model', type=str, default='swinunet', 
                       choices=['swinunet', 'missformer'],
                       help='Model architecture to use')
    
    # Required configuration
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE",
                       help='Path to model configuration file (YAML) (required for SwinUnet, optional for MissFormer)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save trained models and logs')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='UDIADS_BIB',
                       choices=['UDIADS_BIB', 'DIVAHISDB'],
                       help='Dataset to use for training')
    parser.add_argument('--udiadsbib_root', type=str, default='U-DIADS-Bib-MS',
                       help='Root directory for U-DIADS-Bib dataset')
    parser.add_argument('--divahisdb_root', type=str, default='DIVAHISDB',
                       help='Root directory for DIVAHISDB dataset')
    parser.add_argument('--use_patched_data', action='store_true',
                       help='Use pre-generated patches instead of extracting patches on-the-fly')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of segmentation classes (auto-detected from dataset if not specified)')
    parser.add_argument('--n_class', type=int, default=None,
                       help='Alternative name for num_classes (for backwards compatibility)')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size (images will be resized to this)')
    
    # Training parameters
    parser.add_argument('--max_epochs', type=int, default=150,
                       help='Maximum number of training epochs')
    parser.add_argument('--max_iterations', type=int, default=30000,
                       help='Maximum number of training iterations')
    parser.add_argument('--batch_size', type=int, default=24,
                       help='Training batch size')
    parser.add_argument('--base_lr', type=float, default=0.001,
                       help='Base learning rate')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience (stop if no improvement for N epochs)')
    
    # Learning rate scheduler parameters
    parser.add_argument('--lr_factor', type=float, default=0.5,
                       help='Factor to reduce learning rate by when plateauing (default: 0.5)')
    parser.add_argument('--lr_patience', type=int, default=10,
                       help='Patience for learning rate reduction (default: 10 epochs)')
    parser.add_argument('--lr_min', type=float, default=1e-7,
                       help='Minimum learning rate (default: 1e-7)')
    parser.add_argument('--lr_threshold', type=float, default=1e-4,
                       help='Threshold for considering improvement (default: 1e-4)')
    
    # Patch-based training parameters
    parser.add_argument('--patch_size', type=int, default=224,
                       help='Size of patches for patch-based training')
    parser.add_argument('--patch_stride', type=int, default=224,
                       help='Stride for patch extraction (224 = no overlap)')
    
    # Training behavior
    parser.add_argument('--deterministic', type=int, default=1,
                       help='Use deterministic training (1) or fast training (0)')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed for reproducible training')
    parser.add_argument('--n_gpu', type=int, default=1,
                       help='Number of GPUs to use')
    
    # Evaluation
    parser.add_argument('--eval_interval', type=int, default=1,
                       help='Evaluate model every N epochs')
    parser.add_argument('--eval', action='store_true',
                       help='Only run evaluation, do not train')
    
    # Advanced options (mostly for model configuration)
    parser.add_argument('--opts', nargs='+', default=None,
                       help='Modify config options by adding KEY VALUE pairs')
    parser.add_argument('--zip', action='store_true',
                       help='Use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part',
                       choices=['no', 'full', 'part'],
                       help='Dataset caching strategy')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--accumulation-steps', type=int, default=None,
                       help='Gradient accumulation steps')
    parser.add_argument('--use-checkpoint', action='store_true',
                       help='Use gradient checkpointing to save memory')
    parser.add_argument('--amp-opt-level', type=str, default='O1',
                       choices=['O0', 'O1', 'O2'],
                       help='Mixed precision optimization level')
    parser.add_argument('--tag', type=str, default=None,
                       help='Tag for this experiment (for logging)')
    parser.add_argument('--throughput', action='store_true',
                       help='Test model throughput only')
    
    # Legacy arguments (kept for backwards compatibility)
    parser.add_argument('--root_path', type=str, 
                       default='../data/Synapse/train_npz',
                       help='Legacy root directory (for non-UDIADS_BIB datasets)')
    parser.add_argument('--list_dir', type=str,
                       default='./lists/lists_Synapse',
                       help='Legacy list directory (for non-UDIADS_BIB datasets)')
    parser.add_argument('--udiadsbib_split', type=str, default='training',
                       help='Legacy split argument (not used)')
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    print("=== Historical Document Segmentation Training ===")
    print()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate arguments
    try:
        validate_arguments(args)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Set up reproducible or fast training
    if args.deterministic:
        setup_reproducible_training(args.seed)
    else:
        setup_fast_training()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    if args.model.lower() == 'swinunet':
        print("Loading configuration for SwinUnet...")
        config = get_config(args)
    elif args.model.lower() == 'missformer':
        print("MissFormer model - no configuration file needed")
        config = None
    else:
        print(f"Unknown model: {args.model}. Supported: swinunet, missformer")
        sys.exit(1)
    
    # Set up datasets
    try:
        train_dataset, val_dataset = setup_datasets(args)
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    except Exception as e:
        print(f"ERROR: Failed to load datasets: {e}")
        sys.exit(1)
    
    # Create model
    try:
        model = get_model(args, config)
        print(f"Model created successfully with {args.num_classes} classes")
    except Exception as e:
        print(f"ERROR: Failed to create model: {e}")
        sys.exit(1)
    
    # Start training
    print()
    print("=== Starting Training ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.base_lr}")
    print()
    
    try:
        trainer_synapse(
            args, 
            model, 
            args.output_dir, 
            train_dataset=train_dataset, 
            val_dataset=val_dataset
        )
        
        print()
        print("=== TRAINING COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
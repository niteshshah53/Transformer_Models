"""
Testing Script for Historical Document Segmentation Models

This script evaluates trained models on historical document test datasets by:
- Loading trained model checkpoints
- Running inference on test images using patch-based approach
- Computing segmentation metrics (IoU, Precision, Recall, F1)
- Saving prediction visualizations

Supported datasets: U-DIADS-Bib, DIVAHISDB

Usage:
    # For Hybrid EfficientNet-Swin (no config needed):
    python test.py --model hybrid2 --output_dir ./models/ --manuscript Latin2 --is_savenii
    
Author: Clean Code Version
"""

import argparse
import json
import logging
import os
import random
import sys
import warnings
import glob

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Add common directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import config system
from configs.config import get_config


def parse_arguments():
    """Parse command line arguments for testing script."""
    parser = argparse.ArgumentParser(
        description='Test Hybrid EfficientNet-Swin model on historical document datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on U-DIADS-Bib dataset with Hybrid EfficientNet-Swin
  python test.py --model hybrid2 --output_dir ./models/ \\
                 --dataset UDIADS_BIB --manuscript Latin2 --is_savenii
  
  # Test on DIVAHISDB dataset with Hybrid EfficientNet-Swin
  python test.py --model hybrid2 --dataset DIVAHISDB \\
                 --output_dir ./models/ --manuscript Latin2
        """
    )
    
    # Core arguments
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE",
                       help='Path to model configuration file (optional when using --yaml)')
    parser.add_argument('--yaml', type=str, default='swintiny',
                       choices=['swintiny', 'simmim'],
                       help="Choose which preset YAML to use from common/configs. Default: 'swintiny'. Options: 'swintiny' or 'simmim'. If provided, --cfg is optional and will be overridden.")
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing trained model checkpoints')
    
    # Hybrid2 model variants
    parser.add_argument('--use_baseline', action='store_true', default=False,
                       help='Use baseline Hybrid2 (required). If used alone, defaults to simple decoder. Must be used with --decoder for other decoders.')
    parser.add_argument('--decoder', type=str, default='simple',
                       choices=['simple', 'EfficientNet-B4', 'ResNet50'],
                       help='Decoder type: simple (default when --use_baseline is used alone), EfficientNet-B4, or ResNet50 (requires --use_baseline, must match training)')
    parser.add_argument('--efficientnet_variant', type=str, default='b4', choices=['b0', 'b4'],
                       help='EfficientNet variant for simple decoder (b0, b4). Only used when --decoder simple, must match training')
    
    # Hybrid2 baseline enhancement flags (only used with --use_baseline, must match training)
    parser.add_argument('--use_deep_supervision', action='store_true', default=False,
                       help='Enable deep supervision (must match training configuration)')
    parser.add_argument('--use_cbam', action='store_true', default=False,
                       help='Enable CBAM attention (must match training configuration)')
    parser.add_argument('--use_smart_skip', action='store_true', default=False,
                       help='Use smart skip connections (must match training configuration)')
    parser.add_argument('--use_cross_attn', action='store_true', default=False,
                       help='Enable cross-attention (must match training configuration)')
    parser.add_argument('--use_multiscale_agg', action='store_true', default=False,
                       help='Enable multi-scale aggregation (must match training configuration)')
    parser.add_argument('--use_groupnorm', action='store_true', default=True,
                       help='Use GroupNorm (must match training configuration, default: True)')
    parser.add_argument('--use_batchnorm', action='store_true', default=False,
                       help='Use BatchNorm instead of GroupNorm (overrides --use_groupnorm, must match training configuration)')
    parser.add_argument('--use_pos_embed', action='store_true', default=True,
                       help='Enable positional embeddings (default: True, must match training configuration)')
    parser.add_argument('--no_pos_embed', dest='use_pos_embed', action='store_false',
                       help='Disable positional embeddings')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='UDIADS_BIB',
                       choices=['UDIADS_BIB', 'DIVAHISDB'],
                       help='Dataset to test on')
    parser.add_argument('--manuscript', type=str, required=True,
                       help='Manuscript to test (e.g., Latin2, Latin14396, Latin16746, Syr341, Latin2FS, etc.)')
    parser.add_argument('--udiadsbib_root', type=str, default='../../U-DIADS-Bib-MS',
                       help='Root directory for U-DIADS-Bib dataset')
    parser.add_argument('--divahisdb_root', type=str, default='../../DivaHisDB',
                       help='Root directory for DIVAHISDB dataset')
    parser.add_argument('--use_patched_data', action='store_true',
                       help='Use pre-generated patches instead of full images')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of segmentation classes (auto-detected from dataset)')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input patch size for inference')
    parser.add_argument('--batch_size', type=int, default=24,
                       help='Batch size for testing')
    
    # Output options
    parser.add_argument('--is_savenii', action="store_true",
                       help='Save prediction results during inference')
    parser.add_argument('--test_save_dir', type=str, default='../predictions',
                       help='Directory to save prediction results')
    
    # CRF post-processing
    parser.add_argument('--use_crf', action='store_true', default=False,
                       help='Enable CRF post-processing: refine predictions using DenseCRF for spatial coherence')
    
    # System configuration
    parser.add_argument('--deterministic', type=int, default=1,
                       help='Use deterministic testing')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed for reproducibility')
    
    # Advanced options (required by config.py)
    parser.add_argument('--opts', nargs='+', default=None,
                       help='Modify config options')
    parser.add_argument('--zip', action='store_true',
                       help='Use zipped dataset')
    parser.add_argument('--cache-mode', type=str, default='part',
                       choices=['no', 'full', 'part'],
                       help='Dataset caching strategy')
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                       help='Gradient accumulation steps')
    parser.add_argument('--use-checkpoint', action='store_true',
                       help='Use gradient checkpointing')
    parser.add_argument('--amp-opt-level', type=str, default='O1',
                       choices=['O0', 'O1', 'O2'],
                       help='Mixed precision optimization level')
    parser.add_argument('--tag', help='Experiment tag')
    parser.add_argument('--eval', action='store_true',
                       help='Evaluation only mode')
    parser.add_argument('--throughput', action='store_true',
                       help='Test throughput only')
    
    # Test-time augmentation
    parser.add_argument('--use_tta', action='store_true',
                       help='Use test-time augmentation for improved accuracy (+2-4% mIoU)')
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate and normalize command line arguments."""
    # Check for suspicious command line tokens
    bad_tokens = [t for t in sys.argv[1:] if t.lstrip('-').startswith('mg_')]
    if bad_tokens:
        print(f"Warning: suspicious argv tokens detected: {bad_tokens}")
        print("Did you accidentally paste a continuation fragment?")
    
    # Map the --yaml shortcut to an actual config file path FIRST
    # This must be done BEFORE checking if config exists
    base_config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../common/configs'))
    
    if args.yaml == 'swintiny':
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
    
    # Check if config file exists (required for hybrid model)
    if not args.cfg:
        raise ValueError("Config file is required. Use --cfg <path> or --yaml <swintiny|simmim>")
    
    if not os.path.exists(args.cfg):
        raise FileNotFoundError(f"Config file not found: {args.cfg}")
    
    # Validate required paths
    if not os.path.exists(args.output_dir):
        raise ValueError(f"Output directory not found: {args.output_dir}")
    
    # Set dataset-specific parameters
    if args.dataset.upper() == "UDIADS_BIB":
        # Determine number of classes based on manuscript
        if args.manuscript in ['Syr341FS', 'Syr341']:
            args.num_classes = 5
            print("Detected Syriaque341 manuscript: using 5 classes (no Chapter Headings)")
        else:
            args.num_classes = 6
            print(f"Using 6 classes for manuscript: {args.manuscript}")
        
        if not os.path.exists(args.udiadsbib_root):
            raise ValueError(f"U-DIADS-Bib dataset path not found: {args.udiadsbib_root}")
    elif args.dataset.upper() == "DIVAHISDB":
        args.num_classes = 4
        if not os.path.exists(args.divahisdb_root):
            raise ValueError(f"DIVAHISDB dataset path not found: {args.divahisdb_root}")
    
    # Validate decoder and use_baseline flag usage (matching train.py)
    use_baseline = getattr(args, 'use_baseline', False)
    decoder_type = getattr(args, 'decoder', 'simple')
    
    # Rule 1: If decoder is specified (and not 'simple'), must use --use_baseline
    if decoder_type != 'simple' and not use_baseline:
        raise ValueError(f"--decoder {decoder_type} requires --use_baseline flag. Usage: --use_baseline --decoder {decoder_type}")
    
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
        raise ValueError(f"Enhancement flags {used_enhancement_flags} require --use_baseline flag. Usage: --use_baseline {' '.join([f'--{flag}' for flag in used_enhancement_flags])}")


def get_model(args, config):
    """
    Create and load the Hybrid2 model.
    
    Hybrid model always uses SwinUnet encoder, so it uses config file for encoder parameters.
    
    Args:
        args: Command line arguments containing model parameters
        config: Configuration object with model settings (required for encoder architecture)
        
    Returns:
        torch.nn.Module: Initialized Hybrid2 model ready for testing
    """
    use_baseline = getattr(args, 'use_baseline', False)
    decoder_type = getattr(args, 'decoder', 'simple')
    
    # Simplified logic (matching train.py):
    # - If --use_baseline is used without --decoder -> decoder='simple' (default)
    # - If --use_baseline is used with --decoder -> use specified decoder
    # - If --decoder is used without --use_baseline -> error (caught in validate_arguments)
    if not use_baseline:
        # No use_baseline flag -> error (should be caught in validate, but handle gracefully)
        print("ERROR: --use_baseline flag is required")
        print("Usage: --use_baseline [--decoder simple|EfficientNet-B4|ResNet50]")
        raise ValueError("--use_baseline flag is required")
    
    # Handle batch norm flag (if use_batchnorm is set, disable groupnorm)
    use_groupnorm_value = getattr(args, 'use_groupnorm', True)
    if getattr(args, 'use_batchnorm', False):
        use_groupnorm_value = False
    
    # Extract encoder parameters from config (Hybrid model uses SwinUnet encoder)
    if config is not None:
        embed_dim = config.MODEL.SWIN.EMBED_DIM if hasattr(config.MODEL, 'SWIN') and hasattr(config.MODEL.SWIN, 'EMBED_DIM') else 96
        depths = config.MODEL.SWIN.DEPTHS if hasattr(config.MODEL, 'SWIN') and hasattr(config.MODEL.SWIN, 'DEPTHS') else [2, 2, 2, 2]
        num_heads = config.MODEL.SWIN.NUM_HEADS if hasattr(config.MODEL, 'SWIN') and hasattr(config.MODEL.SWIN, 'NUM_HEADS') else [3, 6, 12, 24]
        window_size = config.MODEL.SWIN.WINDOW_SIZE if hasattr(config.MODEL, 'SWIN') and hasattr(config.MODEL.SWIN, 'WINDOW_SIZE') else 7
        drop_path_rate = config.MODEL.DROP_PATH_RATE if hasattr(config.MODEL, 'DROP_PATH_RATE') else 0.1
        img_size = config.DATA.IMG_SIZE if hasattr(config, 'DATA') and hasattr(config.DATA, 'IMG_SIZE') else args.img_size
    else:
        # Fallback to defaults if no config (should not happen, but handle gracefully)
        print("WARNING: No config provided, using default encoder parameters")
        embed_dim = 96
        depths = [2, 2, 2, 2]
        num_heads = [3, 6, 12, 24]
        window_size = 7
        drop_path_rate = 0.1
        img_size = args.img_size
    
    print("=" * 80)
    print(f"Loading Hybrid2 with {decoder_type} Decoder for Testing")
    print(f"   Encoder: SwinUnet (from config)")
    if config is not None:
        print(f"   Config: {config.MODEL.NAME if hasattr(config.MODEL, 'NAME') else 'Unknown'}")
        print(f"   Encoder params: embed_dim={embed_dim}, depths={depths}, num_heads={num_heads}")
    print("=" * 80)
    
    from hybrid2.model import create_hybrid2_baseline
    model = create_hybrid2_baseline(
        num_classes=args.num_classes,
        img_size=img_size,
        decoder=decoder_type,
        efficientnet_variant=getattr(args, 'efficientnet_variant', 'b4'),
        use_deep_supervision=getattr(args, 'use_deep_supervision', False),
        use_cbam=getattr(args, 'use_cbam', False),
        use_smart_skip=getattr(args, 'use_smart_skip', False),
        use_cross_attn=getattr(args, 'use_cross_attn', False),
        use_multiscale_agg=getattr(args, 'use_multiscale_agg', False),
        use_groupnorm=use_groupnorm_value,
        use_pos_embed=getattr(args, 'use_pos_embed', True),
        # Pass encoder config parameters from YAML
        encoder_embed_dim=embed_dim,
        encoder_depths=depths,
        encoder_num_heads=num_heads,
        encoder_window_size=window_size,
        encoder_drop_path_rate=drop_path_rate
    ).cuda()
    
    return model


def setup_logging(log_folder, snapshot_name):
    """Set up logging configuration."""
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, f"{snapshot_name}.txt")
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def setup_reproducible_testing(args):
    """Set up reproducible testing environment."""
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def load_model_checkpoint(model, args):
    """
    Load trained model checkpoint.
    
    Args:
        model: Model to load weights into
        args: Command line arguments
        
    Returns:
        str: Name of loaded checkpoint file
    """
    checkpoint_path = os.path.join(args.output_dir, 'best_model_latest.pth')
    
    if not os.path.exists(checkpoint_path):
        # Try alternative checkpoint names
        alt_path = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"No checkpoint found in {args.output_dir}")
    
    # Load checkpoint with appropriate strictness based on architecture match
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model_state from checkpoint (checkpoint may be a dict with 'model_state' key or direct state_dict)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model_state_dict = checkpoint['model_state']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    else:
        # If checkpoint is already a state_dict, use it directly (backward compatibility)
        model_state_dict = checkpoint
    
    # Load model state with strict=False to handle minor differences
    msg = model.load_state_dict(model_state_dict, strict=False)
    
    # Print loading results
    if msg.missing_keys:
        print(f"  WARNING: Missing keys (will use initialized values): {len(msg.missing_keys)}")
        if len(msg.missing_keys) <= 10:
            for key in msg.missing_keys:
                print(f"     - {key}")
        else:
            for key in msg.missing_keys[:5]:
                print(f"     - {key}")
            print(f"     ... and {len(msg.missing_keys) - 5} more")
    
    if msg.unexpected_keys:
        print(f"  WARNING: Unexpected keys (ignored): {len(msg.unexpected_keys)}")
        if len(msg.unexpected_keys) <= 10:
            for key in msg.unexpected_keys:
                print(f"     - {key}")
        else:
            for key in msg.unexpected_keys[:5]:
                print(f"     - {key}")
            print(f"     ... and {len(msg.unexpected_keys) - 5} more")
    
    if not msg.missing_keys and not msg.unexpected_keys:
        print("  Model checkpoint loaded successfully (exact match)")
    else:
        print("  Model checkpoint loaded successfully (with warnings)")
    
    return os.path.basename(checkpoint_path)


def get_dataset_info(dataset_type, manuscript=None):
    """
    Get dataset-specific information.
    
    Args:
        dataset_type (str): Type of dataset
        manuscript (str): Manuscript name for class-specific logic
        
    Returns:
        tuple: (class_colors, class_names, rgb_to_class_function)
    """
    if dataset_type.upper() == "UDIADS_BIB":
        from datasets.dataset_udiadsbib_2 import rgb_to_class
        
        class_colors = [
            (0, 0, 0),         # 0: Background (black)
            (255, 255, 0),     # 1: Paratext (yellow)
            (0, 255, 255),     # 2: Decoration (cyan)  
            (255, 0, 255),     # 3: Main Text (magenta)
            (255, 0, 0),       # 4: Title (red)
            (0, 255, 0),       # 5: Chapter Headings (lime)
        ]
        
        # Adjust class names based on manuscript
        if manuscript in ['Syr341', 'Syr341FS']:
            # Syr341 manuscripts don't have Chapter Headings
            class_names = [
                'Background', 'Paratext', 'Decoration', 
                'Main Text', 'Title'
            ]
            class_colors = class_colors[:5]  # Only first 5 colors
        else:
            class_names = [
                'Background', 'Paratext', 'Decoration', 
                'Main Text', 'Title', 'Chapter Headings'
            ]
        
        return class_colors, class_names, rgb_to_class
        
    elif dataset_type.upper() == "DIVAHISDB":
        try:
            from datasets.dataset_divahisdb import rgb_to_class
        except ImportError:
            print("Warning: DIVAHISDB dataset class not available")
            rgb_to_class = None
        
        class_colors = [
            (0, 0, 0),      # 0: Background (black)
            (0, 255, 0),    # 1: Comment (green)
            (255, 0, 0),    # 2: Decoration (red)
            (0, 0, 255),    # 3: Main Text (blue)
        ]
        
        class_names = ['Background', 'Comment', 'Decoration', 'Main Text']
        
        return class_colors, class_names, rgb_to_class
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_dataset_paths(args):
    """
    Get dataset-specific file paths.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (patch_dir, mask_dir, original_img_dir, original_mask_dir)
    """
    manuscript_name = args.manuscript
    
    if args.dataset.upper() == "UDIADS_BIB":
        if args.use_patched_data:
            patch_dir = f'{args.udiadsbib_root}/{manuscript_name}/Image/test'
            mask_dir = f'{args.udiadsbib_root}/{manuscript_name}/mask/test_labels'
        else:
            patch_dir = f'{args.udiadsbib_root}/{manuscript_name}/img-{manuscript_name}/test'
            mask_dir = f'{args.udiadsbib_root}/{manuscript_name}/pixel-level-gt-{manuscript_name}/test'
        
        # Use the original dataset directory (before patching) for original images
        # Extract the base directory name from the patched data root
        base_dir = args.udiadsbib_root.replace('_patched', '')
        original_img_dir = f'{base_dir}/{manuscript_name}/img-{manuscript_name}/test'
        original_mask_dir = f'{base_dir}/{manuscript_name}/pixel-level-gt-{manuscript_name}/test'
        
    elif args.dataset.upper() == "DIVAHISDB":
        if args.use_patched_data:
            patch_dir = f'{args.divahisdb_root}/{manuscript_name}/Image/test'
            mask_dir = f'{args.divahisdb_root}/{manuscript_name}/mask/test_labels'
        else:
            patch_dir = f'{args.divahisdb_root}/img/{manuscript_name}/test'
            mask_dir = f'{args.divahisdb_root}/pixel-level-gt/{manuscript_name}/test'
        
        # Use the original dataset directory (before patching) for original images
        # Extract the base directory name from the patched data root
        base_dir = args.divahisdb_root.replace('_patched', '')
        original_img_dir = f'{base_dir}/img/{manuscript_name}/test'
        original_mask_dir = f'{base_dir}/pixel-level-gt/{manuscript_name}/test'
    
    return patch_dir, mask_dir, original_img_dir, original_mask_dir


def process_patch_groups(patch_files):
    """
    Group patch files by their original image names.
    
    Args:
        patch_files (list): List of patch file paths
        
    Returns:
        tuple: (patch_groups, patch_positions) dictionaries
    """
    patch_groups = {}
    patch_positions = {}
    
    for patch_path in patch_files:
        filename = os.path.basename(patch_path)
        parts = filename.split('_')
        
        if len(parts) >= 2:
            original_name = '_'.join(parts[:-1])
            patch_id = int(parts[-1].split('.')[0])
            
            if original_name not in patch_groups:
                patch_groups[original_name] = []
            patch_groups[original_name].append(patch_path)
            patch_positions[patch_path] = patch_id
    
    return patch_groups, patch_positions


def estimate_image_dimensions(original_name, original_img_dir, patches, patch_positions, patch_size=224):
    """
    Estimate original image dimensions from patch information.
    
    Args:
        original_name (str): Name of original image
        original_img_dir (str): Directory containing original images
        patches (list): List of patch paths
        patch_positions (dict): Mapping of patch paths to positions
        patch_size (int): Size of each patch
        
    Returns:
        tuple: (width, height, patches_per_row)
    """
    # Try to find original image for exact dimensions
    for ext in ['.jpg', '.png', '.tif', '.tiff']:
        orig_path = os.path.join(original_img_dir, f"{original_name}{ext}")
        if os.path.exists(orig_path):
            with Image.open(orig_path) as img:
                orig_width, orig_height = img.size
            
            patches_per_row = orig_width // patch_size
            if patches_per_row == 0:
                patches_per_row = 1
            
            max_x = ((orig_width // patch_size) + (1 if orig_width % patch_size else 0)) * patch_size
            max_y = ((orig_height // patch_size) + (1 if orig_height % patch_size else 0)) * patch_size
            
            return max_x, max_y, patches_per_row
    
    # Estimate from patch positions if original not found
    logging.warning(f"Could not find original image for {original_name}, estimating dimensions")
    patches_per_row = 10  # Default fallback
    max_patch_id = max([patch_positions[p] for p in patches])
    max_x = ((max_patch_id % patches_per_row) + 1) * patch_size
    max_y = ((max_patch_id // patches_per_row) + 1) * patch_size
    
    return max_x, max_y, patches_per_row


def apply_tta_single_patch(model, patch_tensor, num_classes, return_probs=False):
    """
    Apply test-time augmentation to a single patch for higher accuracy.
    
    Averages predictions across:
    - Original
    - Horizontal flip  
    - Vertical flip
    - 90 degree rotation
    
    Args:
        model: Neural network model
        patch_tensor: Input patch tensor
        num_classes: Number of segmentation classes
        return_probs (bool): If True, return softmax probabilities instead of predictions
        
    Returns:
        np.ndarray: Augmented prediction or probabilities [H, W] or [H, W, C]
    """
    predictions = []
    
    # Original
    with torch.no_grad():
        output = model(patch_tensor)
        # Handle deep supervision
        if isinstance(output, tuple):
            output = output[0]
        predictions.append(torch.softmax(output, dim=1))
    
    # Horizontal flip
    with torch.no_grad():
        flipped = torch.flip(patch_tensor, dims=[3])
        output = model(flipped)
        if isinstance(output, tuple):
            output = output[0]
        output = torch.flip(output, dims=[3])
        predictions.append(torch.softmax(output, dim=1))
    
    # Vertical flip
    with torch.no_grad():
        flipped = torch.flip(patch_tensor, dims=[2])
        output = model(flipped)
        if isinstance(output, tuple):
            output = output[0]
        output = torch.flip(output, dims=[2])
        predictions.append(torch.softmax(output, dim=1))
    
    # 90 degree rotation
    with torch.no_grad():
        rotated = torch.rot90(patch_tensor, k=1, dims=[2, 3])
        output = model(rotated)
        if isinstance(output, tuple):
            output = output[0]
        output = torch.rot90(output, k=-1, dims=[2, 3])
        predictions.append(torch.softmax(output, dim=1))
    
    # Average all predictions
    avg_prediction = torch.stack(predictions).mean(dim=0)
    
    if return_probs:
        # Return probabilities [H, W, C]
        return avg_prediction.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    else:
        # Return class predictions [H, W]
        return torch.argmax(avg_prediction, dim=1).cpu().numpy()[0]


def stitch_patches(patches, patch_positions, max_x, max_y, patches_per_row, patch_size, model, use_tta=False, num_classes=6, return_probs=False):
    """
    Stitch together patch predictions into full image.
    
    Args:
        patches (list): List of patch file paths
        patch_positions (dict): Mapping of patch paths to positions
        max_x, max_y (int): Maximum image dimensions
        patches_per_row (int): Number of patches per row
        patch_size (int): Size of each patch
        model: Neural network model
        use_tta (bool): Whether to use test-time augmentation
        num_classes (int): Number of segmentation classes
        return_probs (bool): If True, return softmax probabilities instead of predictions
        
    Returns:
        numpy.ndarray: Stitched prediction map [H, W] or probability map [H, W, C]
    """
    import torchvision.transforms.functional as TF
    
    if return_probs:
        prob_full = np.zeros((max_y, max_x, num_classes), dtype=np.float32)
        count_map = np.zeros((max_y, max_x), dtype=np.int32)
    else:
        pred_full = np.zeros((max_y, max_x), dtype=np.int32)
        count_map = np.zeros((max_y, max_x), dtype=np.int32)
    
    for i, patch_path in enumerate(patches):
        patch_id = patch_positions[patch_path]
        
        # Calculate patch position
        x = (patch_id % patches_per_row) * patch_size
        y = (patch_id // patches_per_row) * patch_size
        
        # Load and process patch
        patch = Image.open(patch_path).convert("RGB")
        patch_tensor = TF.to_tensor(patch).unsqueeze(0).cuda()
        
        if use_tta:
            # Use TTA for improved accuracy
            if return_probs:
                patch_probs = apply_tta_single_patch(model, patch_tensor, num_classes, return_probs=True)
            else:
                pred_patch = apply_tta_single_patch(model, patch_tensor, num_classes, return_probs=False)
        else:
            # Standard inference
            with torch.no_grad():
                output = model(patch_tensor)
                # Handle deep supervision (tuple of main + aux outputs)
                if isinstance(output, tuple):
                    output = output[0]  # Use only main output for inference
                
                if return_probs:
                    patch_probs = torch.softmax(output, dim=1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
                else:
                    pred_patch = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        # Add to prediction/probability map with boundary checking
        if return_probs:
            if y + patch_size <= prob_full.shape[0] and x + patch_size <= prob_full.shape[1]:
                prob_full[y:y+patch_size, x:x+patch_size] += patch_probs
                count_map[y:y+patch_size, x:x+patch_size] += 1
            else:
                # Handle edge cases
                valid_h = min(patch_size, prob_full.shape[0] - y)
                valid_w = min(patch_size, prob_full.shape[1] - x)
                if valid_h > 0 and valid_w > 0:
                    prob_full[y:y+valid_h, x:x+valid_w] += patch_probs[:valid_h, :valid_w]
                    count_map[y:y+valid_h, x:x+valid_w] += 1
        else:
            if y + patch_size <= pred_full.shape[0] and x + patch_size <= pred_full.shape[1]:
                pred_full[y:y+patch_size, x:x+patch_size] += pred_patch
                count_map[y:y+patch_size, x:x+patch_size] += 1
            else:
                # Handle edge cases
                valid_h = min(patch_size, pred_full.shape[0] - y)
                valid_w = min(patch_size, pred_full.shape[1] - x)
                if valid_h > 0 and valid_w > 0:
                    pred_full[y:y+valid_h, x:x+valid_w] += pred_patch[:valid_h, :valid_w]
                    count_map[y:y+valid_h, x:x+valid_w] += 1
    
    # Normalize by count map
    if return_probs:
        prob_full = prob_full / np.maximum(count_map[..., np.newaxis], 1)
        return prob_full
    else:
        pred_full = np.round(pred_full / np.maximum(count_map, 1)).astype(np.uint8)
        return pred_full


def apply_crf_postprocessing(prob_map, rgb_image, num_classes=6, 
                              spatial_weight=3.0, spatial_x_stddev=3.0, spatial_y_stddev=3.0,
                              color_weight=10.0, color_stddev=50.0,
                              num_iterations=10):
    """
    Apply DenseCRF post-processing to refine segmentation predictions.
    
    Args:
        prob_map: Probability map [H, W, C] with class probabilities
        rgb_image: Original RGB image [H, W, 3] for pairwise potentials
        num_classes: Number of segmentation classes
        spatial_weight: Weight for spatial pairwise potentials
        spatial_x_stddev: Standard deviation for spatial x dimension
        spatial_y_stddev: Standard deviation for spatial y dimension
        color_weight: Weight for color pairwise potentials
        color_stddev: Standard deviation for color similarity
        num_iterations: Number of CRF iterations
        
    Returns:
        numpy.ndarray: Refined prediction map [H, W] with class indices
    """
    try:
        # pydensecrf2 package installs as 'pydensecrf' module
        try:
            import pydensecrf2.densecrf as dcrf
            from pydensecrf2.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
        except ImportError:
            # Fallback: try importing as pydensecrf (actual module name)
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
    except ImportError:
        error_msg = (
            "pydensecrf2 is not installed. CRF post-processing requires this package.\n"
            "Installation options:\n"
            "  1. pip install pydensecrf2\n"
            "  2. conda install -c conda-forge pydensecrf2\n"
            "  3. python3 -m pip install pydensecrf2\n"
            "Note: The package installs as 'pydensecrf' module even though pip package is 'pydensecrf2'.\n"
            "Note: If using conda and encountering symbol errors, try: conda install libgcc"
        )
        logging.error(error_msg)
        raise ImportError("pydensecrf2 is required for CRF post-processing")
    
    H, W = prob_map.shape[:2]
    
    # Ensure probabilities are in correct format and range
    if prob_map.shape[2] != num_classes:
        raise ValueError(f"Probability map has {prob_map.shape[2]} classes but expected {num_classes}")
    
    # Ensure prob_map is C-contiguous (required by pydensecrf)
    prob_map = np.ascontiguousarray(prob_map, dtype=np.float32)
    
    # Resize RGB image if dimensions don't match
    if rgb_image.shape[:2] != (H, W):
        rgb_image_resized = np.array(Image.fromarray(rgb_image).resize((W, H), Image.BILINEAR))
    else:
        rgb_image_resized = rgb_image.copy()
    
    # Ensure RGB image is uint8 and C-contiguous (required by pydensecrf)
    if rgb_image_resized.dtype != np.uint8:
        rgb_image_resized = np.clip(rgb_image_resized, 0, 255).astype(np.uint8)
    rgb_image_resized = np.ascontiguousarray(rgb_image_resized, dtype=np.uint8)
    
    # Transpose probability map to [C, H, W] format for DenseCRF
    prob_map_transposed = prob_map.transpose(2, 0, 1)  # [C, H, W]
    # Ensure transposed array is C-contiguous (required by pydensecrf)
    prob_map_transposed = np.ascontiguousarray(prob_map_transposed, dtype=np.float32)
    
    # Create CRF model
    crf = dcrf.DenseCRF2D(W, H, num_classes)
    
    # Set unary potentials (negative log probabilities)
    unary = unary_from_softmax(prob_map_transposed)
    crf.setUnaryEnergy(unary)
    
    # Add pairwise potentials
    
    # 1. Spatial pairwise potential (encourages nearby pixels to have same label)
    pairwise_gaussian = create_pairwise_gaussian(sdims=(spatial_y_stddev, spatial_x_stddev), shape=(H, W))
    crf.addPairwiseEnergy(pairwise_gaussian, compat=spatial_weight)
    
    # 2. Bilateral pairwise potential (encourages similar colored pixels to have same label)
    pairwise_bilateral = create_pairwise_bilateral(
        sdims=(spatial_y_stddev, spatial_x_stddev),
        schan=(color_stddev, color_stddev, color_stddev),
        img=rgb_image_resized,
        chdim=2
    )
    crf.addPairwiseEnergy(pairwise_bilateral, compat=color_weight)
    
    # Run inference
    Q = crf.inference(num_iterations)
    
    # Get refined probabilities and convert to prediction
    refined_probs = np.array(Q).reshape((num_classes, H, W)).transpose(1, 2, 0)
    refined_pred = np.argmax(refined_probs, axis=2).astype(np.uint8)
    
    return refined_pred


def save_prediction_results(pred_full, original_name, class_colors, result_dir):
    """Save prediction results as RGB image."""
    if result_dir is not None:
        os.makedirs(result_dir, exist_ok=True)
        
        # Convert class indices to RGB
        rgb_mask = np.zeros((pred_full.shape[0], pred_full.shape[1], 3), dtype=np.uint8)
        for idx, color in enumerate(class_colors):
            rgb_mask[pred_full == idx] = color
        
        pred_png_path = os.path.join(result_dir, f"{original_name}.png")
        Image.fromarray(rgb_mask).save(pred_png_path)


def save_comparison_visualization(pred_full, gt_class, original_name, original_img_dir, 
                                test_save_path, class_colors, class_names):
    """Save side-by-side comparison visualization."""
    
    compare_dir = os.path.join(test_save_path, 'compare')
    os.makedirs(compare_dir, exist_ok=True)
    
    cmap = ListedColormap(class_colors)
    n_classes = len(class_colors)
    
    # Resize ground truth if dimensions don't match
    if gt_class.shape != pred_full.shape:
        logging.warning(f"Resizing ground truth for {original_name}")
        gt_class_resized = np.zeros_like(pred_full)
        min_h = min(gt_class.shape[0], pred_full.shape[0])
        min_w = min(gt_class.shape[1], pred_full.shape[1])
        gt_class_resized[:min_h, :min_w] = gt_class[:min_h, :min_w]
        gt_class = gt_class_resized
    
    # Create visualization
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Find and load original image
    orig_img_path = None
    for ext in ['.jpg', '.png', '.tif', '.tiff']:
        test_path = os.path.join(original_img_dir, f"{original_name}{ext}")
        if os.path.exists(test_path):
            orig_img_path = test_path
            break
    
    if orig_img_path:
        orig_img = Image.open(orig_img_path).convert("RGB")
        if orig_img.size != (pred_full.shape[1], pred_full.shape[0]):
            orig_img = orig_img.resize((pred_full.shape[1], pred_full.shape[0]), Image.BILINEAR)
        axs[0].imshow(np.array(orig_img))
    else:
        axs[0].imshow(np.zeros((pred_full.shape[0], pred_full.shape[1], 3), dtype=np.uint8))
    
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(pred_full, cmap=cmap, vmin=0, vmax=(n_classes - 1))
    axs[1].set_title('Prediction')
    axs[1].axis('off')
    
    axs[2].imshow(gt_class, cmap=cmap, vmin=0, vmax=(n_classes - 1))
    axs[2].set_title('Ground Truth')
    axs[2].axis('off')
    
    plt.tight_layout()
    save_img_path = os.path.join(compare_dir, f"{original_name}_compare.png")
    plt.savefig(save_img_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def compute_segmentation_metrics(pred_full, gt_class, n_classes, TP, FP, FN):
    """Compute segmentation metrics for each class."""
    for cls in range(n_classes):
        pred_c = (pred_full == cls)
        gt_c = (gt_class == cls)
        TP[cls] += np.logical_and(pred_c, gt_c).sum()
        FP[cls] += np.logical_and(pred_c, np.logical_not(gt_c)).sum()
        FN[cls] += np.logical_and(np.logical_not(pred_c), gt_c).sum()


def print_final_metrics(TP, FP, FN, class_names, num_processed_images):
    """Print final computed metrics."""
    n_classes = len(class_names)
    
    if num_processed_images > 0:
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou_per_class = TP / (TP + FP + FN + 1e-8)
    else:
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1 = np.zeros(n_classes)
        iou_per_class = np.zeros(n_classes)
    
    print("\nPer-class metrics:")
    print("-" * 80)
    for cls in range(n_classes):
        print(f"{class_names[cls]:<15}: Precision={precision[cls]:.4f}, "
              f"Recall={recall[cls]:.4f}, F1={f1[cls]:.4f}, IoU={iou_per_class[cls]:.4f}")
    
    print("\nMean metrics:")
    print("-" * 40)
    print(f"Mean Precision: {np.mean(precision):.4f}")
    print(f"Mean Recall: {np.mean(recall):.4f}")
    print(f"Mean F1-Score: {np.mean(f1):.4f}")
    print(f"Mean IoU: {np.mean(iou_per_class):.4f}")
    
    logging.info("\nPer-class metrics:")
    logging.info("-" * 80)
    for cls in range(n_classes):
        logging.info(f"{class_names[cls]:<15}: Precision={precision[cls]:.4f}, "
                     f"Recall={recall[cls]:.4f}, F1={f1[cls]:.4f}, IoU={iou_per_class[cls]:.4f}")
    
    logging.info("\nMean metrics:")
    logging.info("-" * 40)
    logging.info(f"Mean Precision: {np.mean(precision):.4f}")
    logging.info(f"Mean Recall: {np.mean(recall):.4f}")
    logging.info(f"Mean F1-Score: {np.mean(f1):.4f}")
    logging.info(f"Mean IoU: {np.mean(iou_per_class):.4f}")


def save_metrics_to_file(args, TP, FP, FN, class_names, num_processed_images):
    """Save metrics to a JSON file for later aggregation."""
    n_classes = len(class_names)
    
    if num_processed_images > 0:
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou_per_class = TP / (TP + FP + FN + 1e-8)
        
        mean_precision = float(np.mean(precision))
        mean_recall = float(np.mean(recall))
        mean_f1 = float(np.mean(f1))
        mean_iou = float(np.mean(iou_per_class))
    else:
        mean_precision = mean_recall = mean_f1 = mean_iou = 0.0
    
    parent_dir = os.path.dirname(args.output_dir) if os.path.basename(args.output_dir) == args.manuscript else args.output_dir
    os.makedirs(parent_dir, exist_ok=True)
    
    metrics_file = os.path.join(parent_dir, f"metrics_{args.manuscript}.json")
    
    metrics_data = {
        "manuscript": args.manuscript,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "mean_iou": mean_iou,
        "num_images": num_processed_images
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    return parent_dir


def calculate_and_display_average_metrics(args):
    """Calculate and display average metrics across all manuscripts."""
    if 'FS' in args.manuscript or args.manuscript.endswith('FS'):
        expected_manuscripts = ['Latin2FS', 'Latin14396FS', 'Latin16746FS', 'Syr341FS']
    else:
        expected_manuscripts = ['Latin2', 'Latin14396', 'Latin16746', 'Syr341']
    
    parent_dir = os.path.dirname(args.output_dir) if os.path.basename(args.output_dir) == args.manuscript else args.output_dir
    
    all_metrics = []
    found_manuscripts = []
    metrics_files = []
    
    print("\nCalculating average metrics...")
    print(f"Looking for metrics files in: {parent_dir}")
    
    for manuscript in expected_manuscripts:
        metrics_file = os.path.join(parent_dir, f"metrics_{manuscript}.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    all_metrics.append(data)
                    found_manuscripts.append(manuscript)
                    metrics_files.append(metrics_file)
                    print(f"  Found metrics for {manuscript}")
            except Exception as e:
                logging.warning(f"Failed to load metrics for {manuscript}: {e}")
                print(f"  Metrics file not found: {metrics_file}")
        else:
            print(f"  Metrics file not found: {metrics_file}")
    
    if len(found_manuscripts) > 0:
        avg_precision = sum(m['mean_precision'] for m in all_metrics) / len(all_metrics)
        avg_recall = sum(m['mean_recall'] for m in all_metrics) / len(all_metrics)
        avg_f1 = sum(m['mean_f1'] for m in all_metrics) / len(all_metrics)
        avg_iou = sum(m['mean_iou'] for m in all_metrics) / len(all_metrics)
        
        print("\n" + "="*80)
        if len(found_manuscripts) == len(expected_manuscripts):
            print("AVERAGE METRICS ACROSS ALL MANUSCRIPTS")
        else:
            print(f"AVERAGE METRICS ACROSS {len(found_manuscripts)} MANUSCRIPT(S)")
        print("="*80)
        print(f"Manuscripts: {', '.join(found_manuscripts)}")
        if len(found_manuscripts) < len(expected_manuscripts):
            missing = [m for m in expected_manuscripts if m not in found_manuscripts]
            print(f"Missing: {', '.join(missing)}")
        print("-"*80)
        print(f"Mean Precision: {avg_precision:.4f}")
        print(f"Mean Recall:    {avg_recall:.4f}")
        print(f"Mean F1-Score:  {avg_f1:.4f}")
        print(f"Mean IoU:       {avg_iou:.4f}")
        print("="*80)
        sys.stdout.flush()
        
        # Clean up temporary metrics files
        for metrics_file in metrics_files:
            try:
                os.remove(metrics_file)
            except Exception as e:
                logging.warning(f"Failed to delete temporary metrics file {metrics_file}: {e}")
        
        return True
    else:
        print("No metrics files found for aggregation.")
        return False
    


def inference(args, model, test_save_path=None):
    """
    Run inference on historical document dataset.
    
    Args:
        args: Command line arguments
        model: Trained neural network model
        test_save_path: Path to save test results
        
    Returns:
        str: Status message
    """
    logging.info(f"Starting inference on {args.dataset} dataset")
    
    # Get dataset-specific information
    class_colors, class_names, rgb_to_class_func = get_dataset_info(args.dataset, args.manuscript)
    n_classes = len(class_colors)
    # Use the actual patch size that the model was trained on (224x224)
    # args.img_size is the full image size, not the patch size
    patch_size = 224
    
    # Get dataset paths
    patch_dir, mask_dir, original_img_dir, original_mask_dir = get_dataset_paths(args)
    
    # Check if directories exist
    if not os.path.exists(patch_dir):
        logging.error(f"Patch directory not found: {patch_dir}")
        return "Testing Failed!"
    
    # Find patch files
    patch_files = sorted(glob.glob(os.path.join(patch_dir, '*.png')))
    if len(patch_files) == 0:
        logging.info(f"No patch files found in {patch_dir}")
        return "Testing Finished!"
    
    logging.info(f"Found {len(patch_files)} patches for {args.manuscript}")
    
    use_crf = getattr(args, 'use_crf', False)
    
    if args.use_tta:
        logging.info("Using Test-Time Augmentation (TTA) for improved accuracy")
    else:
        logging.info("Using standard inference (no TTA)")
    
    if use_crf:
        logging.info("CRF post-processing: Enabled")
        logging.info("  - DenseCRF with spatial and color pairwise potentials")
    else:
        logging.info("CRF post-processing: Disabled")
    
    # Initialize metrics
    TP = np.zeros(n_classes, dtype=np.float64)
    FP = np.zeros(n_classes, dtype=np.float64)  
    FN = np.zeros(n_classes, dtype=np.float64)
    
    # Set up result directory
    result_dir = os.path.join(test_save_path, "result") if test_save_path else None
    
    # Group patches by original image
    patch_groups, patch_positions = process_patch_groups(patch_files)
    
    # Process each original image
    num_processed_images = 0
    
    for original_name, patches in patch_groups.items():
        logging.info(f"Processing: {original_name} ({len(patches)} patches)")
        
        # Estimate image dimensions
        max_x, max_y, patches_per_row = estimate_image_dimensions(
            original_name, original_img_dir, patches, patch_positions, patch_size
        )
        
        # Stitch patches together
        if use_crf:
            # Get both predictions and probabilities for CRF
            prob_full = stitch_patches(
                patches, patch_positions, max_x, max_y, 
                patches_per_row, patch_size, model,
                use_tta=args.use_tta, num_classes=n_classes, return_probs=True
            )
            
            # Load original RGB image for CRF pairwise potentials
            orig_img_rgb = None
            for ext in ['.jpg', '.png', '.tif', '.tiff']:
                orig_img_path = os.path.join(original_img_dir, f"{original_name}{ext}")
                if os.path.exists(orig_img_path):
                    orig_img_pil = Image.open(orig_img_path).convert("RGB")
                    orig_img_rgb = np.array(orig_img_pil)
                    break
            
            if orig_img_rgb is not None:
                logging.info(f"Applying CRF post-processing to {original_name}")
                try:
                    # Apply CRF refinement
                    pred_full = apply_crf_postprocessing(
                        prob_full, orig_img_rgb, 
                        num_classes=n_classes,
                        spatial_weight=3.0,
                        spatial_x_stddev=3.0,
                        spatial_y_stddev=3.0,
                        color_weight=10.0,
                        color_stddev=50.0,
                        num_iterations=10
                    )
                    logging.info(f"CRF post-processing completed for {original_name}")
                except Exception as e:
                    logging.warning(f"CRF post-processing failed for {original_name}: {e}")
                    logging.warning("Falling back to non-CRF predictions")
                    # Fall back to argmax if CRF fails
                    pred_full = np.argmax(prob_full, axis=2).astype(np.uint8)
            else:
                logging.warning(f"Original image not found for CRF: {original_name}, using non-CRF predictions")
                pred_full = np.argmax(prob_full, axis=2).astype(np.uint8)
        else:
            # Standard inference without CRF
            pred_full = stitch_patches(
                patches, patch_positions, max_x, max_y, 
                patches_per_row, patch_size, model,
                use_tta=args.use_tta, num_classes=n_classes, return_probs=False
            )
        
        # Save prediction results
        save_prediction_results(pred_full, original_name, class_colors, result_dir)
        
        # Load ground truth for evaluation
        gt_found = False
        for ext in ['.png', '.jpg', '.tif', '.tiff']:
            gt_path = os.path.join(original_mask_dir, f"{original_name}{ext}")
            if os.path.exists(gt_path):
                gt_pil = Image.open(gt_path).convert("RGB")
                gt_np = np.array(gt_pil)
                
                if rgb_to_class_func:
                    gt_class = rgb_to_class_func(gt_np)
                    gt_found = True
                    break
        
        if not gt_found:
            logging.warning(f"No ground truth found for {original_name}")
            gt_class = np.zeros_like(pred_full)
        
        # Save comparison visualization (matching Network model)
        if test_save_path and gt_found:
            save_comparison_visualization(
                pred_full, gt_class, original_name, original_img_dir,
                test_save_path, class_colors, class_names
            )
        
        # Compute metrics
        if gt_found:
            # Ensure ground truth has same dimensions as prediction
            if gt_class.shape != pred_full.shape:
                logging.warning(f"Resizing ground truth for metrics computation: {gt_class.shape} -> {pred_full.shape}")
                gt_class_resized = np.zeros_like(pred_full)
                min_h = min(gt_class.shape[0], pred_full.shape[0])
                min_w = min(gt_class.shape[1], pred_full.shape[1])
                gt_class_resized[:min_h, :min_w] = gt_class[:min_h, :min_w]
                gt_class = gt_class_resized
            
            compute_segmentation_metrics(pred_full, gt_class, n_classes, TP, FP, FN)
            num_processed_images += 1
        
        logging.info(f"Completed: {original_name}")
    
    # Print final metrics
    print_final_metrics(TP, FP, FN, class_names, num_processed_images)
    logging.info(f"Inference completed on {num_processed_images} images")
    
    # Save metrics to JSON file for aggregation
    save_metrics_to_file(args, TP, FP, FN, class_names, num_processed_images)
    
    return "Testing Finished!"


def main():
    """Main testing function."""
    print("=== Historical Document Segmentation Testing ===")
    print()
    
    # Parse and validate arguments
    args = parse_arguments()
    
    try:
        validate_arguments(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Set up reproducible testing
    setup_reproducible_testing(args)
    
    # Load configuration (required for hybrid model to match encoder architecture)
    print("Loading configuration for Hybrid2 (SwinUnet encoder)...")
    print(f"  Config file: {args.cfg}")
    config = get_config(args)
    print(f"  Config name: {config.MODEL.NAME if hasattr(config.MODEL, 'NAME') else 'Unknown'}")
    
    # Print configuration
    print(f"Model: Hybrid2 (SwinUnet Encoder + {getattr(args, 'decoder', 'simple')} Decoder)")
    print(f"Dataset: {args.dataset}")
    print(f"Manuscript: {args.manuscript}")
    print(f"Test-Time Augmentation: {'Enabled' if args.use_tta else 'Disabled'}")
    print(f"CRF post-processing: {'Enabled' if args.use_crf else 'Disabled'}")
    if args.use_crf:
        print("  - DenseCRF with spatial and color pairwise potentials")
    print()
    
    # Create model (now requires config for encoder architecture)
    model = get_model(args, config)
    
    # Load trained model checkpoint
    try:
        checkpoint_name = load_model_checkpoint(model, args)
        print(f"Loaded checkpoint: {checkpoint_name}")
    except Exception as e:
        print(f"ERROR: Failed to load model checkpoint: {e}")
        sys.exit(1)
    
    # Set up logging
    log_folder = './test_log/test_log_'
    setup_logging(log_folder, checkpoint_name)
    
    logging.info(str(args))
    logging.info(f"Testing with checkpoint: {checkpoint_name}")
    
    # Set up test save directory
    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, "predictions")
        os.makedirs(test_save_path, exist_ok=True)
        logging.info(f"Saving predictions to: {test_save_path}")
    else:
        test_save_path = None
        logging.info("Not saving prediction files")
    
    # Run inference
    print()
    print("=== Starting Testing ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model: Hybrid2")  # Hardcoded instead of args.model
    print(f"Manuscript: {args.manuscript}")
    print(f"Save predictions: {args.is_savenii}")
    print()
    
    try:
        result = inference(args, model, test_save_path)
        print()
        print("=== TESTING COMPLETED SUCCESSFULLY ===")
        print(f"Results saved to: {test_save_path if test_save_path else 'No files saved'}")
        print("="*50)
        
        # Calculate and display average metrics across all manuscripts
        calculate_and_display_average_metrics(args)
        
        return result
        
    except Exception as e:
        print(f"ERROR: Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
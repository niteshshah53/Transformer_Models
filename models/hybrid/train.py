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
import torch.nn.functional as F

# Add common directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import training modules
from trainer import trainer_hybrid
from configs.config import get_config

# Import dataset classes
from datasets.dataset_udiadsbib_2 import UDiadsBibDataset
try:
    from datasets.dataset_divahisdb import DivaHisDBDataset
    DIVAHISDB_AVAILABLE = True
except ImportError:
    DivaHisDBDataset = None
    DIVAHISDB_AVAILABLE = False


def _is_simmim_config(config):
    """
    Check if the SimMIM config is being used.

    Args:
        config: Configuration object

    Returns:
        bool: True if SimMIM config is detected
    """
    # Check config name
    if hasattr(config.MODEL, 'NAME') and 'simmim' in str(config.MODEL.NAME).lower():
        return True

    # Check pretrained checkpoint path
    if hasattr(config.MODEL, 'PRETRAIN_CKPT') and config.MODEL.PRETRAIN_CKPT:
        pretrained_path = str(config.MODEL.PRETRAIN_CKPT)
        if 'simmim' in pretrained_path.lower():
            return True

    return False


def _resize_relative_position_bias_table(table, old_window, new_window):
    """
    Interpolate relative position bias table from old_window to new_window.

    Args:
        table: Relative position bias table tensor with shape (old_size * old_size, num_heads)
        old_window: Original window size (e.g., 6 for SimMIM)
        new_window: Target window size (e.g., 7 for Swin encoder)

    Returns:
        Interpolated table with shape (new_size * new_size, num_heads)
    """
    old_size = 2 * old_window - 1
    new_size = 2 * new_window - 1

    # Reshape to (1, num_heads, old_size, old_size)
    num_heads = table.shape[-1]
    table = table.reshape(old_size, old_size, num_heads).permute(2, 0, 1).unsqueeze(0)

    # Interpolate using bicubic interpolation
    table = F.interpolate(table, size=(new_size, new_size), mode='bicubic', align_corners=False)

    # Reshape back to (new_size * new_size, num_heads)
    table = table.squeeze(0).permute(1, 2, 0).reshape(new_size * new_size, num_heads)

    return table


def load_pretrained_encoder_weights(model, config):
    """
    Load pretrained Swin encoder weights from config.
    
    Hybrid model always uses SwinUnet encoder, so it loads encoder weights from the config.
    
    Args:
        model: Hybrid2 model
        config: Configuration object with MODEL.PRETRAIN_CKPT path
    """
    pretrained_path = config.MODEL.PRETRAIN_CKPT
    if pretrained_path is None:
        print("  No pretrained checkpoint path in config, skipping weight loading")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if path is relative and resolve it
    if not os.path.isabs(pretrained_path):
        # Resolve relative to config file location
        config_file = getattr(config, '_config_file', None)
        if config_file:
            config_dir = os.path.dirname(os.path.abspath(config_file))
            pretrained_path = os.path.join(config_dir, pretrained_path)
        
        # If still not found, try relative to common/configs
        if not os.path.exists(pretrained_path):
            base_dir = os.path.join(os.path.dirname(__file__), '../../')
            pretrained_path = os.path.join(base_dir, config.MODEL.PRETRAIN_CKPT)
    
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
    
    print(f"  Loading pretrained encoder weights from: {pretrained_path}")
    pretrained_dict = torch.load(pretrained_path, map_location=device)

    # Handle different checkpoint formats
    if "model" in pretrained_dict:
        pretrained_dict = pretrained_dict['model']

    # Handle SimMIM relative position bias interpolation (window_size 6 -> 7)
    is_simmim = _is_simmim_config(config)
    old_window_size = 6  # SimMIM pretraining uses window_size=6
    new_window_size = getattr(config.MODEL.SWIN, 'WINDOW_SIZE', 7) if hasattr(config.MODEL, 'SWIN') else 7

    if is_simmim and old_window_size != new_window_size:
        interpolated_count = 0
        old_size = (2 * old_window_size - 1) ** 2
        for k in list(pretrained_dict.keys()):
            if "relative_position_bias_table" in k:
                tensor = pretrained_dict[k]
                if tensor.shape[0] == old_size:
                    pretrained_dict[k] = _resize_relative_position_bias_table(
                        tensor, old_window_size, new_window_size
                    )
                    interpolated_count += 1
        if interpolated_count > 0:
            print(
                f"  SimMIM config detected: interpolated {interpolated_count} "
                f"relative position bias tables from window_size={old_window_size} "
                f"to window_size={new_window_size} for encoder"
            )

    # Get encoder state dict
    encoder_dict = model.encoder.state_dict()

    # Filter pretrained dict to only include encoder weights
    # Swin encoder keys typically start with: patch_embed, layers, norm
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        # Skip decoder/upsample layers (layers_up) - these are for SwinUnet decoder, not encoder
        if "layers_up" in k or "output" in k or "decode_head" in k:
            continue

        # Map encoder layers
        if k in encoder_dict:
            if encoder_dict[k].shape == v.shape:
                filtered_dict[k] = v
        # Handle layers. prefix (Swin format)
        elif "layers." in k:
            # Keep as is for encoder
            if k in encoder_dict and encoder_dict[k].shape == v.shape:
                filtered_dict[k] = v

    # Load filtered weights
    msg = model.encoder.load_state_dict(filtered_dict, strict=False)
    if msg.missing_keys:
        print(f"  WARNING: Missing encoder keys: {len(msg.missing_keys)} (this is OK, decoder keys are expected)")
    if msg.unexpected_keys:
        print(f"  WARNING: Unexpected encoder keys: {len(msg.unexpected_keys)}")
    if not msg.missing_keys and not msg.unexpected_keys:
        print(f"  [OK] All encoder weights loaded successfully")


def get_model(args, config):
    """
    Create and initialize the Hybrid2 model.
    
    Hybrid model always uses SwinUnet encoder, so it uses config file for encoder parameters.
    
    Args:
        args: Command line arguments containing model parameters
        config: Configuration object with model settings (required for pretrained encoder weights)
        
    Returns:
        torch.nn.Module: Initialized Hybrid2 model ready for training
    """
    use_baseline = getattr(args, 'use_baseline', False)
    decoder_type = getattr(args, 'decoder', 'simple')
    
    # Simplified logic:
    # - If --use_baseline is used without --decoder -> decoder='simple' (default)
    # - If --use_baseline is used with --decoder -> use specified decoder
    # - If --decoder is used without --use_baseline -> error (caught in validate_arguments)
    if use_baseline:
        # use_baseline is set, use specified decoder (or 'simple' if not specified)
        decoder_type = decoder_type  # Already set from args
    else:
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
    
    # Always use baseline with configurable decoder (flags control enhancements)
    print("=" * 80)
    print(f"Loading Hybrid2 with {decoder_type} Decoder")
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
        fusion_method=getattr(args, 'fusion_method', 'simple'),
        # Pass encoder config parameters from YAML
        encoder_embed_dim=embed_dim,
        encoder_depths=depths,
        encoder_num_heads=num_heads,
        encoder_window_size=window_size,
        encoder_drop_path_rate=drop_path_rate
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to CUDA")
    else:
        print("CUDA not available, using CPU")
    
    # Load pretrained encoder weights from config (Hybrid model uses SwinUnet encoder)
    if config is not None:
        try:
            load_pretrained_encoder_weights(model, config)
            print("[OK] Pretrained encoder weights loaded successfully")
        except FileNotFoundError as e:
            print(f"WARNING: Pretrained checkpoint not found: {e}")
            print("   Continuing training without pretrained weights (random initialization)")
        except Exception as e:
            print(f"WARNING: Failed to load pretrained weights: {e}")
            print("   Continuing training without pretrained weights (random initialization)")
    
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
            print("[OK] Class-aware augmentation enabled (stronger augmentation for rare classes)")
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
    print(f"Using config file: {args.cfg}")
    if args.yaml:
        print(f"Config preset: {args.yaml} (Hybrid model uses SwinUnet encoder)")


def parse_arguments():
    """
    Parse command line arguments for Hybrid model training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Hybrid2 Model Training for Historical Document Segmentation')
    
    # Model configuration
    # Config file support (required for hybrid model since it uses SwinUnet encoder)
    parser.add_argument('--cfg', type=str, required=False, 
                       help='Path to config file (optional when using --yaml)')
    parser.add_argument('--yaml', type=str, default='swintiny',
                       choices=['swintiny', 'simmim'],
                       help="Choose which preset YAML to use from common/configs. Default: 'swintiny'. Options: 'swintiny' or 'simmim'. If provided, --cfg is optional and will be overridden.")
    
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
    parser.add_argument('--fusion_method', type=str, default='simple',
                       choices=['simple', 'smart', 'fourier'],
                       help='Feature fusion method: simple (concat), smart (CBAM attention), or fourier (FFT-based)')
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
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=300, help='Maximum number of epochs')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='Base learning rate')
    parser.add_argument('--patience', type=int, default=70, help='Early stopping patience')
    parser.add_argument('--scheduler_type', type=str, default='OneCycleLR',
                       choices=['CosineAnnealingWarmRestarts', 'OneCycleLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'],
                       help='Learning rate scheduler type')
    parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
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
    
    # Load configuration (required for hybrid model to load pretrained encoder weights)
    print("Loading configuration for Hybrid2 (SwinUnet encoder)...")
    print(f"  Config file: {args.cfg}")
    config = get_config(args)
    print(f"  Config name: {config.MODEL.NAME if hasattr(config.MODEL, 'NAME') else 'Unknown'}")
    if hasattr(config.MODEL, 'PRETRAIN_CKPT') and config.MODEL.PRETRAIN_CKPT:
        print(f"  Pretrained checkpoint: {config.MODEL.PRETRAIN_CKPT}")
    
    # Set up datasets
    train_dataset, val_dataset = setup_datasets(args)
    
    # Create balanced sampler if requested
    if getattr(args, 'use_balanced_sampler', False):
        from trainer import create_balanced_sampler
        balanced_sampler = create_balanced_sampler(train_dataset, args.num_classes)
        args.balanced_sampler = balanced_sampler
        if balanced_sampler is not None:
            print("[OK] Balanced sampler enabled (oversampling rare classes)")
        else:
            print("WARNING: Balanced sampler requested but could not be created (using default shuffling)")
            args.balanced_sampler = None
    else:
        args.balanced_sampler = None
    
    # Create model (now requires config for pretrained encoder weights)
    model = get_model(args, config)
    print("Model created successfully with {} classes".format(args.num_classes))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Debug: Verify output_dir is correct (should not include dataset name)
    print(f"DEBUG: args.output_dir = {args.output_dir}")
    
    # Start training
    print("\n=== Starting Training ===")
    print("Dataset: {}".format(args.dataset))
    print("Model: Hybrid2 (SwinUnet Encoder + {} Decoder)".format(getattr(args, 'decoder', 'simple')))
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
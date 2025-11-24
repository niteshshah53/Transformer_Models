"""
Testing Script for CNN-Transformer Historical Document Segmentation Models

This script evaluates trained CNN-Transformer models on historical document test datasets.
"""

import argparse
import logging
import os
import random
import sys
import warnings
import glob
import json

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torchvision.transforms.functional as TF

sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments for testing script."""
    parser = argparse.ArgumentParser(
        description='Test CNN-Transformer model on historical document datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py --output_dir ./models/ --dataset UDIADS_BIB --manuscript Latin2 --is_savenii
  python test.py --dataset DIVAHISDB --output_dir ./models/ --manuscript Latin2
        """
    )
    
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing trained model checkpoints')
    parser.add_argument('--dataset', type=str, default='UDIADS_BIB',
                       choices=['UDIADS_BIB', 'DIVAHISDB'],
                       help='Dataset to test on')
    parser.add_argument('--manuscript', type=str, required=True,
                       help='Manuscript to test')
    parser.add_argument('--udiadsbib_root', type=str, default='../../U-DIADS-Bib-MS',
                       help='Root directory for U-DIADS-Bib dataset')
    parser.add_argument('--divahisdb_root', type=str, default='../../DivaHisDB',
                       help='Root directory for DIVAHISDB dataset')
    parser.add_argument('--use_patched_data', action='store_true',
                       help='Use pre-generated patches instead of full images')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of segmentation classes (auto-detected)')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input patch size for inference')
    parser.add_argument('--batch_size', type=int, default=24,
                       help='Batch size for testing')
    # Architecture flags (all independent, matching train.py)
    parser.add_argument('--bottleneck', action='store_true', default=True,
                       help='Enable bottleneck with 2 Swin Transformer blocks (default: True)')
    parser.add_argument('--no_bottleneck', dest='bottleneck', action='store_false',
                       help='Disable bottleneck')
    parser.add_argument('--adapter_mode', type=str, default='streaming',
                       choices=['external', 'streaming'],
                       help='Adapter placement mode: external (separate adapters) or streaming (integrated adapters) (default: streaming)')
    parser.add_argument('--fusion_method', type=str, default='simple',
                       choices=['simple', 'fourier', 'smart', 'gcff'],
                       help='Feature fusion method: simple (concat), fourier (FFT-based), smart (attention-based smart skip connections), gcff (Global Context Feature Fusion from MSAGHNet) (default: simple)')
    parser.add_argument('--deep_supervision', action='store_true', default=False,
                       help='Enable deep supervision with 3 auxiliary outputs (default: False)')
    parser.add_argument('--use_multiscale_agg', action='store_true', default=False,
                       help='Enable multi-scale aggregation in bottleneck (default: False)')
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
    
    # Legacy flag (kept for backward compatibility, but not used)
    parser.add_argument('--use_baseline', action='store_true', default=False,
                       help='[DEPRECATED: All flags are now independent] Use baseline CNN-Transformer configuration')
    
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                       help='Freeze encoder during testing')
    parser.add_argument('--is_savenii', action="store_true",
                       help='Save prediction results during inference')
    parser.add_argument('--test_save_dir', type=str, default='../predictions',
                       help='Directory to save prediction results')
    parser.add_argument('--use_tta', action='store_true', default=False,
                       help='Enable test-time augmentation')
    parser.add_argument('--use_crf', action='store_true', default=False,
                       help='Enable CRF post-processing')
    parser.add_argument('--deterministic', type=int, default=1,
                       help='Use deterministic testing')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed for reproducibility')
    # Testing-specific flags (not in train.py, but needed for testing)
    # Note: Legacy/unused flags removed for clarity
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate and normalize command line arguments."""
    bad_tokens = [t for t in sys.argv[1:] if t.lstrip('-').startswith('mg_')]
    if bad_tokens:
        logging.warning(f"Suspicious argv tokens detected: {bad_tokens}")
    
    if not os.path.exists(args.output_dir):
        raise ValueError(f"Output directory not found: {args.output_dir}")
    
    if args.dataset.upper() == "UDIADS_BIB":
        if args.manuscript in ['Syr341FS', 'Syr341']:
            args.num_classes = 5
        else:
            args.num_classes = 6
        
        if not os.path.exists(args.udiadsbib_root):
            raise ValueError(f"U-DIADS-Bib dataset path not found: {args.udiadsbib_root}")
    elif args.dataset.upper() == "DIVAHISDB":
        args.num_classes = 4
        if not os.path.exists(args.divahisdb_root):
            raise ValueError(f"DIVAHISDB dataset path not found: {args.divahisdb_root}")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # All architecture flags are independent (matching train.py)
    # No need to validate use_baseline flag - all flags work independently


def get_model(args, config):
    """
    Create and load the CNN-Transformer model.
    
    All architecture flags are independent and can be combined freely (matching train.py).
    """
    from vision_transformer_cnn import CNNTransformerUnet as ViT_seg
    
    # Get all model configuration from args (all flags are independent, matching train.py)
    use_bottleneck = getattr(args, 'bottleneck', True)
    adapter_mode = getattr(args, 'adapter_mode', 'streaming')
    fusion_method = getattr(args, 'fusion_method', 'simple')
    use_deep_supervision = getattr(args, 'deep_supervision', False)
    use_multiscale_agg = getattr(args, 'use_multiscale_agg', False)
    use_groupnorm = getattr(args, 'use_groupnorm', True)
    use_se_msfe = getattr(args, 'use_se_msfe', False)
    use_msfa_mct_bottleneck = getattr(args, 'use_msfa_mct_bottleneck', False)
    encoder_type = getattr(args, 'encoder_type', 'efficientnet')  # 'efficientnet' or 'resnet50'
    
    # Print configuration (matching train.py format)
    print("=" * 80)
    print("🚀 Loading CNN-Transformer Model for Testing")
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
    
    # Create model with all flags (all independent and compatible, matching train.py)
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
    
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        logging.warning("CUDA not available, using CPU")
    
    if getattr(args, 'freeze_encoder', False):
        model.model.freeze_encoder()
    
    model.load_from(None)
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
    """Set up reproducible testing environment.
    
    IMPORTANT: For full reproducibility, set args.deterministic=True.
    This enables PyTorch's deterministic algorithms and cuDNN deterministic mode.
    Without this, results may vary across runs due to non-deterministic operations.
    """
    if args.deterministic:
        # Enable deterministic mode for reproducible results
        cudnn.benchmark = False
        cudnn.deterministic = True
        # Enable PyTorch's deterministic algorithms (critical for reproducibility)
        # This ensures atomic operations (scatter, gather) are deterministic
        torch.use_deterministic_algorithms(True)
        # Set CUBLAS workspace config to silence warnings about non-deterministic ops
        # This is required for deterministic behavior in some CUDA operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        logging.info("✓ Deterministic mode enabled (fully reproducible results)")
    else:
        # Performance mode (non-deterministic but faster)
        cudnn.benchmark = True
        cudnn.deterministic = False
        logging.info("⚠️  Deterministic mode disabled (results may vary across runs)")
    
    # Always set seeds for reproducibility (even in non-deterministic mode, seeds help)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # Set seed for all GPUs


def load_model_checkpoint(model, args):
    """Load trained model checkpoint."""
    checkpoint_path = os.path.join(args.output_dir, 'best_model_latest.pth')
    
    if not os.path.exists(checkpoint_path):
        alt_path = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"No checkpoint found in {args.output_dir}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model_state_dict = checkpoint['model_state']
    else:
        model_state_dict = checkpoint
    
    has_aux_heads = any('aux_heads' in key for key in model_state_dict.keys())
    has_skip_fusions = any('skip_fusions' in key for key in model_state_dict.keys())
    has_smart_skips = any('smart_skips' in key for key in model_state_dict.keys())
    has_bottleneck = any('bottleneck_layer' in key and 'upsample' not in key 
                        for key in model_state_dict.keys())
    has_feature_adapters = any('feature_adapters' in key for key in model_state_dict.keys())
    has_streaming_proj = any('streaming_proj' in key for key in model_state_dict.keys())
    has_multiscale_agg = any('multiscale_proj' in key or 'multiscale_fusion' in key 
                            for key in model_state_dict.keys())
    
    ds_mismatch = (has_aux_heads != model.model.use_deep_supervision)
    
    adapter_mismatch = False
    if model.model.adapter_mode == 'external':
        adapter_mismatch = not has_feature_adapters or has_streaming_proj
    elif model.model.adapter_mode == 'streaming':
        adapter_mismatch = not has_streaming_proj or has_feature_adapters
    
    fusion_mismatch = False
    if model.model.fusion_method == 'fourier':
        fusion_mismatch = not has_skip_fusions
    elif model.model.fusion_method == 'smart':
        fusion_mismatch = not has_smart_skips
    elif model.model.fusion_method == 'gcff':
        fusion_mismatch = not hasattr(model.model, 'gcff_skips') or model.model.gcff_skips is None
    else:
        fusion_mismatch = (has_skip_fusions or has_smart_skips)
    
    bottleneck_mismatch = (has_bottleneck != model.model.use_bottleneck)
    multiscale_mismatch = (has_multiscale_agg != model.model.use_multiscale_agg)
    
    if ds_mismatch or fusion_mismatch or bottleneck_mismatch or adapter_mismatch or multiscale_mismatch:
        print("⚠️  WARNING: Checkpoint and model have architecture differences!")
        print(f"   Deep Supervision mismatch: {ds_mismatch}")
        print(f"   Fusion mismatch: {fusion_mismatch}")
        print(f"   Bottleneck mismatch: {bottleneck_mismatch}")
        print(f"   Adapter mismatch: {adapter_mismatch}")
        print(f"   Multi-scale mismatch: {multiscale_mismatch}")
        print("   Loading with strict=False (some weights may not load correctly)")
        logging.warning("Checkpoint and model have architecture differences - loading with strict=False")
        logging.warning(f"  Deep Supervision: {ds_mismatch}, Fusion: {fusion_mismatch}, Bottleneck: {bottleneck_mismatch}, Adapter: {adapter_mismatch}, Multi-scale: {multiscale_mismatch}")
        msg = model.load_state_dict(model_state_dict, strict=False)
        
        if msg.unexpected_keys:
            print(f"⚠️  Ignored {len(msg.unexpected_keys)} unexpected keys in checkpoint")
            logging.warning(f"Ignored {len(msg.unexpected_keys)} unexpected keys")
        if msg.missing_keys:
            missing_fusion = [k for k in msg.missing_keys if 'skip_fusions' in k or 'smart_skips' in k]
            missing_other = [k for k in msg.missing_keys if 'skip_fusions' not in k and 'smart_skips' not in k]
            if missing_fusion:
                print(f"⚠️  Missing {len(missing_fusion)} fusion-related keys (expected if fusion method differs)")
                logging.warning(f"Missing {len(missing_fusion)} fusion-related keys")
            if missing_other:
                print(f"⚠️  CRITICAL: Missing {len(missing_other)} other keys - model may not work correctly!")
                logging.error(f"Missing {len(missing_other)} other keys - model may not work correctly!")
                if len(missing_other) > 10:
                    print(f"   First 10 missing keys: {missing_other[:10]}")
    else:
        print("✓ Checkpoint architecture matches model - loading with strict=True")
        logging.info("Checkpoint architecture matches model - loading with strict=True")
        model.load_state_dict(model_state_dict, strict=True)
    
    return os.path.basename(checkpoint_path)


def get_dataset_info(dataset_type, manuscript=None):
    """Get dataset-specific information."""
    if dataset_type.upper() == "UDIADS_BIB":
        from datasets.dataset_udiadsbib import rgb_to_class
        
        class_colors = [
            (0, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (255, 0, 0), (0, 255, 0)
        ]
        
        if manuscript in ['Syr341', 'Syr341FS']:
            class_names = ['Background', 'Paratext', 'Decoration', 'Main Text', 'Title']
            class_colors = class_colors[:5]
        else:
            class_names = ['Background', 'Paratext', 'Decoration', 'Main Text', 'Title', 'Chapter Headings']
        
        return class_colors, class_names, rgb_to_class
        
    elif dataset_type.upper() == "DIVAHISDB":
        try:
            from datasets.dataset_divahisdb import rgb_to_class
        except ImportError:
            logging.warning("DIVAHISDB dataset class not available")
            rgb_to_class = None
        
        class_colors = [(0, 0, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255)]
        class_names = ['Background', 'Comment', 'Decoration', 'Main Text']
        
        return class_colors, class_names, rgb_to_class
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_dataset_paths(args):
    """Get dataset-specific file paths."""
    manuscript_name = args.manuscript
    
    if args.dataset.upper() == "UDIADS_BIB":
        if args.use_patched_data:
            patch_dir = f'{args.udiadsbib_root}/{manuscript_name}/Image/test'
            mask_dir = f'{args.udiadsbib_root}/{manuscript_name}/mask/test_labels'
        else:
            patch_dir = f'{args.udiadsbib_root}/{manuscript_name}/img-{manuscript_name}/test'
            mask_dir = f'{args.udiadsbib_root}/{manuscript_name}/pixel-level-gt-{manuscript_name}/test'
        
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
        
        base_dir = args.divahisdb_root.replace('_patched', '')
        original_img_dir = f'{base_dir}/img/{manuscript_name}/test'
        original_mask_dir = f'{base_dir}/pixel-level-gt/{manuscript_name}/test'
    
    return patch_dir, mask_dir, original_img_dir, original_mask_dir


def process_patch_groups(patch_files):
    """Group patch files by their original image names."""
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
    """Estimate original image dimensions from patch information."""
    for ext in ['.jpg', '.png', '.tif', '.tiff']:
        orig_path = os.path.join(original_img_dir, f"{original_name}{ext}")
        if os.path.exists(orig_path):
            with Image.open(orig_path) as img:
                orig_width, orig_height = img.size
            
            patches_per_row = max(orig_width // patch_size, 1)
            max_x = ((orig_width // patch_size) + (1 if orig_width % patch_size else 0)) * patch_size
            max_y = ((orig_height // patch_size) + (1 if orig_height % patch_size else 0)) * patch_size
            
            return max_x, max_y, patches_per_row
    
    logging.warning(f"Could not find original image for {original_name}, estimating dimensions")
    patches_per_row = 10
    max_patch_id = max([patch_positions[p] for p in patches])
    max_x = ((max_patch_id % patches_per_row) + 1) * patch_size
    max_y = ((max_patch_id // patches_per_row) + 1) * patch_size
    
    return max_x, max_y, patches_per_row


def predict_patch_with_tta(patch_tensor, model, return_probs=False, use_amp=True):
    """Predict patch with test-time augmentation (single patch, kept for backward compatibility)."""
    import torchvision.transforms.functional as TF
    
    device = patch_tensor.device
    augmented_outputs = []
    
    transforms = [
        ('original', lambda x: x, lambda x: x),
        ('hflip', TF.hflip, TF.hflip),
        ('vflip', TF.vflip, TF.vflip),
        ('rot90', lambda x: TF.rotate(x, angle=-90), lambda x: TF.rotate(x, angle=90))
    ]
    
    for name, forward_transform, reverse_transform in transforms:
        if name == 'original':
            transformed = patch_tensor
        else:
            transformed = forward_transform(patch_tensor.squeeze(0)).unsqueeze(0)
        
        with torch.no_grad():
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = model(transformed.to(device))
            else:
                output = model(transformed.to(device))
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.softmax(output, dim=1)
            
            if name != 'original':
                probs = reverse_transform(probs.squeeze(0)).unsqueeze(0)
            
            augmented_outputs.append(probs)
    
    averaged_probs = torch.stack(augmented_outputs).mean(dim=0)
    
    if return_probs:
        return averaged_probs.squeeze(0).cpu().numpy()
    else:
        return torch.argmax(averaged_probs, dim=1).squeeze(0).cpu().numpy()


def predict_batch_with_tta(batch_tensor, model, return_probs=False, use_amp=True):
    """Predict batch with test-time augmentation (batched for efficiency).
    
    This function processes all augmentations in a single forward pass, providing
    2-3x speedup compared to processing each augmentation separately.
    
    Args:
        batch_tensor: Input batch tensor of shape (B, C, H, W)
        model: Model for inference
        return_probs: Whether to return probability maps or predictions
        use_amp: Whether to use mixed precision (FP16) for faster inference
    
    Returns:
        If return_probs=True: Averaged probability maps of shape (B, H, W, num_classes) as numpy array
        If return_probs=False: Predictions of shape (B, H, W) as numpy array
    """
    B, C, H, W = batch_tensor.shape
    device = batch_tensor.device
    
    # Create all augmentations at once (4 augmentations per patch)
    # Original, H-flip, V-flip, Rot90
    augmented_batch = []
    augmented_batch.append(batch_tensor)  # Original
    augmented_batch.append(torch.flip(batch_tensor, dims=[3]))  # H-flip (flip width dimension)
    augmented_batch.append(torch.flip(batch_tensor, dims=[2]))  # V-flip (flip height dimension)
    augmented_batch.append(torch.rot90(batch_tensor, k=1, dims=[2, 3]))  # Rot90 (rotate height/width dims)
    
    # Stack all augmentations: (4*B, C, H, W)
    all_augmented = torch.cat(augmented_batch, dim=0)
    
    # Single forward pass for all augmentations (much more efficient!)
    with torch.no_grad():
        if use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                output = model(all_augmented)
        else:
            output = model(all_augmented)
        if isinstance(output, tuple):
            output = output[0]
        probs = torch.softmax(output, dim=1)  # (4*B, num_classes, H, W)
    
    # Split back to (4, B, num_classes, H, W)
    num_classes = probs.shape[1]
    probs = probs.view(4, B, num_classes, H, W)
    
    # Reverse transforms to align all augmentations
    # Original: no reverse needed (index 0)
    probs[1] = torch.flip(probs[1], dims=[3])  # Reverse H-flip (flip width dimension)
    probs[2] = torch.flip(probs[2], dims=[2])  # Reverse V-flip (flip height dimension)
    probs[3] = torch.rot90(probs[3], k=3, dims=[2, 3])  # Reverse Rot90 (rotate 270 degrees = 3 * 90)
    
    # Average probabilities across all augmentations: (B, num_classes, H, W)
    avg_probs = probs.mean(dim=0)
    
    if return_probs:
        # Return as (B, H, W, num_classes) numpy array
        return avg_probs.permute(0, 2, 3, 1).cpu().numpy()
    else:
        # Return predictions as (B, H, W) numpy array
        return torch.argmax(avg_probs, dim=1).cpu().numpy()


def apply_crf_postprocessing(prob_map, rgb_image, num_classes=6, 
                              spatial_weight=5.0, spatial_x_stddev=5.0, spatial_y_stddev=3.0,
                              color_weight=5.0, color_stddev=80.0, num_iterations=5):
    """Apply DenseCRF post-processing to refine segmentation predictions.
    
    IMPORTANT: CRF parameters should be tuned for your specific application!
    The default parameters are optimized for historical documents (parchment/paper backgrounds),
    but may not be optimal for all datasets. Poorly tuned CRF can reduce IoU by 2-3%.
    
    Tuning Recommendations:
    1. Validate CRF helps: Test with/without CRF on validation set
    2. Grid search: Try different parameter combinations on validation set
    3. Consider domain: Historical docs have different color/texture properties than natural images
    
    Parameter Guidelines (based on DenseCRF paper and DeepLab):
    - spatial_weight: Controls spatial coherence strength (higher = stronger coherence)
      * Historical docs: 5.0 (text is linear, needs stronger coherence)
      * Natural images: 3.0 (default from ImageNet)
    - spatial_x_stddev: Horizontal spatial standard deviation
      * Historical docs: 5.0 (horizontal text flow)
      * Natural images: 3.0
    - spatial_y_stddev: Vertical spatial standard deviation
      * Historical docs: 3.0 (tighter vertical, line height)
      * Natural images: 3.0
    - color_weight: Controls color similarity strength (higher = stronger color influence)
      * Historical docs: 5.0 (weaker color, aged documents vary)
      * Natural images: 10.0 (stronger color cues)
    - color_stddev: Color standard deviation (higher = more tolerance for color variation)
      * Historical docs: 80.0 (higher tolerance for aged/variegated colors)
      * Natural images: 50.0
    - num_iterations: Number of CRF inference iterations
      * Historical docs: 5 (diminishing returns after 5 iterations)
      * Natural images: 10
    
    Args:
        prob_map: Probability map of shape (H, W, num_classes)
        rgb_image: RGB image of shape (H, W, 3) for color-based pairwise terms
        num_classes: Number of segmentation classes
        spatial_weight: Weight for spatial pairwise term (default: 5.0 for historical docs)
        spatial_x_stddev: Horizontal spatial standard deviation (default: 5.0)
        spatial_y_stddev: Vertical spatial standard deviation (default: 3.0)
        color_weight: Weight for color-based pairwise term (default: 5.0 for historical docs)
        color_stddev: Color standard deviation (default: 80.0 for historical docs)
        num_iterations: Number of CRF inference iterations (default: 5)
    
    Returns:
        Refined prediction map of shape (H, W) as uint8 array
    """
    try:
        try:
            import pydensecrf2.densecrf as dcrf
            from pydensecrf2.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
        except ImportError:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
    except ImportError:
        logging.error("pydensecrf2 is required for CRF post-processing")
        raise ImportError("pydensecrf2 is required for CRF post-processing")
    
    H, W = prob_map.shape[:2]
    
    if prob_map.shape[2] != num_classes:
        raise ValueError(f"Probability map has {prob_map.shape[2]} classes but expected {num_classes}")
    
    prob_map = np.ascontiguousarray(prob_map, dtype=np.float32)
    
    if rgb_image.shape[:2] != (H, W):
        rgb_image_resized = np.array(Image.fromarray(rgb_image).resize((W, H), Image.BILINEAR))
    else:
        rgb_image_resized = rgb_image.copy()
    
    if rgb_image_resized.dtype != np.uint8:
        rgb_image_resized = np.clip(rgb_image_resized, 0, 255).astype(np.uint8)
    rgb_image_resized = np.ascontiguousarray(rgb_image_resized, dtype=np.uint8)
    
    prob_map_transposed = np.ascontiguousarray(prob_map.transpose(2, 0, 1), dtype=np.float32)
    
    crf = dcrf.DenseCRF2D(W, H, num_classes)
    unary = unary_from_softmax(prob_map_transposed)
    crf.setUnaryEnergy(unary)
    
    pairwise_gaussian = create_pairwise_gaussian(sdims=(spatial_y_stddev, spatial_x_stddev), shape=(H, W))
    crf.addPairwiseEnergy(pairwise_gaussian, compat=spatial_weight)
    
    pairwise_bilateral = create_pairwise_bilateral(
        sdims=(spatial_y_stddev, spatial_x_stddev),
        schan=(color_stddev, color_stddev, color_stddev),
        img=rgb_image_resized,
        chdim=2
    )
    crf.addPairwiseEnergy(pairwise_bilateral, compat=color_weight)
    
    Q = crf.inference(num_iterations)
    refined_probs = np.array(Q).reshape((num_classes, H, W)).transpose(1, 2, 0)
    refined_pred = np.argmax(refined_probs, axis=2).astype(np.uint8)
    
    return refined_pred


def stitch_patches(patches, patch_positions, max_x, max_y, patches_per_row, patch_size, model, use_tta=False, return_probs=False, batch_size=32, use_amp=True):
    """Stitch together patch predictions into full image.
    
    IMPORTANT: Always accumulates probabilities (not class indices) for overlapping patches.
    Averaging class indices is mathematically incorrect - we must average probabilities, then take argmax.
    
    Args:
        patches: List of patch file paths
        patch_positions: Dictionary mapping patch paths to patch IDs
        max_x, max_y: Maximum image dimensions
        patches_per_row: Number of patches per row
        patch_size: Size of each patch
        model: Model for inference
        use_tta: Whether to use test-time augmentation (TTA requires single-patch processing)
        return_probs: Whether to return probability maps
        batch_size: Batch size for non-TTA inference (default: 32)
        use_amp: Whether to use mixed precision (FP16) for faster inference (default: True)
    """
    import torchvision.transforms.functional as TF
    
    # Always use probability accumulation for overlapping patches (mathematically correct)
    # We'll determine num_classes from first batch
    count_map = np.zeros((max_y, max_x), dtype=np.int32)
    prob_full = None
    num_classes = None
    
    # TTA with batched processing (all augmentations in single forward pass)
    # For non-TTA, use batched processing for efficiency (3-5x speedup)
    if use_tta:
        # Batched TTA processing (2-3x faster than single-patch TTA)
        # Split patches into batches
        patch_list = list(patches)
        num_batches = (len(patch_list) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(patch_list))
            batch_patches = patch_list[start_idx:end_idx]
            
            # Load batch of patches
            batch_tensors = []
            batch_positions = []
            
            for patch_path in batch_patches:
                # Convert to numpy immediately to avoid memory leak (PIL Image not explicitly closed)
                # This prevents memory accumulation in long testing loops (100+ images)
                patch_np = np.array(Image.open(patch_path).convert("RGB"))
                patch_tensor = TF.to_tensor(patch_np)  # Normalizes to [0, 1]
                # CRITICAL: Apply ImageNet normalization to match training preprocessing
                # The model uses EfficientNet-B4 encoder (pretrained on ImageNet)
                # Without this normalization, the model receives wrong input distribution
                patch_tensor = TF.normalize(
                    patch_tensor,
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean
                    std=[0.229, 0.224, 0.225]    # ImageNet std
                )
                batch_tensors.append(patch_tensor)
                batch_positions.append(patch_positions[patch_path])
            
            # Stack into batch tensor
            batch = torch.stack(batch_tensors).cuda()
            
            # Process batch with TTA (all augmentations in single forward pass)
            # Always get probabilities for correct overlap handling (average probabilities, not class indices)
            probs_batch = predict_batch_with_tta(batch, model, return_probs=True, use_amp=use_amp)
            if prob_full is None:
                num_classes = probs_batch.shape[3]
                prob_full = np.zeros((max_y, max_x, num_classes), dtype=np.float32)
            
            # Stitch batch results - always accumulate probabilities (not class indices)
            for i, patch_id in enumerate(batch_positions):
                x = (patch_id % patches_per_row) * patch_size
                y = (patch_id // patches_per_row) * patch_size
                
                probs = probs_batch[i]  # (H, W, num_classes)
                
                # Stitch patch - accumulate probabilities (mathematically correct for overlapping patches)
                if y + patch_size <= prob_full.shape[0] and x + patch_size <= prob_full.shape[1]:
                    prob_full[y:y+patch_size, x:x+patch_size, :] += probs
                    count_map[y:y+patch_size, x:x+patch_size] += 1
                else:
                    valid_h = min(patch_size, prob_full.shape[0] - y)
                    valid_w = min(patch_size, prob_full.shape[1] - x)
                    if valid_h > 0 and valid_w > 0:
                        prob_full[y:y+valid_h, x:x+valid_w, :] += probs[:valid_h, :valid_w, :]
                        count_map[y:y+valid_h, x:x+valid_w] += 1
    else:
        # Batched processing for non-TTA (much faster)
        # Split patches into batches
        patch_list = list(patches)
        num_batches = (len(patch_list) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(patch_list))
            batch_patches = patch_list[start_idx:end_idx]
            
            # Load batch of patches
            batch_tensors = []
            batch_positions = []
            
            for patch_path in batch_patches:
                # Convert to numpy immediately to avoid memory leak (PIL Image not explicitly closed)
                # This prevents memory accumulation in long testing loops (100+ images)
                patch_np = np.array(Image.open(patch_path).convert("RGB"))
                patch_tensor = TF.to_tensor(patch_np)  # Normalizes to [0, 1]
                # CRITICAL: Apply ImageNet normalization to match training preprocessing
                # The model uses EfficientNet-B4 encoder (pretrained on ImageNet)
                # Without this normalization, the model receives wrong input distribution
                patch_tensor = TF.normalize(
                    patch_tensor,
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean
                    std=[0.229, 0.224, 0.225]    # ImageNet std
                )
                batch_tensors.append(patch_tensor)
                batch_positions.append(patch_positions[patch_path])
            
            # Stack into batch tensor
            batch = torch.stack(batch_tensors).cuda()
            
            # Forward pass on batch
            # Always compute probabilities for correct overlap handling (average probabilities, not class indices)
            with torch.no_grad():
                if use_amp and batch.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        output = model(batch)
                else:
                    output = model(batch)
                if isinstance(output, tuple):
                    output = output[0]
                
                # Always compute probabilities (not just when return_probs=True)
                probs_batch = torch.softmax(output, dim=1).cpu().numpy()  # (B, num_classes, H, W)
                if prob_full is None:
                    num_classes = probs_batch.shape[1]
                    prob_full = np.zeros((max_y, max_x, num_classes), dtype=np.float32)
            
            # Stitch batch results - always accumulate probabilities (not class indices)
            for i, patch_id in enumerate(batch_positions):
                x = (patch_id % patches_per_row) * patch_size
                y = (patch_id // patches_per_row) * patch_size
                
                probs = probs_batch[i].transpose(1, 2, 0)  # (H, W, num_classes)
                
                # Stitch patch - accumulate probabilities (mathematically correct for overlapping patches)
                if y + patch_size <= prob_full.shape[0] and x + patch_size <= prob_full.shape[1]:
                    prob_full[y:y+patch_size, x:x+patch_size, :] += probs
                    count_map[y:y+patch_size, x:x+patch_size] += 1
                else:
                    valid_h = min(patch_size, prob_full.shape[0] - y)
                    valid_w = min(patch_size, prob_full.shape[1] - x)
                    if valid_h > 0 and valid_w > 0:
                        prob_full[y:y+valid_h, x:x+valid_w, :] += probs[:valid_h, :valid_w, :]
                        count_map[y:y+valid_h, x:x+valid_w] += 1
    
    # Average probabilities across overlapping patches (mathematically correct)
    prob_full = prob_full / np.maximum(count_map[:, :, np.newaxis], 1)
    
    # Take argmax of averaged probabilities to get final predictions
    pred_full = np.argmax(prob_full, axis=2).astype(np.uint8)
    
    if return_probs:
        return pred_full, prob_full
    else:
        return pred_full


def save_prediction_results(pred_full, original_name, class_colors, result_dir):
    """Save prediction results as RGB image."""
    if result_dir is not None:
        os.makedirs(result_dir, exist_ok=True)
        rgb_mask = np.zeros((pred_full.shape[0], pred_full.shape[1], 3), dtype=np.uint8)
        for idx, color in enumerate(class_colors):
            rgb_mask[pred_full == idx] = color
        Image.fromarray(rgb_mask).save(os.path.join(result_dir, f"{original_name}.png"))


def save_comparison_visualization(pred_full, gt_class, original_name, original_img_dir, 
                                test_save_path, class_colors, class_names):
    """Save side-by-side comparison visualization."""
    compare_dir = os.path.join(test_save_path, 'compare')
    os.makedirs(compare_dir, exist_ok=True)
    
    cmap = ListedColormap(class_colors)
    n_classes = len(class_colors)
    
    if gt_class.shape != pred_full.shape:
        logging.warning(f"Resizing ground truth for {original_name}")
        gt_class_resized = np.zeros_like(pred_full)
        min_h = min(gt_class.shape[0], pred_full.shape[0])
        min_w = min(gt_class.shape[1], pred_full.shape[1])
        gt_class_resized[:min_h, :min_w] = gt_class[:min_h, :min_w]
        gt_class = gt_class_resized
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    orig_img_path = None
    for ext in ['.jpg', '.png', '.tif', '.tiff']:
        test_path = os.path.join(original_img_dir, f"{original_name}{ext}")
        if os.path.exists(test_path):
            orig_img_path = test_path
            break
    
    if orig_img_path:
        # Convert to numpy immediately to avoid memory leak (PIL Image not explicitly closed)
        with Image.open(orig_img_path) as orig_img:
            orig_img = orig_img.convert("RGB")
            if orig_img.size != (pred_full.shape[1], pred_full.shape[0]):
                orig_img = orig_img.resize((pred_full.shape[1], pred_full.shape[0]), Image.BILINEAR)
            orig_img_np = np.array(orig_img)
        axs[0].imshow(orig_img_np)
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
    plt.savefig(os.path.join(compare_dir, f"{original_name}_compare.png"), 
                bbox_inches='tight', dpi=150)
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
    
    # Print to stdout (visible in output) and log file
    print("\n" + "=" * 80)
    print("SEGMENTATION METRICS")
    print("=" * 80)
    print(f"Images Processed: {num_processed_images}")
    print("\nPer-class metrics:")
    print("-" * 80)
    for cls in range(n_classes):
        print(f"{class_names[cls]:<20}: Precision={precision[cls]:.4f}, "
              f"Recall={recall[cls]:.4f}, F1={f1[cls]:.4f}, IoU={iou_per_class[cls]:.4f}")
    
    print("\nMean metrics:")
    print("-" * 80)
    print(f"Mean Precision: {np.mean(precision):.4f}")
    print(f"Mean Recall:    {np.mean(recall):.4f}")
    print(f"Mean F1-Score:  {np.mean(f1):.4f}")
    print(f"Mean IoU:       {np.mean(iou_per_class):.4f}")
    print("=" * 80)
    sys.stdout.flush()
    
    # Also log for file
    logging.info("\nPer-class metrics:")
    logging.info("-" * 80)
    for cls in range(n_classes):
        logging.info(f"{class_names[cls]:<15}: Precision={precision[cls]:.4f}, "
                    f"Recall={recall[cls]:.4f}, F1={f1[cls]:.4f}, IoU={iou_per_class[cls]:.4f}")
    
    logging.info("\nMean metrics:")
    logging.info("-" * 40)
    logging.info(f"Mean Precision: {np.mean(precision):.4f}")
    logging.info(f"Mean Recall: {np.mean(recall):.4f}")
    logging.info(f"Mean F1: {np.mean(f1):.4f}")
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


def inference(args, model, test_save_path=None):
    """Run inference on historical document dataset."""
    # CRITICAL: Set model to evaluation mode before inference
    # This ensures BatchNorm/GroupNorm use running statistics and Dropout is disabled
    # Without this, inference will use training statistics, causing stochastic predictions and lower accuracy
    model.eval()
    
    # Enable mixed precision (FP16) for faster inference on modern GPUs (2-3x speedup)
    # Modern GPUs (V100, A100, RTX series) have fast FP16 tensor cores
    # FP32 inference is 2-3x slower than FP16 on these GPUs
    use_amp = torch.cuda.is_available() and getattr(args, 'use_amp', True)
    if use_amp:
        logging.info("🚀 Using mixed precision (FP16) for faster inference")
    else:
        logging.info("Using FP32 precision (AMP disabled or CPU mode)")
    
    logging.info(f"Starting inference on {args.dataset} dataset")
    
    class_colors, class_names, rgb_to_class_func = get_dataset_info(args.dataset, args.manuscript)
    n_classes = len(class_colors)
    patch_size = 224
    
    patch_dir, mask_dir, original_img_dir, original_mask_dir = get_dataset_paths(args)
    
    if not os.path.exists(patch_dir):
        logging.error(f"Patch directory not found: {patch_dir}")
        return "Testing Failed!"
    
    patch_files = sorted(glob.glob(os.path.join(patch_dir, '*.png')))
    if len(patch_files) == 0:
        logging.info(f"No patch files found in {patch_dir}")
        return "Testing Finished!"
    
    logging.info(f"Found {len(patch_files)} patches for {args.manuscript}")
    
    TP = np.zeros(n_classes, dtype=np.float64)
    FP = np.zeros(n_classes, dtype=np.float64)  
    FN = np.zeros(n_classes, dtype=np.float64)
    
    result_dir = os.path.join(test_save_path, "result") if test_save_path else None
    patch_groups, patch_positions = process_patch_groups(patch_files)
    num_processed_images = 0
    
    print(f"\nFound {len(patch_groups)} original images to process")
    logging.info(f"Found {len(patch_groups)} original images to process")
    
    for original_name, patches in patch_groups.items():
        print(f"Processing: {original_name} ({len(patches)} patches)")
        logging.info(f"Processing: {original_name} ({len(patches)} patches)")
        
        max_x, max_y, patches_per_row = estimate_image_dimensions(
            original_name, original_img_dir, patches, patch_positions, patch_size
        )
        
        use_tta = getattr(args, 'use_tta', False)
        use_crf = getattr(args, 'use_crf', False)
        
        if use_tta:
            logging.info(f"🔀 Using Test-Time Augmentation for {original_name}")
        
        if use_crf:
            pred_full, prob_full = stitch_patches(
                patches, patch_positions, max_x, max_y, 
                patches_per_row, patch_size, model, use_tta=use_tta, return_probs=True, 
                batch_size=getattr(args, 'batch_size', 32), use_amp=use_amp
            )
            
            orig_img_rgb = None
            for ext in ['.jpg', '.png', '.tif', '.tiff']:
                orig_path = os.path.join(original_img_dir, f"{original_name}{ext}")
                if os.path.exists(orig_path):
                    # Convert to numpy immediately to avoid memory leak (PIL Image not explicitly closed)
                    # This prevents memory accumulation in long testing loops (100+ images)
                    with Image.open(orig_path) as orig_img_pil:
                        orig_img_pil = orig_img_pil.convert("RGB")
                        if orig_img_pil.size != (max_x, max_y):
                            orig_img_pil = orig_img_pil.resize((max_x, max_y), Image.BILINEAR)
                        orig_img_rgb = np.array(orig_img_pil)
                    break
            
            if orig_img_rgb is not None:
                try:
                    logging.info(f"🎯 Applying CRF post-processing for {original_name}")
                    # Use default parameters optimized for historical documents
                    # Parameters can be tuned via args if needed (see apply_crf_postprocessing docstring)
                    pred_full = apply_crf_postprocessing(
                        prob_full, orig_img_rgb, num_classes=n_classes
                        # Default parameters (optimized for historical documents):
                        # spatial_weight=5.0, spatial_x_stddev=5.0, spatial_y_stddev=3.0,
                        # color_weight=5.0, color_stddev=80.0, num_iterations=5
                    )
                    logging.info(f"✓ CRF post-processing completed for {original_name}")
                except Exception as e:
                    logging.warning(f"CRF post-processing failed for {original_name}: {e}")
                    pred_full = np.argmax(prob_full, axis=2).astype(np.uint8)
            else:
                logging.warning(f"Original image not found for CRF: {original_name}")
                pred_full = np.argmax(prob_full, axis=2).astype(np.uint8)
        else:
            pred_full = stitch_patches(
                patches, patch_positions, max_x, max_y, 
                patches_per_row, patch_size, model, use_tta=use_tta, return_probs=False,
                batch_size=getattr(args, 'batch_size', 32), use_amp=use_amp
            )
        
        save_prediction_results(pred_full, original_name, class_colors, result_dir)
        
        gt_found = False
        for ext in ['.png', '.jpg', '.tif', '.tiff']:
            gt_path = os.path.join(original_mask_dir, f"{original_name}{ext}")
            if os.path.exists(gt_path):
                # Convert to numpy immediately to avoid memory leak (PIL Image not explicitly closed)
                # This prevents memory accumulation in long testing loops (100+ images)
                with Image.open(gt_path) as gt_pil:
                    gt_pil = gt_pil.convert("RGB")
                    gt_np = np.array(gt_pil)
                if rgb_to_class_func:
                    gt_class = rgb_to_class_func(gt_np)
                    gt_found = True
                    break
        
        if not gt_found:
            print(f"⚠️  Warning: No ground truth found for {original_name}")
            logging.warning(f"No ground truth found for {original_name}")
            gt_class = np.zeros_like(pred_full)
        else:
            print(f"✓ Ground truth found for {original_name}")
        
        if test_save_path and gt_found:
            save_comparison_visualization(
                pred_full, gt_class, original_name, original_img_dir,
                test_save_path, class_colors, class_names
            )
        
        if gt_found:
            if gt_class.shape != pred_full.shape:
                logging.warning(f"Resizing ground truth: {gt_class.shape} -> {pred_full.shape}")
                gt_class_resized = np.zeros_like(pred_full)
                min_h = min(gt_class.shape[0], pred_full.shape[0])
                min_w = min(gt_class.shape[1], pred_full.shape[1])
                gt_class_resized[:min_h, :min_w] = gt_class[:min_h, :min_w]
                gt_class = gt_class_resized
            
            compute_segmentation_metrics(pred_full, gt_class, n_classes, TP, FP, FN)
            num_processed_images += 1
        
        print(f"✓ Completed: {original_name}")
        logging.info(f"Completed: {original_name}")
    
    print(f"\n{'='*80}")
    print(f"Testing Summary: Processed {num_processed_images} images with ground truth")
    print(f"{'='*80}\n")
    logging.info(f"Inference completed on {num_processed_images} images")
    
    print_final_metrics(TP, FP, FN, class_names, num_processed_images)
    
    save_metrics_to_file(args, TP, FP, FN, class_names, num_processed_images)
    
    return "Testing Finished!"


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
    
    for manuscript in expected_manuscripts:
        metrics_file = os.path.join(parent_dir, f"metrics_{manuscript}.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    all_metrics.append(data)
                    found_manuscripts.append(manuscript)
                    metrics_files.append(metrics_file)
            except Exception as e:
                logging.warning(f"Failed to load metrics for {manuscript}: {e}")
    
    if len(found_manuscripts) == 4:
        avg_precision = sum(m['mean_precision'] for m in all_metrics) / len(all_metrics)
        avg_recall = sum(m['mean_recall'] for m in all_metrics) / len(all_metrics)
        avg_f1 = sum(m['mean_f1'] for m in all_metrics) / len(all_metrics)
        avg_iou = sum(m['mean_iou'] for m in all_metrics) / len(all_metrics)
        
        print("\n" + "="*80)
        print("AVERAGE METRICS ACROSS ALL MANUSCRIPTS")
        print("="*80)
        print(f"Manuscripts: {', '.join(found_manuscripts)}")
        print("-"*80)
        print(f"Mean Precision: {avg_precision:.4f}")
        print(f"Mean Recall:    {avg_recall:.4f}")
        print(f"Mean F1-Score:  {avg_f1:.4f}")
        print(f"Mean IoU:       {avg_iou:.4f}")
        print("="*80)
        sys.stdout.flush()
        
        for metrics_file in metrics_files:
            try:
                os.remove(metrics_file)
            except Exception as e:
                logging.warning(f"Failed to delete temporary metrics file {metrics_file}: {e}")
        
        return True
    else:
        return False


def main():
    """Main testing function."""
    print("=== Historical Document Segmentation Testing ===")
    print()
    
    args = parse_arguments()
    
    try:
        validate_arguments(args)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    setup_reproducible_testing(args)
    model = get_model(args, None)
    
    try:
        checkpoint_name = load_model_checkpoint(model, args)
        print(f"Loaded checkpoint: {checkpoint_name}")
    except Exception as e:
        print(f"ERROR: Failed to load model checkpoint: {e}")
        sys.exit(1)
    
    log_folder = './test_log/test_log_'
    setup_logging(log_folder, checkpoint_name)
    
    logging.info(str(args))
    logging.info(f"Testing with checkpoint: {checkpoint_name}")
    
    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, "predictions")
        os.makedirs(test_save_path, exist_ok=True)
        logging.info(f"Saving predictions to: {test_save_path}")
    else:
        test_save_path = None
    
    print("\n=== Starting Testing ===")
    print(f"Dataset: {args.dataset} | Manuscript: {args.manuscript}")
    print("-" * 80)
    print("TEST-TIME AUGMENTATION (TTA):", "✓ ENABLED" if args.use_tta else "✗ DISABLED")
    if args.use_tta:
        print("  → Using 4 augmentations: original, horizontal flip, vertical flip, rotation 90°")
        print("  → Averaging predictions across all augmentations")
    print("CRF POST-PROCESSING:", "✓ ENABLED" if args.use_crf else "✗ DISABLED")
    if args.use_crf:
        print("  → Using DenseCRF with spatial and color pairwise potentials")
        print("  → Parameters: spatial_weight=3.0, color_weight=10.0, iterations=10")
    print("-" * 80)
    print()
    
    try:
        result = inference(args, model, test_save_path)
        calculate_and_display_average_metrics(args)
        print("\n=== TESTING COMPLETED SUCCESSFULLY ===")
        print(result)
    except Exception as e:
        print(f"ERROR: Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
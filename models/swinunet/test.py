import argparse
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

from configs.config import get_config  # pyright: ignore[reportMissingImports]

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments for testing script."""
    parser = argparse.ArgumentParser(
        description='Test SwinUnet model on historical document datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on U-DIADS-Bib dataset with SwinUnet using swintiny config
  python test.py --yaml swintiny --output_dir ./models/ --manuscript Latin2 --is_savenii
  
  # Test using simmim config
  python test.py --yaml simmim --output_dir ./models/ --manuscript Latin2
        """
    )
    
    # Core arguments
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE",
                       help='Path to model configuration file (optional when using --yaml)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing trained model checkpoints')
    
    # New shorthand flag to choose a preset config
    parser.add_argument('--yaml', type=str, default='swintiny',
                        choices=['swintiny', 'simmim'],
                        help="Choose which preset YAML to use from common/configs: 'swintiny' or 'simmim'. If provided, --cfg is optional and will be overridden.")
    
    # Model selection
    parser.add_argument('--model', type=str, default='swinunet',
                       help='Model architecture to test')
    
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
    parser.add_argument('--use_tta', action='store_true', default=False,
                       help='Enable test-time augmentation (TTA) for improved accuracy (slower inference)')
    parser.add_argument('--multiscale', action='store_true', default=False,
                       help='Enable multi-scale testing (0.75x, 1.0x, 1.25x) for improved accuracy on varying text sizes (slower inference)')
    
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
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate and normalize command line arguments."""
    # Check for suspicious command line tokens
    bad_tokens = [t for t in sys.argv[1:] if t.lstrip('-').startswith('mg_')]
    if bad_tokens:
        print(f"Warning: suspicious argv tokens detected: {bad_tokens}")
        print("Did you accidentally paste a continuation fragment?")
    
    # Validate required paths
    if not args.cfg:
        raise ValueError("--cfg argument is required for SwinUnet model")
    if not os.path.exists(args.cfg):
        raise ValueError(f"Config file not found: {args.cfg}")
    
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
            pass  # num_classes is set, no need to print
        
        if not os.path.exists(args.udiadsbib_root):
            raise ValueError(f"U-DIADS-Bib dataset path not found: {args.udiadsbib_root}")
    elif args.dataset.upper() == "DIVAHISDB":
        args.num_classes = 4
        if not os.path.exists(args.divahisdb_root):
            raise ValueError(f"DIVAHISDB dataset path not found: {args.divahisdb_root}")


def get_model(args, config):
    """Create and load the SwinUnet model."""
    from vision_transformer import SwinUnet as ViT_seg
    model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    
    # Load pretrained weights if available
    try:
        model.load_from(config)
        pass  # Pretrained weights loaded, no need to print
    except FileNotFoundError as e:
        print(f"Warning: Pretrained checkpoint not found: {e}")
        print("Continuing without pretrained weights (random initialization)")
    except Exception as e:
        print(f"Warning: Failed to load pretrained weights: {e}")
        print("Continuing without pretrained weights (random initialization)")
    
    return model


def setup_logging(log_folder, snapshot_name):
    """Add file handler to existing logger (basicConfig already called in main())."""
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, f"{snapshot_name}.txt")
    
    # Add file handler to existing logger (basicConfig already called in main())
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logging.getLogger().addHandler(file_handler)


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
    """Load trained model checkpoint with support for full checkpoint format."""
    checkpoint_path = os.path.join(args.output_dir, 'best_model_latest.pth')
    
    if not os.path.exists(checkpoint_path):
        # Try alternative checkpoint names
        alt_path = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"No checkpoint found in {args.output_dir}")
    
    # Load checkpoint with appropriate strictness
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model_state from checkpoint (checkpoint may be a dict with 'model_state' key or direct state_dict)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model_state_dict = checkpoint['model_state']
    else:
        # If checkpoint is already a state_dict, use it directly (backward compatibility)
        model_state_dict = checkpoint
    
    # Validate num_classes before loading
    checkpoint_num_classes = None
    output_layer_key = 'swin_unet.output.weight'
    if output_layer_key in model_state_dict:
        # Output layer shape: (num_classes, embed_dim, 1, 1)
        checkpoint_num_classes = model_state_dict[output_layer_key].shape[0]
    elif 'model_state' in checkpoint and isinstance(checkpoint['model_state'], dict):
        # Try alternative key format
        if output_layer_key in checkpoint['model_state']:
            checkpoint_num_classes = checkpoint['model_state'][output_layer_key].shape[0]
    
    if checkpoint_num_classes is not None:
        if checkpoint_num_classes != args.num_classes:
            logging.warning("="*80)
            logging.warning(f"WARNING: Number of classes mismatch detected!")
            logging.warning(f"  Checkpoint was trained with: {checkpoint_num_classes} classes")
            logging.warning(f"  Test arguments specify: {args.num_classes} classes")
            logging.warning(f"  This will cause incorrect predictions!")
            logging.warning(f"  Please use --num_classes {checkpoint_num_classes} or retrain with {args.num_classes} classes")
            logging.warning("="*80)
        else:
            pass  # Verification passed, no need to print
    else:
        logging.warning(f"Could not determine num_classes from checkpoint (output layer key '{output_layer_key}' not found)")
        logging.warning("  Proceeding without validation - please verify num_classes manually")
    
    # Load model state with strict=False to handle minor differences
    msg = model.load_state_dict(model_state_dict, strict=False)
    
    # Print loading results
    if msg.missing_keys:
        logging.warning(f"  Missing keys (will use initialized values): {len(msg.missing_keys)}")
        if len(msg.missing_keys) <= 10:
            for key in msg.missing_keys:
                logging.warning(f"     - {key}")
        else:
            for key in msg.missing_keys[:5]:
                logging.warning(f"     - {key}")
            logging.warning(f"     ... and {len(msg.missing_keys) - 5} more")
    
    if msg.unexpected_keys:
        logging.warning(f"  Unexpected keys (ignored): {len(msg.unexpected_keys)}")
        if len(msg.unexpected_keys) <= 10:
            for key in msg.unexpected_keys:
                logging.warning(f"     - {key}")
        else:
            for key in msg.unexpected_keys[:5]:
                logging.warning(f"     - {key}")
            logging.warning(f"     ... and {len(msg.unexpected_keys) - 5} more")
    
    if not msg.missing_keys and not msg.unexpected_keys:
        pass  # All keys matched, no need to print 
    
    return os.path.basename(checkpoint_path)


def get_dataset_info(dataset_type, manuscript=None):
    """Get dataset-specific information (class colors, names, and RGB-to-class mapping)."""
    if dataset_type.upper() == "UDIADS_BIB":
        from datasets.dataset_udiadsbib_2 import rgb_to_class  # pyright: ignore[reportMissingImports]
        
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
            from datasets.dataset_divahisdb import rgb_to_class  # pyright: ignore[reportMissingImports]
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
    """Get dataset-specific file paths for patches, masks, and original images."""
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
    """Group patch files by their original image names and extract patch positions."""
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


def verify_rotation_transforms():
    """
    Verify that forward+reverse rotation transforms produce the original image.
    PyTorch's TF.rotate uses counter-clockwise rotation for positive angles.
    So: forward=-90 (clockwise) + reverse=+90 (counter-clockwise) = identity.
    """
    import torchvision.transforms.functional as TF
    import torch
    
    # Create a test tensor with distinct pattern
    test_tensor = torch.zeros(1, 3, 10, 10)
    test_tensor[0, 0, 2:8, 2:8] = 1.0  # Red square in center
    test_tensor[0, 1, 0:3, 0:3] = 1.0  # Green square in top-left
    
    # Test rotation pairs
    test_cases = [
        ('rot90', lambda x: TF.rotate(x, angle=-90), lambda x: TF.rotate(x, angle=90)),
        ('rot180', lambda x: TF.rotate(x, angle=-180), lambda x: TF.rotate(x, angle=180)),
        ('rot270', lambda x: TF.rotate(x, angle=-270), lambda x: TF.rotate(x, angle=270)),
    ]
    
    for name, forward, reverse in test_cases:
        transformed = forward(test_tensor.squeeze(0))
        restored = reverse(transformed).unsqueeze(0)
        diff = torch.abs(test_tensor - restored).max().item()
        # Allow small differences due to interpolation artifacts (especially at edges)
        # For 90/180/270 degree rotations, exact pixel-perfect restoration may not be possible
        # due to interpolation, but the overall structure should be preserved
        if diff > 0.01:  # More lenient threshold for interpolation artifacts
            print(f"WARNING: {name} transform verification failed! Max diff: {diff:.6f}")
            print(f"  This may indicate incorrect rotation direction. Check TF.rotate documentation.")
            return False
    
    return True


def predict_patch_with_tta(patch_tensor, model, use_amp=True):
    """
    Predict patch with 8-augmentation TTA ensemble for improved rare class stability.
    
    Rotation convention: PyTorch's TF.rotate uses counter-clockwise rotation for positive angles.
    - Forward: angle=-90 (clockwise 90°) -> Reverse: angle=+90 (counter-clockwise 90°) = identity
    - Forward: angle=-180 (clockwise 180°) -> Reverse: angle=+180 (counter-clockwise 180°) = identity
    - Forward: angle=-270 (clockwise 270°) -> Reverse: angle=+270 (counter-clockwise 270°) = identity
    
    Note: The transforms are verified on first call to ensure forward+reverse = identity.
    """
    import torchvision.transforms.functional as TF
    
    # Verify rotation transforms on first call (lazy verification)
    if not hasattr(predict_patch_with_tta, '_verified'):
        if not verify_rotation_transforms():
            import warnings
            warnings.warn("Rotation transform verification failed! TTA results may be incorrect.")
        predict_patch_with_tta._verified = True
    
    device = patch_tensor.device
    augmented_outputs = []
    
    # 8 augmentations: original, hflip, vflip, hflip+vflip, rot90, rot180, rot270, rot90+hflip
    # Rotation convention: TF.rotate uses counter-clockwise for positive angles
    # Forward=-90 (clockwise) + Reverse=+90 (counter-clockwise) = identity
    transforms = [
        ('original', lambda x: x, lambda x: x),
        ('hflip', TF.hflip, TF.hflip),
        ('vflip', TF.vflip, TF.vflip),
        ('hflip_vflip', lambda x: TF.vflip(TF.hflip(x)), lambda x: TF.hflip(TF.vflip(x))),
        ('rot90', lambda x: TF.rotate(x, angle=-90), lambda x: TF.rotate(x, angle=90)),
        ('rot180', lambda x: TF.rotate(x, angle=-180), lambda x: TF.rotate(x, angle=180)),
        ('rot270', lambda x: TF.rotate(x, angle=-270), lambda x: TF.rotate(x, angle=270)),
        ('rot90_hflip', lambda x: TF.hflip(TF.rotate(x, angle=-90)), lambda x: TF.rotate(TF.hflip(x), angle=90))
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
            
            probs = torch.softmax(output, dim=1)
            
            if name != 'original':
                probs = reverse_transform(probs.squeeze(0)).unsqueeze(0)
            
            augmented_outputs.append(probs)
    
    averaged_probs = torch.stack(augmented_outputs).mean(dim=0)
    return averaged_probs.squeeze(0).cpu().numpy()


def predict_patch_multiscale(patch_tensor, model, scales=[0.75, 1.0, 1.25], use_amp=True):
    """
    Predict patch with multi-scale testing by averaging predictions across different scales.
    
    Args:
        patch_tensor: Input patch tensor (1, C, H, W)
        model: Model for inference
        scales: List of scales to test (default: [0.75, 1.0, 1.25])
        use_amp: Whether to use mixed precision
    
    Returns:
        numpy.ndarray: Averaged probability map (H, W, C)
    """
    import torchvision.transforms.functional as TF
    
    device = patch_tensor.device
    scale_outputs = []
    original_size = patch_tensor.shape[-2:]  # (H, W)
    
    for scale in scales:
        if scale == 1.0:
            # Use original scale
            scaled_tensor = patch_tensor
        else:
            # Resize to scaled size
            new_h = int(original_size[0] * scale)
            new_w = int(original_size[1] * scale)
            scaled_tensor = TF.resize(patch_tensor.squeeze(0), size=(new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = model(scaled_tensor.to(device))
            else:
                output = model(scaled_tensor.to(device))
            
            probs = torch.softmax(output, dim=1)
            
            # Resize probabilities back to original size if scaled
            if scale != 1.0:
                probs = TF.resize(probs.squeeze(0), size=original_size, interpolation=TF.InterpolationMode.BILINEAR).unsqueeze(0)
            
            scale_outputs.append(probs.squeeze(0).cpu().numpy())
    
    # Average probabilities across scales
    # scale_outputs are (C, H, W) arrays, stack them and average
    averaged_probs = np.stack(scale_outputs).mean(axis=0)  # Result: (C, H, W)
    return averaged_probs.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)


def predict_patch_with_tta_and_multiscale(patch_tensor, model, scales=[0.75, 1.0, 1.25], use_amp=True):
    """Predict patch with both TTA and multi-scale: apply TTA at each scale, then average across scales."""
    import torchvision.transforms.functional as TF
    
    device = patch_tensor.device
    scale_outputs = []
    original_size = patch_tensor.shape[-2:]  # (H, W) - should be (224, 224)
    model_input_size = (224, 224)  # Model always requires 224x224 input
    
    for scale in scales:
        if scale == 1.0:
            # Use original patch directly (already 224x224)
            model_input_tensor = patch_tensor
            scaled_size = original_size
        else:
            # Scale the patch (e.g., 0.75x → 168x168, 1.25x → 280x280)
            scaled_h = int(original_size[0] * scale)
            scaled_w = int(original_size[1] * scale)
            scaled_size = (scaled_h, scaled_w)
            scaled_tensor = TF.resize(patch_tensor.squeeze(0), size=scaled_size, interpolation=TF.InterpolationMode.BILINEAR).unsqueeze(0)
            # Resize scaled tensor to model input size (224x224) - model requires fixed input size
            model_input_tensor = TF.resize(scaled_tensor.squeeze(0), size=model_input_size, interpolation=TF.InterpolationMode.BILINEAR).unsqueeze(0)
        
        # Apply TTA to the model input tensor (always 224x224)
        probs_tta = predict_patch_with_tta(model_input_tensor, model, use_amp=use_amp)  # Returns (C, H, W) numpy array, shape (C, 224, 224)
        
        # Resize probabilities to match the scaled size, then back to original size
        if scale != 1.0:
            # First resize to scaled size (e.g., 168x168 for 0.75x)
            probs_tta_tensor = torch.from_numpy(probs_tta).unsqueeze(0).to(device)  # (1, C, 224, 224)
            probs_scaled = TF.resize(probs_tta_tensor.squeeze(0), size=scaled_size, interpolation=TF.InterpolationMode.BILINEAR).unsqueeze(0)
            # Then resize back to original size (224x224)
            probs_tta_tensor = TF.resize(probs_scaled.squeeze(0), size=original_size, interpolation=TF.InterpolationMode.BILINEAR).unsqueeze(0)
            probs_tta = probs_tta_tensor.squeeze(0).cpu().numpy()  # (C, 224, 224)
        
        scale_outputs.append(probs_tta)
    
    # Average probabilities across scales
    averaged_probs = np.stack(scale_outputs).mean(axis=0)  # Result: (C, H, W)
    return averaged_probs.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)


def stitch_patches(patches, patch_positions, max_x, max_y, patches_per_row, patch_size, model, num_classes, use_tta=False, use_multiscale=False, batch_size=24):
    """Stitch together patch predictions by accumulating probabilities (not class indices) for correct overlap handling with batch processing."""
    import torchvision.transforms.functional as TF
    
    prob_full = np.zeros((max_y, max_x, num_classes), dtype=np.float32)
    count_map = np.zeros((max_y, max_x), dtype=np.int32)
    
    use_amp = torch.cuda.is_available()
    
    # Prepare patch data (paths and positions) for batch processing
    patch_data = []
    for patch_path in patches:
        patch_id = patch_positions[patch_path]
        x = (patch_id % patches_per_row) * patch_size
        y = (patch_id // patches_per_row) * patch_size
        patch_data.append((patch_path, x, y))
    
    # Process patches in batches
    num_patches = len(patch_data)
    for batch_start in range(0, num_patches, batch_size):
        batch_end = min(batch_start + batch_size, num_patches)
        batch_patches = patch_data[batch_start:batch_end]
        
        # Load and preprocess batch
        batch_tensors = []
        batch_positions = []
        
        for patch_path, x, y in batch_patches:
            patch = Image.open(patch_path).convert("RGB")
            patch_tensor = TF.to_tensor(patch)
            # Apply ImageNet normalization for SwinUnet model (same as training dataset)
            # Note: test.py loads patches directly (not through dataset), so normalization must be applied here
            # to match the normalized data the model was trained on
            patch_tensor = TF.normalize(
                patch_tensor,
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]     # ImageNet std
            )
            batch_tensors.append(patch_tensor)
            batch_positions.append((x, y))
        
        # Stack into batch tensor
        batch_tensor = torch.stack(batch_tensors).cuda()
        
        # Run batch inference
        if use_tta or use_multiscale:
            # TTA and/or multi-scale: process each patch individually (requires individual processing)
            batch_probs = []
            for i in range(batch_tensor.shape[0]):
                patch_tensor_single = batch_tensor[i:i+1]
                if use_tta and use_multiscale:
                    # Combine TTA + multi-scale: apply TTA at each scale, then average across scales
                    probs_patch = predict_patch_with_tta_and_multiscale(patch_tensor_single, model, use_amp=use_amp)
                elif use_tta:
                    probs_patch = predict_patch_with_tta(patch_tensor_single, model, use_amp=use_amp)
                else:  # use_multiscale only
                    probs_patch = predict_patch_multiscale(patch_tensor_single, model, use_amp=use_amp)
                batch_probs.append(probs_patch)
        else:
            # Standard batch inference
            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(batch_tensor)
                else:
                    output = model(batch_tensor)
                batch_probs_np = torch.softmax(output, dim=1).cpu().numpy()
                # Convert from (B, C, H, W) to list of (H, W, C) arrays
                batch_probs = [batch_probs_np[i].transpose(1, 2, 0) for i in range(batch_probs_np.shape[0])]
        
        # Accumulate probabilities for each patch in batch
        for i, (x, y) in enumerate(batch_positions):
            probs_patch = batch_probs[i]
            
            if y + patch_size <= prob_full.shape[0] and x + patch_size <= prob_full.shape[1]:
                prob_full[y:y+patch_size, x:x+patch_size, :] += probs_patch
                count_map[y:y+patch_size, x:x+patch_size] += 1
            else:
                valid_h = min(patch_size, prob_full.shape[0] - y)
                valid_w = min(patch_size, prob_full.shape[1] - x)
                if valid_h > 0 and valid_w > 0:
                    prob_full[y:y+valid_h, x:x+valid_w, :] += probs_patch[:valid_h, :valid_w, :]
                    count_map[y:y+valid_h, x:x+valid_w] += 1
    
    prob_full = prob_full / np.maximum(count_map[:, :, np.newaxis], 1)
    pred_full = np.argmax(prob_full, axis=2).astype(np.uint8)
    return pred_full


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
    """Save side-by-side comparison visualization of original image, prediction, and ground truth."""
    compare_dir = os.path.join(test_save_path, 'compare')
    os.makedirs(compare_dir, exist_ok=True)
    
    # Create colormap
    cmap = ListedColormap(class_colors)
    n_classes = len(class_colors)
    
    if gt_class.shape != pred_full.shape:
        logging.warning(f"Resizing ground truth for {original_name}: {gt_class.shape} -> {pred_full.shape}")
        gt_class_pil = Image.fromarray(gt_class.astype(np.uint8), mode='L')
        gt_class_pil = gt_class_pil.resize((pred_full.shape[1], pred_full.shape[0]), Image.NEAREST)
        gt_class = np.array(gt_class_pil).astype(gt_class.dtype)
    
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
    """Compute segmentation metrics (TP, FP, FN) for each class and accumulate in provided arrays."""
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


def inference(args, model, test_save_path=None):
    """Run inference on historical document dataset and compute segmentation metrics."""
    pass  # Starting inference, no need to print
    
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
    
    logging.info(f"Found {len(patch_files)} patches")
    
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
        logging.info(f"Processing: {original_name}")
        
        # Estimate image dimensions
        max_x, max_y, patches_per_row = estimate_image_dimensions(
            original_name, original_img_dir, patches, patch_positions, patch_size
        )
        
        # Stitch patches together (with TTA if enabled)
        pred_full = stitch_patches(
            patches, patch_positions, max_x, max_y, 
            patches_per_row, patch_size, model, n_classes, 
            use_tta=args.use_tta, use_multiscale=args.multiscale,
            batch_size=args.batch_size
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
        
        # Save comparison visualization
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
                gt_class_pil = Image.fromarray(gt_class.astype(np.uint8), mode='L')
                gt_class_pil = gt_class_pil.resize((pred_full.shape[1], pred_full.shape[0]), Image.NEAREST)
                gt_class = np.array(gt_class_pil).astype(gt_class.dtype)
            
            compute_segmentation_metrics(pred_full, gt_class, n_classes, TP, FP, FN)
            num_processed_images += 1
        
    # Print final metrics
    print_final_metrics(TP, FP, FN, class_names, num_processed_images)
    
    return "Testing Finished!"


def main():
    """Main testing function."""
    # Set up basic logging first (before any logging calls)
    # This ensures all output goes to both console and file in order
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Parse and validate arguments
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
    
    try:
        validate_arguments(args)
    except ValueError as e:
        logging.error(f"ERROR: {e}")
        sys.exit(1)
    
    # Set up reproducible testing
    setup_reproducible_testing(args)
    
    # Load configuration and create model
    config = get_config(args)
    model = get_model(args, config)
    
    # Load trained model checkpoint
    try:
        checkpoint_name = load_model_checkpoint(model, args)
    except Exception as e:
        logging.error(f"ERROR: Failed to load model checkpoint: {e}")
        sys.exit(1)
    
    # Set up file logging (adds file handler to existing logger)
    log_folder = './test_log/test_log_'
    setup_logging(log_folder, checkpoint_name)
    
    # Set up test save directory
    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, "predictions")
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    
    # Print concise testing configuration
    tta_status = "TTA: ON" if args.use_tta else "TTA: OFF"
    multiscale_status = "Multi-scale: ON" if args.multiscale else "Multi-scale: OFF"
    logging.info(f"Testing: {args.manuscript} | {tta_status} | {multiscale_status}")
    logging.info("")
    
    try:
        result = inference(args, model, test_save_path)
        return result
        
    except Exception as e:
        logging.error(f"ERROR: Testing failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
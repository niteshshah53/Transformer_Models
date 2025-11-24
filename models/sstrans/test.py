"""
Testing Script for Historical Document Segmentation Models

This script evaluates trained models on historical document test datasets by:
- Loading trained model checkpoints
- Running inference on test images using patch-based approach
- Computing segmentation metrics (IoU, Precision, Recall, F1)
- Saving prediction visualizations

Supported datasets: U-DIADS-Bib, DIVAHISDB

Usage:
    # For SSTrans:
    python test.py --cfg config.yaml --output_dir ./models/ --manuscript Latin2 --is_savenii
    
Author: Clean Code Version
"""

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
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

# Add common directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from configs.config import get_config

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# ============================================================================
# IMPROVED INFERENCE CLASSES AND FUNCTIONS FOR MASSIVE SPEEDUP
# ============================================================================

class PatchDataset(Dataset):
    """Dataset for batch processing patches - provides 10-20x speedup"""
    def __init__(self, patch_paths):
        self.patch_paths = patch_paths
    
    def __len__(self):
        return len(self.patch_paths)
    
    def __getitem__(self, idx):
        patch = Image.open(self.patch_paths[idx]).convert("RGB")
        # Apply same normalization as training
        patch_tensor = TF.to_tensor(patch)
        patch_tensor = TF.normalize(patch_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return patch_tensor, self.patch_paths[idx]


def stitch_patches_batched(patches, patch_positions, max_x, max_y, 
                           patches_per_row, patch_size, model, batch_size=24):
    """
    IMPROVED: Batch process patches for massive speedup.
    
    This replaces the slow one-by-one processing in the original code.
    Expected speedup: 10-20x faster
    
    Args:
        patches: List of patch file paths
        patch_positions: Dict mapping patch paths to positions
        max_x, max_y: Dimensions of full image
        patches_per_row: Number of patches per row
        patch_size: Size of each patch
        model: Neural network model
        batch_size: Batch size for processing (adjust based on GPU memory)
        
    Returns:
        np.ndarray: Stitched prediction map
    """
    pred_full = np.zeros((max_y, max_x), dtype=np.float32)
    count_map = np.zeros((max_y, max_x), dtype=np.int32)
    
    # Create dataset and dataloader for batch processing
    dataset = PatchDataset(patches)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4, pin_memory=True)
    
    model.eval()
    print(f"Processing {len(patches)} patches in batches of {batch_size}...")
    
    with torch.no_grad():
        for batch_idx, (batch_tensors, batch_paths) in enumerate(dataloader):
            batch_tensors = batch_tensors.cuda()
            
            # Batch inference - this is where the speedup happens!
            outputs = model(batch_tensors)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Process each prediction in the batch
            for pred_patch, patch_path in zip(predictions, batch_paths):
                patch_id = patch_positions[patch_path]
                
                # Calculate position in full image
                x = (patch_id % patches_per_row) * patch_size
                y = (patch_id // patches_per_row) * patch_size
                
                # Add to prediction map with boundary checking
                valid_h = min(patch_size, pred_full.shape[0] - y)
                valid_w = min(patch_size, pred_full.shape[1] - x)
                
                if valid_h > 0 and valid_w > 0:
                    pred_full[y:y+valid_h, x:x+valid_w] += pred_patch[:valid_h, :valid_w]
                    count_map[y:y+valid_h, x:x+valid_w] += 1
            
            if batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx+1}/{len(dataloader)}")
    
    # Average overlapping predictions (though there shouldn't be any overlap in current implementation)
    pred_full = np.round(pred_full / np.maximum(count_map, 1)).astype(np.uint8)
    print("[+] Batch processing completed!")
    return pred_full


def apply_tta_single_patch(model, patch_tensor, num_classes):
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
        
    Returns:
        np.ndarray: Augmented prediction
    """
    predictions = []
    
    # Original
    with torch.no_grad():
        output = model(patch_tensor)
        predictions.append(torch.softmax(output, dim=1))
    
    # Horizontal flip
    with torch.no_grad():
        flipped = torch.flip(patch_tensor, dims=[3])
        output = model(flipped)
        output = torch.flip(output, dims=[3])
        predictions.append(torch.softmax(output, dim=1))
    
    # Vertical flip
    with torch.no_grad():
        flipped = torch.flip(patch_tensor, dims=[2])
        output = model(flipped)
        output = torch.flip(output, dims=[2])
        predictions.append(torch.softmax(output, dim=1))
    
    # 90 degree rotation
    with torch.no_grad():
        rotated = torch.rot90(patch_tensor, k=1, dims=[2, 3])
        output = model(rotated)
        output = torch.rot90(output, k=-1, dims=[2, 3])
        predictions.append(torch.softmax(output, dim=1))
    
    # Average all predictions
    avg_prediction = torch.stack(predictions).mean(dim=0)
    return torch.argmax(avg_prediction, dim=1).cpu().numpy()[0]


def stitch_patches_with_tta(patches, patch_positions, max_x, max_y,
                            patches_per_row, patch_size, model, num_classes):
    """
    Stitch patches using Test-Time Augmentation.
    Slower but more accurate than standard stitching (expected +2-5% F1).
    
    Args:
        patches: List of patch file paths
        patch_positions: Dict mapping patch paths to positions  
        max_x, max_y: Dimensions of full image
        patches_per_row: Number of patches per row
        patch_size: Size of each patch
        model: Neural network model
        num_classes: Number of segmentation classes
        
    Returns:
        np.ndarray: Stitched prediction map with TTA
    """
    pred_full = np.zeros((max_y, max_x), dtype=np.int32)
    count_map = np.zeros((max_y, max_x), dtype=np.int32)
    
    print(f"Processing {len(patches)} patches with Test-Time Augmentation...")
    
    for i, patch_path in enumerate(patches):
        patch_id = patch_positions[patch_path]
        
        # Calculate position in full image
        x = (patch_id % patches_per_row) * patch_size
        y = (patch_id // patches_per_row) * patch_size
        
        # Load patch and apply normalization
        patch = Image.open(patch_path).convert("RGB")
        patch_tensor = TF.to_tensor(patch).unsqueeze(0).cuda()
        patch_tensor = TF.normalize(patch_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        # Apply TTA
        pred_patch = apply_tta_single_patch(model, patch_tensor, num_classes)
        
        # Add to prediction map
        valid_h = min(patch_size, pred_full.shape[0] - y)
        valid_w = min(patch_size, pred_full.shape[1] - x)
        
        if valid_h > 0 and valid_w > 0:
            pred_full[y:y+valid_h, x:x+valid_w] += pred_patch[:valid_h, :valid_w]
            count_map[y:y+valid_h, x:x+valid_w] += 1
        
        if i % 50 == 0:
            print(f"  Processed patch {i+1}/{len(patches)} with TTA")
    
    pred_full = np.round(pred_full / np.maximum(count_map, 1)).astype(np.uint8)
    print("[+] TTA processing completed!")
    return pred_full


# ============================================================================
# ORIGINAL FUNCTIONS START HERE
# ============================================================================


def parse_arguments():
    """Parse command line arguments for testing script."""
    parser = argparse.ArgumentParser(
        description='Test SSTrans model on historical document datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on U-DIADS-Bib dataset with SSTrans
  python test.py --cfg ../../common/configs/swin_tiny_patch4_window7_224_lite.yaml --output_dir ./models/ \\
                 --dataset UDIADS_BIB --manuscript Latin2 --is_savenii
  
  # Test on DIVAHISDB dataset with SSTrans
  python test.py --cfg ../../common/configs/swin_tiny_patch4_window7_224_lite.yaml --dataset DIVAHISDB \\
                 --output_dir ./models/ --manuscript Latin2
        """
    )
    
    # Core arguments
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE",
                       help='Path to model configuration file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing trained model checkpoints')
    
    # Model selection
    parser.add_argument('--model', type=str, default='sstrans',
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
    
    # Inference optimization arguments
    parser.add_argument('--inference_batch_size', type=int, default=16,
                       help='Batch size for inference (default: 16, higher = faster but more memory)')
    parser.add_argument('--use_tta', action='store_true',
                       help='Enable Test-Time Augmentation for +2-5%% accuracy boost')
    
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
        raise ValueError("--cfg argument is required for SSTrans model")
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
            print(f"Using 6 classes for manuscript: {args.manuscript}")
        
        if not os.path.exists(args.udiadsbib_root):
            raise ValueError(f"U-DIADS-Bib dataset path not found: {args.udiadsbib_root}")
    elif args.dataset.upper() == "DIVAHISDB":
        args.num_classes = 4
        if not os.path.exists(args.divahisdb_root):
            raise ValueError(f"DIVAHISDB dataset path not found: {args.divahisdb_root}")


def get_model(args, config):
    """
    Create and load the SSTrans model.
    
    Args:
        args: Command line arguments
        config: Model configuration
        
    Returns:
        torch.nn.Module: Initialized model
    """
    from vision_transformer import SwinUnet as SSTrans_seg
    model = SSTrans_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    model.load_from(config)
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
    
    msg = model.load_state_dict(torch.load(checkpoint_path))
    print(f"Model checkpoint loaded: {msg}")
    
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
        from datasets.dataset_udiadsbib import rgb_to_class
        
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


def stitch_patches(patches, patch_positions, max_x, max_y, patches_per_row, patch_size, model):
    """
    Stitch together patch predictions into full image.
    
    Args:
        patches (list): List of patch file paths
        patch_positions (dict): Mapping of patch paths to positions
        max_x, max_y (int): Maximum image dimensions
        patches_per_row (int): Number of patches per row
        patch_size (int): Size of each patch
        model: Neural network model
        
    Returns:
        numpy.ndarray: Stitched prediction map
    """
    import torchvision.transforms.functional as TF
    
    pred_full = np.zeros((max_y, max_x), dtype=np.int32)
    count_map = np.zeros((max_y, max_x), dtype=np.int32)
    
    for patch_path in patches:
        patch_id = patch_positions[patch_path]
        
        # Calculate patch position
        x = (patch_id % patches_per_row) * patch_size
        y = (patch_id // patches_per_row) * patch_size
        
        # Load and process patch
        patch = Image.open(patch_path).convert("RGB")
        patch_tensor = TF.to_tensor(patch).unsqueeze(0).cuda()
        
        with torch.no_grad():
            output = model(patch_tensor)
            pred_patch = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        # Add to prediction map with boundary checking
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
    pred_full = np.round(pred_full / np.maximum(count_map, 1)).astype(np.uint8)
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
    """Save side-by-side comparison visualization."""
    compare_dir = os.path.join(test_save_path, 'compare')
    os.makedirs(compare_dir, exist_ok=True)
    
    # Create colormap
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
    """
    Compute segmentation metrics for each class.
    
    Args:
        pred_full: Prediction array
        gt_class: Ground truth array
        n_classes: Number of classes
        TP, FP, FN: Arrays to accumulate metrics
    """
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
        
        # Stitch patches together (choose method based on args)
        if args.use_tta:
            # Use Test-Time Augmentation for higher accuracy
            pred_full = stitch_patches_with_tta(
                patches, patch_positions, max_x, max_y,
                patches_per_row, patch_size, model, args.num_classes
            )
        else:
            # Use fast batch processing
            pred_full = stitch_patches_batched(
                patches, patch_positions, max_x, max_y,
                patches_per_row, patch_size, model,
                batch_size=args.inference_batch_size
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
    
    return "Testing Finished!"


def main():
    """Main testing function."""
    print("=== Historical Document Segmentation Testing ===")
    print()
    
    # Parse and validate arguments
    args = parse_arguments()
    
    try:
        validate_arguments(args)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Set up reproducible testing
    setup_reproducible_testing(args)
    
    # Load configuration and create model
    config = get_config(args)
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
    print(f"Model: {args.model}")
    print(f"Manuscript: {args.manuscript}")
    print(f"Save predictions: {args.is_savenii}")
    print()
    
    try:
        result = inference(args, model, test_save_path)
        print()
        print("=== TESTING COMPLETED SUCCESSFULLY ===")
        print(f"Results saved to: {test_save_path if test_save_path else 'No files saved'}")
        print("="*50)
        return result
        
    except Exception as e:
        print(f"ERROR: Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

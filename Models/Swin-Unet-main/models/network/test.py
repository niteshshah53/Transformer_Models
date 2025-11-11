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
    # Baseline flag (matching train.py)
    parser.add_argument('--use_baseline', action='store_true', default=False,
                       help='Use baseline CNN-Transformer (EfficientNet-B4 encoder + bottleneck + Swin decoder)')
    
    # Baseline enhancement flags (matching train.py, only used with --use_baseline)
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
    
    # Legacy flags (for backward compatibility, ignored when --use_baseline is set)
    parser.add_argument('--adapter_mode', type=str, default='external', 
                       choices=['external', 'streaming'],
                       help='[DEPRECATED: Use --use_baseline instead] Adapter placement mode')
    parser.add_argument('--bottleneck', action='store_true', default=False,
                       help='[DEPRECATED: Use --use_baseline instead] Enable bottleneck with 2 Swin blocks')
    
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
    
    # Validate baseline flag usage (matching train.py)
    use_baseline = getattr(args, 'use_baseline', False)
    
    # Component flags that should only work with --use_baseline
    component_flags = [
        ('use_deep_supervision', getattr(args, 'deep_supervision', False)),
        ('use_fourier_fusion', getattr(args, 'fusion_method', 'simple') == 'fourier'),
        ('use_smart_fusion', getattr(args, 'fusion_method', 'simple') == 'smart'),
        ('use_multiscale_agg', getattr(args, 'use_multiscale_agg', False)),
        ('use_groupnorm', getattr(args, 'use_groupnorm', False) if not use_baseline else False),
    ]
    
    # Check if component flags are used without --use_baseline
    if not use_baseline:
        used_flags = [name for name, used in component_flags if used]
        if used_flags:
            logging.warning(f"Component flags {used_flags} are typically used with --use_baseline flag")
            logging.warning("Consider using --use_baseline for baseline configuration")
    
    # Warn if legacy flags are used with --use_baseline
    if use_baseline:
        if getattr(args, 'adapter_mode', 'external') != 'external' or getattr(args, 'bottleneck', False):
            logging.warning("--use_baseline is set: --adapter_mode and --bottleneck flags will be ignored")
            logging.warning("Baseline uses: adapter_mode='streaming', bottleneck=True")


def get_model(args, config):
    """Create and load the CNN-Transformer model."""
    from vision_transformer_cnn import CNNTransformerUnet as ViT_seg
    
    use_baseline = getattr(args, 'use_baseline', False)
    
    if use_baseline:
        print("=" * 80)
        print("🚀 Loading CNN-Transformer BASELINE for Testing")
        print("=" * 80)
        print("Baseline Configuration:")
        print("  ✓ EfficientNet-B4 Encoder")
        print("  ✓ Bottleneck: 2 Swin Transformer blocks")
        print("  ✓ Swin Transformer Decoder")
        print("  ✓ Simple concatenation skip connections")
        print("  ✓ Adapter mode: streaming (default)")
        print("  ✓ GroupNorm: {}".format(getattr(args, 'use_groupnorm', True)))
        print("=" * 80)
        
        # Baseline defaults (matching train.py exactly)
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
            use_groupnorm=getattr(args, 'use_groupnorm', True)  # Default True for baseline (matching train.py)
        )
    else:
        # Original non-baseline mode (backward compatibility)
        # Use legacy flags if provided, otherwise use defaults
        adapter_mode = getattr(args, 'adapter_mode', 'external')
        use_bottleneck = getattr(args, 'bottleneck', False)
        
        model = ViT_seg(
            None,
            img_size=args.img_size,
            num_classes=args.num_classes,
            use_deep_supervision=getattr(args, 'deep_supervision', False),
            fusion_method=getattr(args, 'fusion_method', 'simple'),
            use_bottleneck=use_bottleneck,
            adapter_mode=adapter_mode,
            use_multiscale_agg=getattr(args, 'use_multiscale_agg', False),
            use_groupnorm=getattr(args, 'use_groupnorm', False)  # Default False for non-baseline
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
    else:
        fusion_mismatch = (has_skip_fusions or has_smart_skips)
    
    bottleneck_mismatch = (has_bottleneck != model.model.use_bottleneck)
    multiscale_mismatch = (has_multiscale_agg != model.model.use_multiscale_agg)
    
    if ds_mismatch or fusion_mismatch or bottleneck_mismatch or adapter_mismatch or multiscale_mismatch:
        logging.info("Checkpoint and model have architecture differences - loading with strict=False")
        msg = model.load_state_dict(model_state_dict, strict=False)
        
        if msg.unexpected_keys:
            logging.warning(f"Ignored {len(msg.unexpected_keys)} unexpected keys")
        if msg.missing_keys:
            missing_fusion = [k for k in msg.missing_keys if 'skip_fusions' in k or 'smart_skips' in k]
            missing_other = [k for k in msg.missing_keys if 'skip_fusions' not in k and 'smart_skips' not in k]
            if missing_fusion:
                logging.warning(f"Missing {len(missing_fusion)} fusion-related keys")
            if missing_other:
                logging.warning(f"Missing {len(missing_other)} other keys")
    else:
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


def predict_patch_with_tta(patch_tensor, model, return_probs=False):
    """Predict patch with test-time augmentation."""
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


def apply_crf_postprocessing(prob_map, rgb_image, num_classes=6, 
                              spatial_weight=3.0, spatial_x_stddev=3.0, spatial_y_stddev=3.0,
                              color_weight=10.0, color_stddev=50.0, num_iterations=10):
    """Apply DenseCRF post-processing to refine segmentation predictions."""
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


def stitch_patches(patches, patch_positions, max_x, max_y, patches_per_row, patch_size, model, use_tta=False, return_probs=False):
    """Stitch together patch predictions into full image."""
    import torchvision.transforms.functional as TF
    
    pred_full = np.zeros((max_y, max_x), dtype=np.int32)
    count_map = np.zeros((max_y, max_x), dtype=np.int32)
    
    if return_probs:
        num_classes = None
        prob_full = None
    
    for patch_path in patches:
        patch_id = patch_positions[patch_path]
        x = (patch_id % patches_per_row) * patch_size
        y = (patch_id // patches_per_row) * patch_size
        
        patch = Image.open(patch_path).convert("RGB")
        patch_tensor = TF.to_tensor(patch).unsqueeze(0).cuda()
        
        if use_tta:
            if return_probs:
                probs = predict_patch_with_tta(patch_tensor, model, return_probs=True)
                if prob_full is None:
                    num_classes = probs.shape[0]
                    prob_full = np.zeros((max_y, max_x, num_classes), dtype=np.float32)
                pred_patch = np.argmax(probs, axis=0)
            else:
                pred_patch = predict_patch_with_tta(patch_tensor, model, return_probs=False)
        else:
            with torch.no_grad():
                output = model(patch_tensor)
                if isinstance(output, tuple):
                    output = output[0]
                
                if return_probs:
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    if prob_full is None:
                        num_classes = probs.shape[0]
                        prob_full = np.zeros((max_y, max_x, num_classes), dtype=np.float32)
                
                pred_patch = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        if y + patch_size <= pred_full.shape[0] and x + patch_size <= pred_full.shape[1]:
            pred_full[y:y+patch_size, x:x+patch_size] += pred_patch
            count_map[y:y+patch_size, x:x+patch_size] += 1
            if return_probs:
                prob_full[y:y+patch_size, x:x+patch_size, :] += probs.transpose(1, 2, 0)
        else:
            valid_h = min(patch_size, pred_full.shape[0] - y)
            valid_w = min(patch_size, pred_full.shape[1] - x)
            if valid_h > 0 and valid_w > 0:
                pred_full[y:y+valid_h, x:x+valid_w] += pred_patch[:valid_h, :valid_w]
                count_map[y:y+valid_h, x:x+valid_w] += 1
                if return_probs:
                    prob_full[y:y+valid_h, x:x+valid_w, :] += probs[:, :valid_h, :valid_w].transpose(1, 2, 0)
    
    pred_full = np.round(pred_full / np.maximum(count_map, 1)).astype(np.uint8)
    
    if return_probs:
        prob_full = prob_full / np.maximum(count_map[:, :, np.newaxis], 1)
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
    
    for original_name, patches in patch_groups.items():
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
                patches_per_row, patch_size, model, use_tta=use_tta, return_probs=True
            )
            
            orig_img_rgb = None
            for ext in ['.jpg', '.png', '.tif', '.tiff']:
                orig_path = os.path.join(original_img_dir, f"{original_name}{ext}")
                if os.path.exists(orig_path):
                    orig_img_pil = Image.open(orig_path).convert("RGB")
                    if orig_img_pil.size != (max_x, max_y):
                        orig_img_pil = orig_img_pil.resize((max_x, max_y), Image.BILINEAR)
                    orig_img_rgb = np.array(orig_img_pil)
                    break
            
            if orig_img_rgb is not None:
                try:
                    logging.info(f"🎯 Applying CRF post-processing for {original_name}")
                    pred_full = apply_crf_postprocessing(
                        prob_full, orig_img_rgb, num_classes=n_classes,
                        spatial_weight=3.0, spatial_x_stddev=3.0, spatial_y_stddev=3.0,
                        color_weight=10.0, color_stddev=50.0, num_iterations=10
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
                patches_per_row, patch_size, model, use_tta=use_tta, return_probs=False
            )
        
        save_prediction_results(pred_full, original_name, class_colors, result_dir)
        
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
        
        logging.info(f"Completed: {original_name}")
    
    print_final_metrics(TP, FP, FN, class_names, num_processed_images)
    logging.info(f"Inference completed on {num_processed_images} images")
    
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
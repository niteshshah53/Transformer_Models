"""
GradCAM Visualization Script for CNN-Transformer Models
Generates visualizations for Baseline and all component variants:
- Baseline (Simple Skip)
- Smart Skip Connection
- Fourier Feature Fusion
- Multi-Scale Aggregation (MSA)
- Deep Supervision

Based on: https://github.com/jacobgil/pytorch-grad-cam
"""

import argparse
import os
import sys
import warnings
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import cv2

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

warnings.filterwarnings("ignore")

# Try to import grad-cam
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, XGradCAM, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRAD_CAM_AVAILABLE = True
except ImportError:
    print("⚠️  pytorch-grad-cam not installed. Installing...")
    print("   Run: pip install grad-cam")
    GRAD_CAM_AVAILABLE = False

from cnn_transformer import EfficientNetSwinUNet
from train import get_model


class SegmentationModelWrapper(torch.nn.Module):
    """
    Wrapper to make segmentation model compatible with GradCAM.
    GradCAM expects a model that outputs class scores, but our model outputs segmentation maps.
    """
    def __init__(self, model, target_class=1):
        super().__init__()
        self.model = model
        self.target_class = target_class  # Class to visualize (1=Paratext, 2=Decoration, etc.)
    
    def forward(self, x):
        output = self.model(x)
        if isinstance(output, tuple):
            # Deep supervision: use main output
            output = output[0]
        
        # Get logits for target class
        # output shape: (B, num_classes, H, W)
        # We want to get the activation for target_class
        if output.dim() == 4:
            # For segmentation: return the feature map for target class
            # GradCAM will compute gradients w.r.t. this
            return output[:, self.target_class, :, :].unsqueeze(1)
        return output


class EncoderTargetLayer:
    """Target layer for encoder features (EfficientNet stages)"""
    def __init__(self, model, stage_idx=3):
        self.model = model
        self.stage_idx = stage_idx  # 0, 1, 2, 3 for stages
    
    def __call__(self, x):
        # Get encoder features
        encoder_features = self.model.encoder(x)
        return encoder_features[self.stage_idx]


class DecoderTargetLayer:
    """Target layer for decoder features"""
    def __init__(self, model, layer_idx=2):
        self.model = model
        self.layer_idx = layer_idx  # 0, 1, 2, 3 for decoder layers
    
    def __call__(self, x):
        # Forward through encoder
        encoder_features = self.model.encoder(x)
        
        # Adapt features
        adapted_features = []
        if self.model.adapter_mode == 'external':
            for i, feat in enumerate(encoder_features):
                adapted = self.model.feature_adapters[i](feat)
                adapted_features.append(adapted)
        else:
            for i, feat in enumerate(encoder_features):
                y = self.model.streaming_proj[i](feat)
                y = y.flatten(2).transpose(1, 2)
                adapted_features.append(y)
        
        # Bottleneck
        x = self.model.norm(adapted_features[-1])
        if self.model.use_bottleneck:
            if self.model.use_multiscale_agg:
                import torch.nn.functional as F
                B, L, C = x.shape
                h = w = int(L ** 0.5)
                projected = []
                for i, feat in enumerate(adapted_features):
                    B_f, L_f, C_f = feat.shape
                    h_f = w_f = int(L_f ** 0.5)
                    proj_feat = self.model.multiscale_proj[i](feat)
                    proj_feat = proj_feat.view(B_f, h_f, w_f, C)
                    proj_feat = proj_feat.permute(0, 3, 1, 2)
                    proj_feat = F.interpolate(proj_feat, size=(h, w), mode='bilinear', align_corners=False)
                    proj_feat = proj_feat.permute(0, 2, 3, 1)
                    proj_feat = proj_feat.reshape(B, -1, C)
                    projected.append(proj_feat)
                aggregated = torch.cat(projected, dim=-1)
                fused = self.model.multiscale_fusion(aggregated)
                x = x + fused
            x = self.model.bottleneck_layer(x)
        
        # Decoder
        for inx, layer_up in enumerate(self.model.layers_up):
            if inx == 0:
                x = layer_up(x)
            elif inx <= self.layer_idx:
                skip_features = adapted_features[3 - inx]
                if self.model.fusion_method == 'fourier':
                    x = self.model.skip_fusions[inx - 1](x, skip_features)
                elif self.model.fusion_method == 'smart' and self.model.smart_skips is not None:
                    x = self.model.smart_skips[inx - 1](skip_features, x)
                elif self.model.fusion_method == 'simple' and self.model.simple_skips is not None:
                    x = self.model.simple_skips[inx - 1](skip_features, x)
                else:
                    x = torch.cat([x, skip_features], -1)
                    x = self.model.concat_back_dim[inx](x)
                x = layer_up(x)
            else:
                break
        
        # Convert tokens to spatial format for visualization
        if x.dim() == 3:  # (B, L, C)
            B, L, C = x.shape
            h = w = int(L ** 0.5)
            x = x.view(B, h, w, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        return x


class SkipConnectionTargetLayer:
    """Target layer for skip connection features (for Smart Skip attention visualization)"""
    def __init__(self, model, skip_idx=0):
        self.model = model
        self.skip_idx = skip_idx  # 0, 1, 2 for skip connections
    
    def __call__(self, x):
        encoder_features = self.model.encoder(x)
        
        # Adapt features
        adapted_features = []
        if self.model.adapter_mode == 'external':
            for i, feat in enumerate(encoder_features):
                adapted = self.model.feature_adapters[i](feat)
                adapted_features.append(adapted)
        else:
            for i, feat in enumerate(encoder_features):
                y = self.model.streaming_proj[i](feat)
                y = y.flatten(2).transpose(1, 2)
                adapted_features.append(y)
        
        # Get to the skip connection point
        x = self.model.norm(adapted_features[-1])
        if self.model.use_bottleneck:
            x = self.model.bottleneck_layer(x)
        
        # Forward to skip connection
        for inx, layer_up in enumerate(self.model.layers_up):
            if inx == 0:
                x = layer_up(x)
            elif inx == self.skip_idx + 1:
                # This is where skip connection happens
                skip_features = adapted_features[3 - inx]
                
                if self.model.fusion_method == 'smart' and self.model.smart_skips is not None:
                    # Get attention output from smart skip
                    skip_tokens = self.model.smart_skips[inx - 1].align(skip_features)
                    # Convert to spatial for visualization
                    B, L, C = skip_tokens.shape
                    h = w = int(L ** 0.5)
                    skip_spatial = skip_tokens.view(B, h, w, C).permute(0, 3, 1, 2)
                    return skip_spatial
                elif self.model.fusion_method == 'simple' and self.model.simple_skips is not None:
                    # Get projected encoder features
                    encoder_proj = self.model.simple_skips[inx - 1].proj(skip_features)
                    # Convert to spatial
                    B, L, C = encoder_proj.shape
                    h = w = int(L ** 0.5)
                    encoder_spatial = encoder_proj.view(B, h, w, C).permute(0, 3, 1, 2)
                    return encoder_spatial
                else:
                    # Convert to spatial
                    B, L, C = skip_features.shape
                    h = w = int(L ** 0.5)
                    skip_spatial = skip_features.view(B, h, w, C).permute(0, 3, 1, 2)
                    return skip_spatial
            else:
                if inx < self.skip_idx + 1:
                    x = layer_up(x)
        
        # Convert to spatial if still tokens
        if x.dim() == 3:
            B, L, C = x.shape
            h = w = int(L ** 0.5)
            x = x.view(B, h, w, C).permute(0, 3, 1, 2)
        return x


def load_model_and_checkpoint(model_dir, fusion_method='simple', use_deep_supervision=False,
                             use_multiscale_agg=False, use_bottleneck=True):
    """Load model with specific configuration"""
    from train import parse_arguments
    
    # Create args object
    class Args:
        pass
    
    args = Args()
    args.img_size = 224
    args.num_classes = 6
    args.efficientnet_model = 'tf_efficientnet_b4_ns'
    args.encoder_type = 'efficientnet'
    args.pretrained = True
    args.embed_dim = 96
    args.depths_decoder = [2, 2, 2, 2]
    args.num_heads = [3, 6, 12, 24]
    args.window_size = 7
    args.mlp_ratio = 4.0
    args.qkv_bias = True
    args.qk_scale = None
    args.drop_rate = 0.0
    args.attn_drop_rate = 0.0
    args.drop_path_rate = 0.1
    args.use_checkpoint = False
    args.use_deep_supervision = use_deep_supervision
    args.fusion_method = fusion_method
    args.use_bottleneck = use_bottleneck
    args.adapter_mode = 'streaming'
    args.use_multiscale_agg = use_multiscale_agg
    args.use_groupnorm = True
    
    # Create model
    model = EfficientNetSwinUNet(
        img_size=args.img_size,
        num_classes=args.num_classes,
        efficientnet_model=args.efficientnet_model,
        encoder_type=args.encoder_type,
        pretrained=args.pretrained,
        embed_dim=args.embed_dim,
        depths_decoder=args.depths_decoder,
        num_heads=args.num_heads,
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=args.qkv_bias,
        qk_scale=args.qk_scale,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        use_checkpoint=args.use_checkpoint,
        use_deep_supervision=args.use_deep_supervision,
        fusion_method=args.fusion_method,
        use_bottleneck=args.use_bottleneck,
        adapter_mode=args.adapter_mode,
        use_multiscale_agg=args.use_multiscale_agg,
        use_groupnorm=args.use_groupnorm
    )
    
    # Load checkpoint
    checkpoint_path = os.path.join(model_dir, 'best_model_latest.pth')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(model_dir, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model_state_dict = checkpoint['model_state']
    else:
        model_state_dict = checkpoint
    
    # Load with error handling
    try:
        model.load_state_dict(model_state_dict, strict=False)
    except Exception as e:
        print(f"⚠️  Warning: Some weights not loaded: {e}")
        model.load_state_dict(model_state_dict, strict=False)
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model


def get_sample_images(dataset_root, manuscript, num_samples=5, use_full_images=True, result_dir=None):
    """
    Get sample images from dataset.
    
    Args:
        dataset_root: Root directory of dataset
        manuscript: Manuscript name
        num_samples: Number of samples to get
        use_full_images: If True, use full images from predictions/result directory
        result_dir: Directory containing predictions/result (e.g., Result/a4/Latin2)
    
    Returns:
        List of image file paths
    """
    if use_full_images and result_dir:
        # Try to get full images from predictions/result directory
        result_image_dir = os.path.join(result_dir, 'predictions', 'result')
        if os.path.exists(result_image_dir):
            full_image_files = sorted(glob.glob(os.path.join(result_image_dir, '*.png')))[:num_samples]
            if len(full_image_files) > 0:
                print(f"  Using full images from: {result_image_dir}")
                return full_image_files
    
    # Fallback to patched images from dataset
    patch_dir = os.path.join(dataset_root, manuscript, 'Image', 'validation')
    if not os.path.exists(patch_dir):
        patch_dir = os.path.join(dataset_root, manuscript, 'Image', 'training')
    
    patch_files = sorted(glob.glob(os.path.join(patch_dir, '*.png')))[:num_samples]
    if len(patch_files) > 0:
        print(f"  Using patched images from: {patch_dir}")
    return patch_files


def preprocess_image_for_model(image_path, img_size=224, keep_aspect_ratio=False):
    """
    Preprocess image for model input.
    
    Args:
        image_path: Path to image file
        img_size: Target size (for square images) or max dimension
        keep_aspect_ratio: If True, resize maintaining aspect ratio and pad to square
    
    Returns:
        tuple: (image_tensor, image_np) where image_np is the original resized image for visualization
    """
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    
    if keep_aspect_ratio:
        # Resize maintaining aspect ratio, then pad to square
        ratio = min(img_size / original_size[0], img_size / original_size[1])
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        image = image.resize(new_size, Image.BILINEAR)
        
        # Create square image with padding
        square_image = Image.new('RGB', (img_size, img_size), (0, 0, 0))
        paste_x = (img_size - new_size[0]) // 2
        paste_y = (img_size - new_size[1]) // 2
        square_image.paste(image, (paste_x, paste_y))
        image = square_image
    else:
        # Simple resize to square
        image = image.resize((img_size, img_size), Image.BILINEAR)
    
    # Convert to tensor and normalize
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor, image_np


def create_custom_cam(model, input_tensor, target_layer, target_class=1):
    """Create custom GradCAM for segmentation model"""
    if not GRAD_CAM_AVAILABLE:
        return None
    
    # Wrap model for GradCAM
    wrapped_model = SegmentationModelWrapper(model, target_class=target_class)
    
    # Create target layer wrapper
    class TargetLayerWrapper(torch.nn.Module):
        def __init__(self, target_layer_func):
            super().__init__()
            self.target_layer_func = target_layer_func
        
        def forward(self, x):
            return self.target_layer_func(x)
    
    target_layer_model = TargetLayerWrapper(target_layer)
    
    # Create GradCAM (newer versions don't use use_cuda parameter, they auto-detect)
    cam = GradCAM(model=wrapped_model, target_layers=[target_layer_model])
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    
    return grayscale_cam[0]


def visualize_component_comparison(model_configs, image_paths, output_dir, dataset_root, manuscript):
    """Generate comparison visualizations for all components"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Component configurations
    components = {
        'Baseline': {
            'fusion_method': 'simple',
            'use_deep_supervision': False,
            'use_multiscale_agg': False,
            'model_dir': None  # Will be set based on args
        },
        'Baseline_DS': {
            'fusion_method': 'simple',
            'use_deep_supervision': True,
            'use_multiscale_agg': False,
            'model_dir': None
        },
        'Smart_Skip': {
            'fusion_method': 'smart',
            'use_deep_supervision': False,
            'use_multiscale_agg': False,
            'model_dir': None
        },
        'Fourier': {
            'fusion_method': 'fourier',
            'use_deep_supervision': False,
            'use_multiscale_agg': False,
            'model_dir': None
        },
        'MSA': {
            'fusion_method': 'simple',
            'use_deep_supervision': False,
            'use_multiscale_agg': True,
            'model_dir': None
        }
    }
    
    # Update model directories from configs
    for comp_name, comp_config in components.items():
        if comp_name in model_configs:
            comp_config['model_dir'] = model_configs[comp_name]
    
    # Process each image
    for img_idx, image_path in enumerate(image_paths[:3]):  # Limit to 3 images for demo
        print(f"\nProcessing image {img_idx + 1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Preprocess image
        input_tensor, image_np = preprocess_image_for_model(image_path)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        
        # Create figure for comparison
        fig, axes = plt.subplots(2, len(components) + 1, figsize=(5 * (len(components) + 1), 10))
        fig.suptitle(f'GradCAM Comparison - Image {img_idx + 1}: {os.path.basename(image_path)}', 
                     fontsize=16, fontweight='bold')
        
        # Show original image
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Process each component
        for comp_idx, (comp_name, comp_config) in enumerate(components.items(), 1):
            if comp_config['model_dir'] is None or not os.path.exists(comp_config['model_dir']):
                axes[0, comp_idx].text(0.5, 0.5, f'{comp_name}\nModel not found', 
                                       ha='center', va='center', fontsize=10)
                axes[0, comp_idx].axis('off')
                axes[1, comp_idx].axis('off')
                continue
            
            try:
                # Load model
                model = load_model_and_checkpoint(
                    comp_config['model_dir'],
                    fusion_method=comp_config['fusion_method'],
                    use_deep_supervision=comp_config['use_deep_supervision'],
                    use_multiscale_agg=comp_config['use_multiscale_agg']
                )
                
                # Get prediction
                with torch.no_grad():
                    output = model(input_tensor)
                    if isinstance(output, tuple):
                        output = output[0]
                    pred = torch.softmax(output, dim=1)
                    pred_mask = torch.argmax(pred, dim=1)[0].cpu().numpy()
                
                # Create GradCAM for encoder (stage 3)
                encoder_target = EncoderTargetLayer(model, stage_idx=3)
                cam_encoder = create_custom_cam(model, input_tensor, encoder_target, target_class=1)
                
                if cam_encoder is not None:
                    # Resize CAM to match image size
                    cam_encoder_resized = cv2.resize(cam_encoder, (224, 224))
                    cam_on_image = show_cam_on_image(image_np, cam_encoder_resized, use_rgb=True)
                    
                    axes[0, comp_idx].imshow(cam_on_image)
                    axes[0, comp_idx].set_title(f'{comp_name}\nEncoder GradCAM', fontsize=11)
                    axes[0, comp_idx].axis('off')
                else:
                    axes[0, comp_idx].text(0.5, 0.5, f'{comp_name}\nGradCAM failed', 
                                          ha='center', va='center', fontsize=10)
                    axes[0, comp_idx].axis('off')
                
                # Show prediction
                # Create colormap for segmentation
                colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], 
                                  [255, 255, 0], [255, 0, 255]], dtype=np.uint8)
                pred_colored = colors[pred_mask]
                pred_colored = pred_colored.astype(np.float32) / 255.0
                
                axes[1, comp_idx].imshow(pred_colored)
                axes[1, comp_idx].set_title(f'{comp_name}\nPrediction', fontsize=11)
                axes[1, comp_idx].axis('off')
                
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ⚠️  Error processing {comp_name}: {e}")
                axes[0, comp_idx].text(0.5, 0.5, f'{comp_name}\nError: {str(e)[:30]}', 
                                      ha='center', va='center', fontsize=9)
                axes[0, comp_idx].axis('off')
                axes[1, comp_idx].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'comparison_image_{img_idx + 1}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path}")


def visualize_attention_heatmaps(model, input_tensor, output_dir, image_name):
    """Visualize attention heatmaps for Smart Skip connections"""
    if model.fusion_method != 'smart' or model.smart_skips is None:
        return
    
    model.eval()
    with torch.no_grad():
        # Forward to get attention weights
        encoder_features = model.encoder(input_tensor)
        
        # Adapt features
        adapted_features = []
        if model.adapter_mode == 'external':
            for i, feat in enumerate(encoder_features):
                adapted = model.feature_adapters[i](feat)
                adapted_features.append(adapted)
        else:
            for i, feat in enumerate(encoder_features):
                y = model.streaming_proj[i](feat)
                y = y.flatten(2).transpose(1, 2)
                adapted_features.append(y)
        
        # Get to skip connection
        x = model.norm(adapted_features[-1])
        if model.use_bottleneck:
            x = model.bottleneck_layer(x)
        
        x = model.layers_up[0](x)
        
        # Visualize each skip connection
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Smart Skip Attention Heatmaps - {image_name}', fontsize=14, fontweight='bold')
        
        for skip_idx in range(min(3, len(model.smart_skips))):
            skip_features = adapted_features[2 - skip_idx]
            
            # Get attention from smart skip
            skip_tokens = model.smart_skips[skip_idx].align(skip_features)
            
            # Get attention weights - need to enable return_attention_weights
            # Since MultiheadAttention doesn't return weights by default, we'll use a workaround
            # Compute attention manually or use the output to create a proxy visualization
            
            # Alternative: Visualize the aligned skip tokens (attention-enhanced features)
            # Convert tokens to spatial
            B, L, C = skip_tokens.shape
            h = w = int(L ** 0.5)
            
            # Get feature magnitude as proxy for attention
            skip_spatial = skip_tokens.view(B, h, w, C).permute(0, 3, 1, 2)
            # Average across channels to get spatial importance map
            attn_map = skip_spatial[0].mean(dim=0).cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            
            attn_map_resized = cv2.resize(attn_map, (224, 224))
            
            im = axes[skip_idx].imshow(attn_map_resized, cmap='jet')
            axes[skip_idx].set_title(f'Skip Connection {skip_idx + 1}\n(H/{16 // (2**skip_idx)})', fontsize=11)
            axes[skip_idx].axis('off')
            plt.colorbar(im, ax=axes[skip_idx])
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'attention_heatmap_{image_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved attention heatmap: {output_path}")


def visualize_fourier_frequency(model, input_tensor, output_dir, image_name):
    """Visualize frequency domain for Fourier fusion"""
    if model.fusion_method != 'fourier':
        return
    
    model.eval()
    with torch.no_grad():
        # Get features before fusion
        encoder_features = model.encoder(input_tensor)
        
        # Adapt features
        adapted_features = []
        if model.adapter_mode == 'external':
            for i, feat in enumerate(encoder_features):
                adapted = model.feature_adapters[i](feat)
                adapted_features.append(adapted)
        else:
            for i, feat in enumerate(encoder_features):
                y = model.streaming_proj[i](feat)
                y = y.flatten(2).transpose(1, 2)
                adapted_features.append(y)
        
        # Get to skip connection point
        x = model.norm(adapted_features[-1])
        if model.use_bottleneck:
            x = model.bottleneck_layer(x)
        
        x = model.layers_up[0](x)
        skip_features = adapted_features[2]  # First skip connection
        
        # Get Fourier fusion features
        skip_fusion = model.skip_fusions[0]
        feat1_proj = skip_fusion.proj1(x)
        feat2_proj = skip_fusion.proj2(skip_features)
        
        # Reshape to spatial
        B, L1, C = feat1_proj.shape
        _, L2, _ = feat2_proj.shape
        H1 = W1 = int(L1 ** 0.5)
        H2 = W2 = int(L2 ** 0.5)
        
        feat1_2d = feat1_proj[0].view(H1, W1, C).permute(2, 0, 1).cpu().numpy()
        feat2_2d = feat2_proj[0].view(H2, W2, C).permute(2, 0, 1).cpu().numpy()
        
        # Resize feat2 to match feat1
        feat2_2d_resized = np.zeros_like(feat1_2d)
        for c in range(min(feat1_2d.shape[0], feat2_2d.shape[0])):
            feat2_2d_resized[c] = cv2.resize(feat2_2d[c], (H1, W1))
        
        # Compute FFT
        feat1_fft = np.fft.fft2(feat1_2d[0])
        feat2_fft = np.fft.fft2(feat2_2d_resized[0])
        
        # Get magnitude
        feat1_mag = np.abs(feat1_fft)
        feat2_mag = np.abs(feat2_fft)
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Fourier Frequency Domain - {image_name}', fontsize=14, fontweight='bold')
        
        axes[0, 0].imshow(np.log(feat1_mag + 1), cmap='hot')
        axes[0, 0].set_title('Decoder Features\nFrequency Magnitude', fontsize=11)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(np.log(feat2_mag + 1), cmap='hot')
        axes[0, 1].set_title('Encoder Skip Features\nFrequency Magnitude', fontsize=11)
        axes[0, 1].axis('off')
        
        # Fused magnitude
        fused_mag = 0.5 * feat1_mag + 0.5 * feat2_mag
        axes[1, 0].imshow(np.log(fused_mag + 1), cmap='hot')
        axes[1, 0].set_title('Fused Features\nFrequency Magnitude', fontsize=11)
        axes[1, 0].axis('off')
        
        # Difference
        diff = np.abs(feat1_mag - feat2_mag)
        axes[1, 1].imshow(np.log(diff + 1), cmap='coolwarm')
        axes[1, 1].set_title('Frequency Difference', fontsize=11)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'fourier_frequency_{image_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved frequency visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate GradCAM visualizations for all model components')
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='Root directory of dataset (e.g., ../../U-DIADS-Bib-MS_patched)')
    parser.add_argument('--manuscript', type=str, required=True,
                       help='Manuscript name (e.g., Latin2)')
    parser.add_argument('--output_dir', type=str, default='./gradcam_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--baseline_dir', type=str,
                       help='Model directory for baseline (simple skip)')
    parser.add_argument('--baseline_ds_dir', type=str,
                       help='Model directory for baseline + deep supervision')
    parser.add_argument('--smart_skip_dir', type=str,
                       help='Model directory for smart skip')
    parser.add_argument('--fourier_dir', type=str,
                       help='Model directory for fourier fusion')
    parser.add_argument('--msa_dir', type=str,
                       help='Model directory for multi-scale aggregation')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of sample images to process')
    
    args = parser.parse_args()
    
    if not GRAD_CAM_AVAILABLE:
        print("❌ Error: pytorch-grad-cam not installed!")
        print("   Install with: pip install grad-cam")
        return
    
    print("=" * 80)
    print("GradCAM Visualization for CNN-Transformer Models")
    print("=" * 80)
    
    # Get sample images - prefer full images from result directory
    # Try to get from the first available model directory
    result_dir_for_images = None
    if args.baseline_ds_dir and os.path.exists(args.baseline_ds_dir):
        result_dir_for_images = args.baseline_ds_dir
    elif args.baseline_dir and os.path.exists(args.baseline_dir):
        result_dir_for_images = args.baseline_dir
    
    image_paths = get_sample_images(
        args.dataset_root, 
        args.manuscript, 
        args.num_samples,
        use_full_images=True,
        result_dir=result_dir_for_images
    )
    if len(image_paths) == 0:
        print(f"❌ No images found in {args.dataset_root}/{args.manuscript}")
        return
    
    print(f"\n✓ Found {len(image_paths)} sample images")
    
    # Model configurations
    model_configs = {}
    if args.baseline_dir:
        model_configs['Baseline'] = args.baseline_dir
    if args.baseline_ds_dir:
        model_configs['Baseline_DS'] = args.baseline_ds_dir
    if args.smart_skip_dir:
        model_configs['Smart_Skip'] = args.smart_skip_dir
    if args.fourier_dir:
        model_configs['Fourier'] = args.fourier_dir
    if args.msa_dir:
        model_configs['MSA'] = args.msa_dir
    
    if len(model_configs) == 0:
        print("⚠️  Warning: No model directories provided. Using default paths.")
        # Try to find models in Result directories
        result_base = './Result'
        if os.path.exists(result_base):
            for subdir in os.listdir(result_base):
                subdir_path = os.path.join(result_base, subdir)
                if os.path.isdir(subdir_path):
                    manuscript_dir = os.path.join(subdir_path, args.manuscript)
                    if os.path.exists(manuscript_dir):
                        if 'baseline' in subdir.lower() and 'deepsupervision' in subdir.lower():
                            model_configs['Baseline_DS'] = manuscript_dir
                        elif 'smart' in subdir.lower():
                            model_configs['Smart_Skip'] = manuscript_dir
                        elif 'fourier' in subdir.lower():
                            model_configs['Fourier'] = manuscript_dir
    
    print(f"\n✓ Found {len(model_configs)} model configurations")
    for name, path in model_configs.items():
        print(f"  - {name}: {path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate comparison visualizations
    print("\n" + "=" * 80)
    print("Generating Component Comparison Visualizations...")
    print("=" * 80)
    visualize_component_comparison(model_configs, image_paths, args.output_dir, 
                                  args.dataset_root, args.manuscript)
    
    # Generate component-specific visualizations
    print("\n" + "=" * 80)
    print("Generating Component-Specific Visualizations...")
    print("=" * 80)
    
    for comp_name, model_dir in model_configs.items():
        if not os.path.exists(model_dir):
            continue
        
        print(f"\nProcessing {comp_name}...")
        
        # Determine configuration
        fusion_method = 'simple'
        use_deep_supervision = False
        use_multiscale_agg = False
        
        if 'smart' in comp_name.lower():
            fusion_method = 'smart'
        elif 'fourier' in comp_name.lower():
            fusion_method = 'fourier'
        if 'ds' in comp_name.lower() or 'deep' in comp_name.lower():
            use_deep_supervision = True
        if 'msa' in comp_name.lower():
            use_multiscale_agg = True
        
        try:
            model = load_model_and_checkpoint(
                model_dir,
                fusion_method=fusion_method,
                use_deep_supervision=use_deep_supervision,
                use_multiscale_agg=use_multiscale_agg
            )
            
            # Process first image for detailed analysis
            if len(image_paths) > 0:
                input_tensor, image_np = preprocess_image_for_model(image_paths[0])
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                
                image_name = os.path.splitext(os.path.basename(image_paths[0]))[0]
                
                # Smart Skip: Attention heatmaps
                if fusion_method == 'smart':
                    visualize_attention_heatmaps(model, input_tensor, args.output_dir, 
                                                f"{comp_name}_{image_name}")
                
                # Fourier: Frequency domain
                if fusion_method == 'fourier':
                    visualize_fourier_frequency(model, input_tensor, args.output_dir,
                                               f"{comp_name}_{image_name}")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ⚠️  Error processing {comp_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✓ Visualization Complete!")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()


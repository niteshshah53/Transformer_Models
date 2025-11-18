#!/usr/bin/env python3
"""
CNN-Transformer Vision Transformer for Historical Document Segmentation
Based on EfficientNetSwinUNet from cnn_transformer.py
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from cnn_transformer import EfficientNetSwinUNet, create_model


class CNNTransformerUnet(nn.Module):
    """
    CNN-Transformer U-Net wrapper for historical document segmentation.
    
    This class wraps the EfficientNetSwinUNet model from cnn_transformer.py
    to provide a consistent interface with the existing training/testing pipeline.
    """
    
    def __init__(self, config, img_size=224, num_classes=6, zero_head=False, vis=False, 
                 use_deep_supervision=False, fusion_method='simple', use_bottleneck=False, 
                 adapter_mode='external', use_multiscale_agg=False, use_groupnorm=False,
                 encoder_type='efficientnet'):
        """
        Initialize CNN-Transformer U-Net model.
        
        Args:
            config: Configuration object (not used directly, kept for compatibility)
            img_size: Input image size (default: 224)
            num_classes: Number of segmentation classes (default: 6)
            zero_head: Whether to zero initialize the head (not used, kept for compatibility)
            vis: Whether to enable visualization (not used, kept for compatibility)
            use_deep_supervision: Whether to enable deep supervision (default: False)
            fusion_method: Feature fusion method - 'simple', 'fourier', or 'smart' (default: 'simple')
            use_bottleneck: Whether to use bottleneck (default: False)
            adapter_mode: Adapter mode - 'external' or 'streaming' (default: 'external')
            use_multiscale_agg: Whether to use multi-scale aggregation (default: False)
            use_groupnorm: Whether to use GroupNorm in adapters (default: False)
            encoder_type: Encoder type - 'efficientnet' or 'resnet50' (default: 'efficientnet')
        """
        super(CNNTransformerUnet, self).__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_deep_supervision = use_deep_supervision
        self.fusion_method = fusion_method
        self.use_bottleneck = use_bottleneck
        self.adapter_mode = adapter_mode
        self.use_multiscale_agg = use_multiscale_agg
        self.use_groupnorm = use_groupnorm
        self.encoder_type = encoder_type
        
        # Create model configuration
        model_config = {
            'img_size': img_size,
            'num_classes': num_classes,
            'encoder_type': encoder_type,  # 'efficientnet' or 'resnet50'
            'efficientnet_model': 'tf_efficientnet_b4_ns',  # EfficientNet-B4 (same as Hybrid1)
            'pretrained': True,
            'embed_dim': 96,
            'depths_decoder': [2, 2, 2, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 7,
            'drop_rate': 0.0,
            'drop_path_rate': 0.1,
            'use_deep_supervision': use_deep_supervision,
            'fusion_method': fusion_method,
            'use_bottleneck': use_bottleneck,
            'adapter_mode': adapter_mode,
            'use_multiscale_agg': use_multiscale_agg,
            'use_groupnorm': use_groupnorm,  # Pass GroupNorm flag
        }
        
        # Create the CNN-Transformer model
        self.model = create_model(model_config)
        
        print(f"CNN-Transformer U-Net initialized:")
        print(f"  - Image size: {img_size}")
        print(f"  - Number of classes: {num_classes}")
        if encoder_type == 'resnet50':
            print(f"  - Encoder: ResNet-50 (official)")
        else:
            print(f"  - Encoder: EfficientNet-B4")
            print(f"  - EfficientNet variant: {model_config['efficientnet_model']}")
        print(f"  - Embed dimension: {model_config['embed_dim']}")
        if use_deep_supervision:
            print(f"  - ✅ Deep Supervision: ENABLED (3 auxiliary outputs)")
        else:
            print(f"  - Deep Supervision: DISABLED")
        print(f"  - Fusion Method: {fusion_method.upper()}")
        print(f"  - Bottleneck (2 Swin blocks): {'ENABLED' if use_bottleneck else 'DISABLED'}")
        if use_multiscale_agg:
            print(f"  - ✅ Multi-Scale Aggregation: ENABLED (bottleneck)")
        print(f"  - Adapter Mode: {adapter_mode.upper()}")
        if use_groupnorm:
            print(f"  - ✅ GroupNorm: ENABLED (in adapters)")
        else:
            print(f"  - GroupNorm: DISABLED (using LayerNorm)")
        
    def forward(self, x):
        """
        Forward pass through the CNN-Transformer model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, num_classes, H, W)
        """
        return self.model(x)
    
    def load_from(self, config):
        """
        Load pretrained weights (placeholder for compatibility).
        
        Args:
            config: Configuration object
            
        Note:
            This method is kept for compatibility with existing training scripts.
            The CNN-Transformer model uses EfficientNet pretrained weights automatically.
        """
        print("CNN-Transformer model uses EfficientNet pretrained weights automatically.")
        print("No additional pretrained weights loading required.")
        return True


# Alias for compatibility
SwinUnet = CNNTransformerUnet
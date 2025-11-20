"""
Hybrid2 Models: Main model classes and factory functions
Contains: Hybrid2Baseline with SimpleDecoder
"""

import torch
import torch.nn as nn

from .components import (
    SwinEncoder,
    SimpleDecoder,
    ResNet50Decoder,
    EfficientNetB4Decoder
)


# ============================================================================
# HYBRID2 BASELINE MODEL (Swin Encoder + Swin Bottleneck + Simple Decoder)
# ============================================================================

class Hybrid2Baseline(nn.Module):
    """Hybrid2 model with configurable enhancements via flags."""
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = decoder.num_classes
        self.img_size = getattr(decoder, 'img_size', 224)
    
    def forward(self, x):
        encoder_features = self.encoder(x)
        encoder_tokens = None
        
        # Extract Stage 4 tokens for bottleneck processing
        if self.decoder.use_bottleneck_swin or self.decoder.use_cross_attn:
            f4 = encoder_features[3]  # (B, 768, 7, 7)
            B, C, H, W = f4.shape
            encoder_tokens = f4.flatten(2).permute(0, 2, 1)  # (B, 49, 768)
        
        return self.decoder(encoder_features, encoder_tokens)
    
    def get_model_info(self):
        """Get comprehensive model information."""
        return {
            'model_type': 'Hybrid2Baseline',
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'decoder': self.decoder.get_model_info(),
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_hybrid2_baseline(num_classes=6, img_size=224, decoder='simple',
                           efficientnet_variant='b4',
                           use_deep_supervision=False, use_cbam=False, 
                           use_smart_skip=False, use_cross_attn=False,
                           use_multiscale_agg=False, use_groupnorm=True,
                           use_pos_embed=True):
    """
    Create Hybrid2 model with configurable decoder and enhancements.
    
    Args:
        decoder: Decoder type - 'simple', 'EfficientNet-B4', or 'ResNet50'
        All other flags control enhancements (same as before)
    """
    encoder = SwinEncoder(
        img_size=img_size,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True
    )
    
    encoder_channels = encoder.get_channels()
    
    # Select decoder based on decoder type
    if decoder.lower() == 'resnet50':
        decoder_module = ResNet50Decoder(
            encoder_channels=encoder_channels,
            num_classes=num_classes,
            use_deep_supervision=use_deep_supervision,
            use_cbam=use_cbam,
            use_smart_skip=use_smart_skip,
            use_cross_attn=use_cross_attn,
            use_multiscale_agg=use_multiscale_agg,
            use_groupnorm=use_groupnorm,
            use_pos_embed=use_pos_embed,
            use_bottleneck_swin=True,
            img_size=img_size
        )
    elif decoder.lower() == 'efficientnet-b4' or decoder.lower() == 'efficientnet_b4':
        # Use REAL EfficientNet-B4 decoder with MBConv blocks
        decoder_module = EfficientNetB4Decoder(
            encoder_channels=encoder_channels,
            num_classes=num_classes,
            use_deep_supervision=use_deep_supervision,
            use_cbam=use_cbam,
            use_smart_skip=use_smart_skip,
            use_cross_attn=use_cross_attn,
            use_multiscale_agg=use_multiscale_agg,
            use_groupnorm=use_groupnorm,
            use_pos_embed=use_pos_embed,
            use_bottleneck_swin=True,
            img_size=img_size
        )
    else:  # 'simple' or default
        decoder_module = SimpleDecoder(
        encoder_channels=encoder_channels,
        num_classes=num_classes,
        efficientnet_variant=efficientnet_variant,
        use_deep_supervision=use_deep_supervision,
        use_cbam=use_cbam,
        use_smart_skip=use_smart_skip,
        use_cross_attn=use_cross_attn,
        use_multiscale_agg=use_multiscale_agg,
        use_groupnorm=use_groupnorm,
        use_pos_embed=use_pos_embed,
        use_bottleneck_swin=True,
        img_size=img_size
    )
    
    return Hybrid2Baseline(encoder, decoder_module)


# Remove all Hybrid2Enhanced classes and factory functions
# Remove Hybrid2Enhanced (lines 18-157)
# Remove Hybrid2EnhancedEfficientNet (lines 159-299)
# Remove create_hybrid2_enhanced functions (lines 347-393)
# Remove create_hybrid2_efficientnet function (lines 396-404)
# Remove convenience aliases (lines 465-466)
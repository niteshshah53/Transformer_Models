"""
Hybrid2 Models: Main model classes and factory functions
Contains: Hybrid2Enhanced, Hybrid2EnhancedEfficientNet, Hybrid2Baseline
"""

import torch
import torch.nn as nn

from .components import (
    SwinEncoder,
    create_hybrid2_enhanced_decoder,
    create_enhanced_efficientnet_decoder,
    BaselineHybrid2Decoder
)


# ============================================================================
# HYBRID2 ENHANCED MODEL (Full Transformer-CNN Hybrid Best Practices)
# ============================================================================

class Hybrid2Enhanced(nn.Module):
    """
    Hybrid2 model with transformer-CNN hybrid best practices implemented.
    
    Architecture: Swin Encoder → Enhanced Decoder
    
    Improvements:
    ✅ Tier 1: Deep Supervision + Differential LR (in trainer)
    ✅ Tier 2: GroupNorm + Positional Embeddings + Multi-Scale Aggregation
    ✅ Tier 3: Cross-Attention Bottleneck
    
    Expected Performance:
    - Baseline Hybrid2: IoU 0.36
    - With Tier 1: IoU 0.48-0.52 (+33-44%)
    - With Tier 2: IoU 0.52-0.55 (+44-53%)
    - With Tier 3: IoU 0.56-0.60 (+56-67%)
    """
    
    def __init__(self, num_classes: int = 6, img_size: int = 224, embed_dim: int = 96,
                 depths: list = [2, 2, 2, 2], num_heads: list = [3, 6, 12, 24],
                 use_deep_supervision: bool = True, use_cross_attn: bool = True,
                 use_multiscale_agg: bool = True):
        super().__init__()
        
        if num_classes not in [4, 5, 6]:
            raise ValueError(f"num_classes must be 4, 5, or 6, got {num_classes}")
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.use_deep_supervision = use_deep_supervision
        self.use_cross_attn = use_cross_attn
        self.use_multiscale_agg = use_multiscale_agg
        
        print("=" * 80)
        print("🚀 HYBRID2 Enhanced (Transformer-CNN Hybrid)")
        print("=" * 80)
        
        # Swin Encoder
        self.encoder = SwinEncoder(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )
        
        encoder_channels = self.encoder.get_channels()
        
        # Enhanced Decoder
        self.decoder = create_hybrid2_enhanced_decoder(
            encoder_channels=encoder_channels,
            num_classes=num_classes,
            use_deep_supervision=use_deep_supervision,
            use_cross_attn=use_cross_attn,
            use_multiscale_agg=use_multiscale_agg
        )
        
        print("\n📊 Model Configuration:")
        print(f"  • Input size: {img_size}x{img_size}")
        print(f"  • Output classes: {num_classes}")
        print(f"  • Encoder channels: {encoder_channels}")
        print(f"  • Embed dim: {embed_dim}")
        print(f"  • Depths: {depths}")
        print(f"  • Num heads: {num_heads}")
        
        print("\n✅ Transformer-CNN Hybrid Features Enabled:")
        print(f"  • Deep Supervision: {use_deep_supervision}")
        print(f"  • Cross-Attention: {use_cross_attn}")
        print(f"  • Multi-Scale Aggregation: {use_multiscale_agg}")
        print(f"  • GroupNorm: ✓ (always enabled)")
        print(f"  • Positional Embeddings: ✓ (always enabled)")
        print(f"  • Differential LR: ✓ (handled by trainer)")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n📈 Parameters:")
        print(f"  • Total: {total_params:,}")
        print(f"  • Trainable: {trainable_params:,}")
        print("=" * 80)
    
    def forward(self, x: torch.Tensor):
        """Forward pass through Hybrid2 Enhanced model."""
        encoder_features = self.encoder(x)
        
        encoder_tokens = None
        if self.use_cross_attn:
            f4 = encoder_features[3]
            B, C, H, W = f4.shape
            encoder_tokens = f4.flatten(2).permute(0, 2, 1)
        
        output = self.decoder(encoder_features, encoder_tokens)
        return output
    
    def get_model_info(self) -> dict:
        """Get comprehensive model information."""
        encoder_info = {
            'encoder_type': 'SwinTransformer',
            'embed_dim': self.embed_dim,
            'channels': self.encoder.get_channels(),
            'strides': self.encoder.get_strides()
        }
        
        decoder_info = self.decoder.get_model_info()
        
        return {
            'model_type': 'Hybrid2Enhanced',
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'encoder': encoder_info,
            'decoder': decoder_info,
            'hybrid_features': {
                'deep_supervision': self.use_deep_supervision,
                'cross_attention': self.use_cross_attn,
                'multiscale_aggregation': self.use_multiscale_agg,
                'groupnorm': True,
                'positional_embeddings': True,
                'differential_lr': 'handled_by_trainer'
            },
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# ============================================================================
# HYBRID2 ENHANCED EFFICIENTNET MODEL (CNN Decoder + transformer-CNN hybrid improvements)
# ============================================================================

class Hybrid2EnhancedEfficientNet(nn.Module):
    """
    Hybrid2 with Enhanced EfficientNet Decoder (Pure CNN + transformer-CNN hybrid improvements).
    
    Architecture: Swin Encoder → Enhanced EfficientNet Decoder (CNN)
    
    Key Features:
    - Pure CNN decoder (EfficientNet-style architecture)
    - Deep Supervision (auxiliary outputs)
    - Cross-Attention Bottleneck (active encoder querying)
    - Multi-Scale Aggregation (combine all encoder scales)
    - GroupNorm (better for small batches)
    - CBAM Attention (channel + spatial)
    - 2D Positional Embeddings
    - Smart Skip Connections
    
    Expected Performance: IoU 0.60-0.65 (vs baseline 0.36, improvement +67-81%)
    """
    
    def __init__(self, num_classes: int = 6, img_size: int = 224, embed_dim: int = 96,
                 depths: list = [2, 2, 2, 2], num_heads: list = [3, 6, 12, 24],
                 use_deep_supervision: bool = True, use_cross_attn: bool = True,
                 use_multiscale_agg: bool = True):
        super().__init__()
        
        if num_classes not in [4, 5, 6]:
            raise ValueError(f"num_classes must be 4, 5, or 6, got {num_classes}")
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.use_deep_supervision = use_deep_supervision
        self.use_cross_attn = use_cross_attn
        self.use_multiscale_agg = use_multiscale_agg
        
        print("=" * 80)
        print("🚀 HYBRID2 Enhanced EfficientNet (Transformer-CNN Hybrid)")
        print("=" * 80)
        
        # Swin Encoder
        self.encoder = SwinEncoder(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )
        
        encoder_channels = self.encoder.get_channels()
        
        # Enhanced EfficientNet Decoder (CNN with transformer-CNN hybrid improvements)
        self.decoder = create_enhanced_efficientnet_decoder(
            encoder_channels=encoder_channels,
            num_classes=num_classes,
            use_deep_supervision=use_deep_supervision,
            use_cross_attn=use_cross_attn,
            use_multiscale_agg=use_multiscale_agg
        )
        
        print("\n📊 Model Configuration:")
        print(f"  • Input size: {img_size}x{img_size}")
        print(f"  • Output classes: {num_classes}")
        print(f"  • Encoder: Swin Transformer")
        print(f"  • Decoder: Enhanced EfficientNet (Pure CNN)")
        print(f"  • Encoder channels: {encoder_channels}")
        
        print("\n✅ Transformer-CNN Hybrid Features Enabled:")
        print(f"  • Deep Supervision: {use_deep_supervision}")
        print(f"  • Cross-Attention: {use_cross_attn}")
        print(f"  • Multi-Scale Aggregation: {use_multiscale_agg}")
        print(f"  • GroupNorm: ✓ (always enabled)")
        print(f"  • CBAM Attention: ✓ (always enabled)")
        print(f"  • Positional Embeddings: ✓ (always enabled)")
        print(f"  • Differential LR: ✓ (handled by trainer)")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n📈 Parameters:")
        print(f"  • Total: {total_params:,}")
        print(f"  • Trainable: {trainable_params:,}")
        print("=" * 80)
    
    def forward(self, x: torch.Tensor):
        """Forward pass."""
        encoder_features = self.encoder(x)
        
        encoder_tokens = None
        if self.use_cross_attn:
            f4 = encoder_features[3]
            B, C, H, W = f4.shape
            encoder_tokens = f4.flatten(2).permute(0, 2, 1)
        
        output = self.decoder(encoder_features, encoder_tokens)
        return output
    
    def get_model_info(self) -> dict:
        """Get comprehensive model information."""
        encoder_info = {
            'encoder_type': 'SwinTransformer',
            'embed_dim': self.embed_dim,
            'channels': self.encoder.get_channels(),
            'strides': self.encoder.get_strides()
        }
        
        decoder_info = self.decoder.get_model_info()
        
        return {
            'model_type': 'Hybrid2EnhancedEfficientNet',
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'encoder': encoder_info,
            'decoder': decoder_info,
            'hybrid_features': {
                'deep_supervision': self.use_deep_supervision,
                'cross_attention': self.use_cross_attn,
                'multiscale_aggregation': self.use_multiscale_agg,
                'groupnorm': True,
                'cbam_attention': True,
                'positional_embeddings': True,
                'differential_lr': 'handled_by_trainer'
            },
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# ============================================================================
# HYBRID2 BASELINE MODEL (Swin Encoder + Swin Bottleneck + EfficientNet Decoder)
# ============================================================================

class Hybrid2Baseline(nn.Module):
    """Baseline Hybrid2 model with configurable enhancements."""
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = decoder.num_classes
        self.img_size = getattr(decoder, 'img_size', 224)
    
    def forward(self, x):
        encoder_features = self.encoder(x)
        encoder_tokens = None
        if self.decoder.use_cross_attn:
            f4 = encoder_features[3]
            B, C, H, W = f4.shape
            encoder_tokens = f4.flatten(2).permute(0, 2, 1)
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
# FACTORY FUNCTIONS
# ============================================================================

def create_hybrid2_enhanced(num_classes: int = 6, img_size: int = 224, embed_dim: int = 96,
                             use_deep_supervision: bool = True, use_cross_attn: bool = True,
                             use_multiscale_agg: bool = True):
    """Create Hybrid2 model with transformer-CNN hybrid best practices."""
    return Hybrid2Enhanced(
        num_classes=num_classes,
        img_size=img_size,
        embed_dim=embed_dim,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        use_deep_supervision=use_deep_supervision,
        use_cross_attn=use_cross_attn,
        use_multiscale_agg=use_multiscale_agg
    )


def create_hybrid2_enhanced_full(num_classes: int = 6, img_size: int = 224):
    """Create Hybrid2 Enhanced with ALL features enabled (best performance)."""
    return create_hybrid2_enhanced(
        num_classes=num_classes,
        img_size=img_size,
        use_deep_supervision=True,
        use_cross_attn=True,
        use_multiscale_agg=True
    )


def create_hybrid2_enhanced_lite(num_classes: int = 6, img_size: int = 224):
    """Create Hybrid2 Enhanced with only Tier 1+2 features (faster training)."""
    return create_hybrid2_enhanced(
        num_classes=num_classes,
        img_size=img_size,
        use_deep_supervision=True,
        use_cross_attn=False,
        use_multiscale_agg=True
    )


def create_hybrid2_enhanced_minimal(num_classes: int = 6, img_size: int = 224):
    """Create Hybrid2 Enhanced with only Tier 1 features (fastest)."""
    return create_hybrid2_enhanced(
        num_classes=num_classes,
        img_size=img_size,
        use_deep_supervision=True,
        use_cross_attn=False,
        use_multiscale_agg=False
    )


def create_hybrid2_efficientnet(num_classes: int = 6, img_size: int = 224):
    """Create Hybrid2 with Enhanced EfficientNet Decoder (CNN + transformer-CNN hybrid)."""
    return Hybrid2EnhancedEfficientNet(
        num_classes=num_classes,
        img_size=img_size,
        use_deep_supervision=True,
        use_cross_attn=True,
        use_multiscale_agg=True
    )


def create_hybrid2_baseline(num_classes=6, img_size=224, efficientnet_variant='b4',
                           use_deep_supervision=False, use_cbam=False, 
                           use_smart_skip=False, use_cross_attn=False,
                           use_multiscale_agg=False, use_groupnorm=False,
                           use_pos_embed=True):
    """
    Create baseline Hybrid2 model with configurable enhancements.
    
    Baseline includes:
    - Swin Encoder (4 stages)
    - Bottleneck: 2 Swin Transformer blocks
    - EfficientNet decoder (B0 or B4)
    - Simple skip connections (token -> CNN conversion)
    - BatchNorm (baseline), GroupNorm (optional)
    - Positional embeddings (default: True, matching SwinUnet pattern)
    
    All enhancements disabled by default except positional embeddings.
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
    
    decoder = BaselineHybrid2Decoder(
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
    
    return Hybrid2Baseline(encoder, decoder)


# ============================================================================
# CONVENIENCE ALIASES
# ============================================================================

Hybrid2BestPractices = Hybrid2Enhanced
create_hybrid2_best = create_hybrid2_enhanced_full

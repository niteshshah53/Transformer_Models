"""
Hybrid2 Model with TransUNet Best Practices
Full implementation of all Tier 1, 2, and 3 improvements
"""

import torch
import torch.nn as nn

from .swin_encoder import SwinEncoder
from .transunet_decoder import create_transunet_enhanced_decoder


class Hybrid2TransUNet(nn.Module):
    """
    Hybrid2 model with ALL TransUNet best practices implemented.
    
    Architecture: Swin Encoder → TransUNet-Enhanced Decoder
    
    TransUNet Improvements:
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
        
        # Validate num_classes
        if num_classes not in [4, 5, 6]:
            raise ValueError(f"num_classes must be 4, 5, or 6, got {num_classes}")
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.use_deep_supervision = use_deep_supervision
        self.use_cross_attn = use_cross_attn
        self.use_multiscale_agg = use_multiscale_agg
        
        print("=" * 80)
        print("🚀 HYBRID2 with TransUNet Best Practices")
        print("=" * 80)
        
        # ==================================================================
        # SWIN ENCODER (Pretrained)
        # ==================================================================
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
        
        # Get encoder channels
        encoder_channels = self.encoder.get_channels()  # [96, 192, 384, 768]
        
        # ==================================================================
        # TRANSUNET-ENHANCED DECODER
        # ==================================================================
        self.decoder = create_transunet_enhanced_decoder(
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
        
        print("\n✅ TransUNet Features Enabled:")
        print(f"  • Deep Supervision: {use_deep_supervision}")
        print(f"  • Cross-Attention: {use_cross_attn}")
        print(f"  • Multi-Scale Aggregation: {use_multiscale_agg}")
        print(f"  • GroupNorm: ✓ (always enabled)")
        print(f"  • Positional Embeddings: ✓ (always enabled)")
        print(f"  • Differential LR: ✓ (handled by trainer)")
        
        # Calculate parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n📈 Parameters:")
        print(f"  • Total: {total_params:,}")
        print(f"  • Trainable: {trainable_params:,}")
        print("=" * 80)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through Hybrid2-TransUNet model.
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            If use_deep_supervision:
                (main_output, [aux_out1, aux_out2, aux_out3])
            Else:
                main_output
            
            All outputs are [B, num_classes, H, W]
        """
        # Encode: Get 4 feature levels + optionally tokens
        encoder_features = self.encoder(x)  # [F1, F2, F3, F4]
        
        # For cross-attention, we need encoder tokens
        # Swin encoder outputs feature maps, need to convert last one to tokens
        encoder_tokens = None
        if self.use_cross_attn:
            # Convert last feature map to tokens for cross-attention
            # F4 shape: [B, 768, H/32, W/32]
            f4 = encoder_features[3]
            B, C, H, W = f4.shape
            encoder_tokens = f4.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Decode
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
            'model_type': 'Hybrid2TransUNet',
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'encoder': encoder_info,
            'decoder': decoder_info,
            'transunet_features': {
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
# FACTORY FUNCTIONS
# ============================================================================

def create_hybrid2_transunet(num_classes: int = 6, img_size: int = 224, embed_dim: int = 96,
                             use_deep_supervision: bool = True, use_cross_attn: bool = True,
                             use_multiscale_agg: bool = True):
    """
    Create Hybrid2 model with full TransUNet best practices.
    
    Args:
        num_classes: Number of segmentation classes (4, 5, or 6)
        img_size: Input image size
        embed_dim: Swin encoder embedding dimension
        use_deep_supervision: Enable auxiliary outputs (recommended)
        use_cross_attn: Enable cross-attention bottleneck (high impact)
        use_multiscale_agg: Enable multi-scale aggregation (medium impact)
    
    Returns:
        Hybrid2TransUNet model
    """
    return Hybrid2TransUNet(
        num_classes=num_classes,
        img_size=img_size,
        embed_dim=embed_dim,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        use_deep_supervision=use_deep_supervision,
        use_cross_attn=use_cross_attn,
        use_multiscale_agg=use_multiscale_agg
    )


def create_hybrid2_transunet_full(num_classes: int = 6, img_size: int = 224):
    """
    Create Hybrid2-TransUNet with ALL features enabled (best performance).
    
    Expected: IoU 0.56-0.60 (56-67% improvement over baseline 0.36)
    """
    return create_hybrid2_transunet(
        num_classes=num_classes,
        img_size=img_size,
        use_deep_supervision=True,
        use_cross_attn=True,
        use_multiscale_agg=True
    )


def create_hybrid2_transunet_lite(num_classes: int = 6, img_size: int = 224):
    """
    Create Hybrid2-TransUNet with only Tier 1+2 features (faster training).
    
    Expected: IoU 0.52-0.55 (44-53% improvement over baseline 0.36)
    """
    return create_hybrid2_transunet(
        num_classes=num_classes,
        img_size=img_size,
        use_deep_supervision=True,
        use_cross_attn=False,  # Disable expensive cross-attention
        use_multiscale_agg=True
    )


def create_hybrid2_transunet_minimal(num_classes: int = 6, img_size: int = 224):
    """
    Create Hybrid2-TransUNet with only Tier 1 features (fastest).
    
    Expected: IoU 0.48-0.52 (33-44% improvement over baseline 0.36)
    """
    return create_hybrid2_transunet(
        num_classes=num_classes,
        img_size=img_size,
        use_deep_supervision=True,
        use_cross_attn=False,
        use_multiscale_agg=False
    )


# ============================================================================
# ENHANCED EFFICIENTNET DECODER MODEL (CNN Decoder + TransUNet improvements)
# ============================================================================

class Hybrid2EnhancedEfficientNet(nn.Module):
    """
    Hybrid2 with Enhanced EfficientNet Decoder (Pure CNN + TransUNet best practices).
    
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
        print("🚀 HYBRID2 with Enhanced EfficientNet Decoder (CNN + TransUNet)")
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
        
        encoder_channels = self.encoder.get_channels()  # [96, 192, 384, 768]
        
        # Enhanced EfficientNet Decoder (CNN with TransUNet improvements)
        from .transunet_improvements import create_enhanced_efficientnet_decoder
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
        
        print("\n✅ TransUNet Features Enabled:")
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
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            If use_deep_supervision:
                (main_output, [aux_out1, aux_out2, aux_out3])
            Else:
                main_output
        """
        # Encode
        encoder_features = self.encoder(x)  # [F1, F2, F3, F4]
        
        # For cross-attention, convert last feature map to tokens
        encoder_tokens = None
        if self.use_cross_attn:
            f4 = encoder_features[3]
            B, C, H, W = f4.shape
            encoder_tokens = f4.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Decode
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
            'transunet_features': {
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


def create_hybrid2_efficientnet(num_classes: int = 6, img_size: int = 224):
    """
    Create Hybrid2 with Enhanced EfficientNet Decoder (CNN + TransUNet).
    
    Expected Performance: IoU 0.60-0.65 (vs baseline 0.36)
    
    Args:
        num_classes: Number of segmentation classes
        img_size: Input image size
    
    Returns:
        Hybrid2EnhancedEfficientNet model
    """
    return Hybrid2EnhancedEfficientNet(
        num_classes=num_classes,
        img_size=img_size,
        use_deep_supervision=True,
        use_cross_attn=True,
        use_multiscale_agg=True
    )


# ============================================================================
# CONVENIENCE ALIASES
# ============================================================================

Hybrid2BestPractices = Hybrid2TransUNet
create_hybrid2_best = create_hybrid2_transunet_full


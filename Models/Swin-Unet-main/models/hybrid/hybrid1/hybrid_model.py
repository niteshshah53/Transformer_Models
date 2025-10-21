import torch
import torch.nn as nn

from .efficientnet_encoder import EfficientNetEncoderWithAdapters
from .swin_decoder import SwinDecoder


# ============================================================================
# HYBRID MODEL: EfficientNet-B4 Encoder + Swin-Unet Decoder
# ============================================================================

class HybridEfficientNetB4SwinDecoder(nn.Module):
    """
    Hybrid model that combines EfficientNet-B4 encoder with Swin-Unet decoder.
    
    This model replaces the Swin-Unet encoder with an EfficientNet-B4 CNN encoder
    while keeping the Swin-Unet decoder intact for segmentation.
    
    Pipeline:
      1. EfficientNet-B4 backbone extracts multi-scale features (strides 4, 8, 16, 32)
      2. Channel adapters map CNN channels to Swin decoder expected dimensions [96, 192, 384, 768]
      3. Features are converted to token sequences and fed into Swin-Unet decoder
      4. Swin decoder performs upsampling with skip connections to produce segmentation masks
    
    Args:
        num_classes: Number of segmentation classes (4, 5, or 6, default: 6)
        img_size: Input image size (default: 224)
        pretrained: Whether to use pretrained EfficientNet weights (default: True)
    """
    
    def __init__(self, num_classes: int = 6, img_size: int = 224, pretrained: bool = True,
                 use_deep_supervision: bool = False, use_multiscale_agg: bool = False,
                 use_smart_skip: bool = False):
        super().__init__()
        
        # Validate num_classes - support 4, 5, and 6 classes
        if num_classes not in [4, 5, 6]:
            raise ValueError(f"num_classes must be 4, 5, or 6, got {num_classes}")
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_deep_supervision = use_deep_supervision
        self.use_multiscale_agg = use_multiscale_agg
        self.use_smart_skip = use_smart_skip
        
        # 1. EfficientNet-B4 encoder with channel adapters
        # Target dims follow Swin-Tiny progression used by the decoder
        target_dims = [96, 192, 384, 768]
        self.encoder = EfficientNetEncoderWithAdapters(
            target_dims=target_dims, 
            pretrained=pretrained
        )
        
        # 2. Enhanced Swin-Unet decoder with TransUNet best practices
        self.decoder = SwinDecoder(
            num_classes=num_classes, 
            img_size=img_size, 
            embed_dim=96,
            use_deep_supervision=use_deep_supervision,
            use_multiscale_agg=use_multiscale_agg,
            use_smart_skip=use_smart_skip
        )
        
        print(f"Hybrid1 model initialized:")
        print(f"  - Encoder: EfficientNet-B4 with skip/bottleneck adapters")
        print(f"  - Decoder: Swin-Unet with BOTTLENECK LAYER (2 SwinBlocks)")
        print(f"  - Segmentation Head: Conv3x3 + ReLU + Conv1x1 (REFERENCE COMPLIANT)")
        if use_deep_supervision:
            print(f"  - ✅ Deep Supervision: ENABLED (3 auxiliary outputs)")
        if use_multiscale_agg:
            print(f"  - ✅ Multi-Scale Aggregation: ENABLED (bottleneck)")
        if use_smart_skip:
            print(f"  - ✅ Smart Skip Connections: ENABLED (attention-based)")
        else:
            print(f"  - ✅ Skip Connections: BASELINE (naive concatenation)")
        print(f"  - Input size: {img_size}x{img_size}")
        print(f"  - Output classes: {num_classes}")
    
    @staticmethod
    def _to_tokens(feat: torch.Tensor, out_dim: int) -> torch.Tensor:
        """
        Convert a feature map (B, C, H, W) to token sequence (B, H*W, out_dim).
        
        Args:
            feat: Feature map tensor of shape (B, C, H, W)
            out_dim: Expected output dimension (should match C)
            
        Returns:
            Token sequence of shape (B, H*W, out_dim)
        """
        b, c, h, w = feat.shape
        x = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        assert c == out_dim, f"Adapter mismatch: got C={c}, expected {out_dim}"
        return x
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            If use_deep_supervision:
                (main_logits, [aux_out1, aux_out2, aux_out3])
            Else:
                main_logits (B, num_classes, H, W)
        """
        # Get 4 feature levels from EfficientNet encoder with adapters
        # feats: [P2 (stride 4), P3 (stride 8), P4 (stride 16), P5 (stride 32)]
        feats = self.encoder(x)
        assert len(feats) == 4, "Encoder did not return 4 feature levels"
        
        # Prepare tokens for decoder: bottom (P5) as current x, others as skip list
        p2, p3, p4, p5 = feats  # strides 4, 8, 16, 32
        
        # Convert to token sequences (B, L, C)
        target_dims = self.encoder.get_target_dims()
        x_tokens = self._to_tokens(p5, out_dim=target_dims[3])  # (B, 7*7, 768) for 224 input
        
        x_downsample = [
            self._to_tokens(p2, out_dim=target_dims[0]),  # (B, 56*56, 96)
            self._to_tokens(p3, out_dim=target_dims[1]),  # (B, 28*28, 192)
            self._to_tokens(p4, out_dim=target_dims[2]),  # (B, 14*14, 384)
            self._to_tokens(p5, out_dim=target_dims[3])   # (B, 7*7, 768)
        ]
        
        # ✅ BOTTLENECK PROCESSING: Process deepest features through 2 SwinBlocks
        # Optionally with multi-scale aggregation
        if self.use_multiscale_agg:
            x_tokens = self.decoder.forward_bottleneck(x_tokens, all_features=x_downsample)
        else:
            x_tokens = self.decoder.forward_bottleneck(x_tokens)
        
        # Decode using Swin-Unet decoder path
        if self.use_deep_supervision:
            x_up, aux_features = self.decoder.forward_up_features(x_tokens, x_downsample)
        else:
            x_up = self.decoder.forward_up_features(x_tokens, x_downsample)
        
        # Main output
        logits = self.decoder.up_x4(x_up)
        
        # Process auxiliary outputs for deep supervision
        if self.use_deep_supervision:
            aux_outputs = self.decoder.process_aux_outputs(aux_features)
            return logits, aux_outputs
        
        return logits
    
    def get_model_info(self) -> dict:
        """Get model information for debugging and analysis."""
        return {
            'model_type': 'HybridEfficientNetB4SwinDecoder',
            'encoder': 'EfficientNet-B4',
            'decoder': 'Swin-Unet',
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'target_dims': self.encoder.get_target_dims(),
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# ============================================================================
# MODEL FACTORY FUNCTION
# ============================================================================

def create_hybrid_model(num_classes: int = 6, img_size: int = 224, pretrained: bool = True,
                        use_deep_supervision: bool = False, use_multiscale_agg: bool = False,
                        use_smart_skip: bool = False) -> HybridEfficientNetB4SwinDecoder:
    """
    Factory function to create a hybrid EfficientNet-Swin model.
    
    Args:
        num_classes: Number of segmentation classes (4, 5, or 6)
        img_size: Input image size
        pretrained: Whether to use pretrained EfficientNet weights
        use_deep_supervision: Enable deep supervision (3 auxiliary outputs)
        use_multiscale_agg: Enable multi-scale aggregation in bottleneck
        use_smart_skip: Enable smart skip connections (attention-based fusion)
        
    Returns:
        Initialized hybrid model
    """
    model = HybridEfficientNetB4SwinDecoder(
        num_classes=num_classes,
        img_size=img_size,
        pretrained=pretrained,
        use_deep_supervision=use_deep_supervision,
        use_multiscale_agg=use_multiscale_agg,
        use_smart_skip=use_smart_skip
    )
    
    return model


def create_enhanced_hybrid1(num_classes: int = 6, img_size: int = 224, pretrained: bool = True,
                           use_smart_skip: bool = False) -> HybridEfficientNetB4SwinDecoder:
    """
    Create Enhanced Hybrid1 with all TransUNet best practices.
    
    Expected Performance: IoU 0.50-0.55 (vs baseline 0.40, +25-38%)
    
    Args:
        num_classes: Number of segmentation classes
        img_size: Input image size
        pretrained: Whether to use pretrained EfficientNet weights
        use_smart_skip: Enable smart skip connections (optional enhancement)
    
    Returns:
        Enhanced Hybrid1 model with deep supervision and multi-scale aggregation
    """
    return create_hybrid_model(
        num_classes=num_classes,
        img_size=img_size,
        pretrained=pretrained,
        use_deep_supervision=True,
        use_multiscale_agg=True,
        use_smart_skip=use_smart_skip
    )


# ============================================================================
# CONVENIENCE ALIASES
# ============================================================================

# Main model class
HybridModel = HybridEfficientNetB4SwinDecoder

# Alternative names for compatibility
EfficientNetSwinUnet = HybridEfficientNetB4SwinDecoder
CNNTransformerHybrid = HybridEfficientNetB4SwinDecoder


# ============================================================================
# CONVENIENCE FACTORY FUNCTIONS FOR SPECIFIC CLASS COUNTS
# ============================================================================

def create_hybrid_4class(img_size: int = 224, pretrained: bool = True) -> HybridEfficientNetB4SwinDecoder:
    """Create hybrid model with 4 classes."""
    return create_hybrid_model(num_classes=4, img_size=img_size, pretrained=pretrained)


def create_hybrid_5class(img_size: int = 224, pretrained: bool = True) -> HybridEfficientNetB4SwinDecoder:
    """Create hybrid model with 5 classes."""
    return create_hybrid_model(num_classes=5, img_size=img_size, pretrained=pretrained)


def create_hybrid_6class(img_size: int = 224, pretrained: bool = True) -> HybridEfficientNetB4SwinDecoder:
    """Create hybrid model with 6 classes."""
    return create_hybrid_model(num_classes=6, img_size=img_size, pretrained=pretrained)

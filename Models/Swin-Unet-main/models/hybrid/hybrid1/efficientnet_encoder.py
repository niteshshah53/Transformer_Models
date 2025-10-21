import torch
import torch.nn as nn
import timm


# ============================================================================
# EFFICIENTNET ENCODER
# ============================================================================

class Conv1x1BNAct(nn.Module):
    """1x1 Convolution with BatchNorm and GELU activation for channel adaptation."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-B4 encoder that extracts multi-scale features for segmentation.
    
    This encoder uses EfficientNet-B4 as backbone and provides 4 feature levels
    at different scales (strides 4, 8, 16, 32) suitable for U-Net style decoders.
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Build EfficientNet-B4 backbone with features_only=True
        # This returns feature maps at 4 scales: strides 4, 8, 16, 32
        self.backbone = timm.create_model(
            'tf_efficientnet_b4_ns', 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=(1, 2, 3, 4)  # Get features at indices 1,2,3,4 (strides 4,8,16,32)
        )
        
        # Get channel dimensions from the backbone
        self.channels = self.backbone.feature_info.channels()
        assert len(self.channels) == 4, f"Expected 4 feature levels, got {self.channels}"
        
        # Print feature info for debugging
        print(f"EfficientNet-B4 feature channels: {self.channels}")
        print(f"Feature strides: {[info['reduction'] for info in self.backbone.feature_info.get_dicts()]}")
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass through EfficientNet encoder.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            List of 4 feature maps at different scales:
            - feat[0]: (B, C1, H/4, W/4)   - stride 4
            - feat[1]: (B, C2, H/8, W/8)   - stride 8  
            - feat[2]: (B, C3, H/16, W/16) - stride 16
            - feat[3]: (B, C4, H/32, W/32) - stride 32
        """
        features = self.backbone(x)
        return features
    
    def get_channels(self) -> list[int]:
        """Get the number of channels for each feature level."""
        return self.channels
    
    def get_strides(self) -> list[int]:
        """Get the stride for each feature level."""
        return [info['reduction'] for info in self.backbone.feature_info.get_dicts()]


class EfficientNetEncoderWithAdapters(nn.Module):
    """
    EfficientNet encoder with bottleneck adapter for Swin decoder compatibility.
    
    REFERENCE ARCHITECTURE COMPLIANCE:
    - Returns RAW encoder features C1, C2, C3 (for skip connections)
    - Only adapts C4 (deepest feature) via 1x1 conv to embed_dim*8 (768 for embed_dim=96)
    - This matches: "1×1 Conv (from C4_channels → embed_dim*8)"
    """
    
    def __init__(self, target_dims: list[int] = [96, 192, 384, 768], pretrained: bool = True):
        super().__init__()
        
        # Build EfficientNet encoder
        self.encoder = EfficientNetEncoder(pretrained=pretrained)
        
        # Get source channels from EfficientNet
        source_channels = self.encoder.get_channels()
        # For EfficientNet-B4: [24, 32, 56, 160] at strides [4, 8, 16, 32]
        
        # REFERENCE COMPLIANCE: Only adapt C4 (bottleneck) to decoder embedding dimension
        # Skip connections use encoder features with adapter to match decoder dimensions
        self.skip_adapters = nn.ModuleList([
            Conv1x1BNAct(in_ch=source_channels[i], out_ch=target_dims[i]) 
            for i in range(3)  # Only for C1, C2, C3 (skip connections)
        ])
        
        # Bottleneck adapter: C4 (160 channels) → target_dims[3] (768 channels)
        self.bottleneck_adapter = Conv1x1BNAct(
            in_ch=source_channels[3],  # 160 for EfficientNet-B4
            out_ch=target_dims[3]      # 768 (embed_dim * 8)
        )
        
        self.target_dims = target_dims
        self.source_channels = source_channels
        
        print(f"✅ REFERENCE ARCHITECTURE MODE:")
        print(f"   EfficientNet channels: {source_channels}")
        print(f"   Skip adapters (C1-C3): {source_channels[:3]} → {target_dims[:3]}")
        print(f"   Bottleneck adapter (C4): {source_channels[3]} → {target_dims[3]}")
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass with bottleneck-only adaptation.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            List of 4 feature maps:
            - features[0-2]: Skip features (C1, C2, C3) - adapted to target dims
            - features[3]: Bottleneck feature (C4) - adapted to embed_dim*8
        """
        # Get features from EfficientNet
        features = self.encoder(x)
        # features[i]: (B, source_channels[i], H/(2^(i+2)), W/(2^(i+2)))
        
        # Adapt skip connection features (C1, C2, C3)
        adapted_features = [
            self.skip_adapters[i](features[i]) for i in range(3)
        ]
        
        # Adapt bottleneck feature (C4)
        adapted_features.append(self.bottleneck_adapter(features[3]))
        
        return adapted_features
    
    def get_target_dims(self) -> list[int]:
        """Get the target channel dimensions."""
        return self.target_dims

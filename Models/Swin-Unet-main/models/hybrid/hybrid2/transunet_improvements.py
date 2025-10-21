"""
TransUNet Best Practices: Advanced Improvements for Hybrid2
Implements Tier 2 and Tier 3 features:
- GroupNorm (better than BatchNorm for small batches)
- 2D Positional Embeddings
- Multi-Scale Aggregation
- Cross-Attention Bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================================
# TIER 2.1: GroupNorm Wrappers
# ============================================================================

def get_norm_layer(channels, norm_type='group', num_groups=32):
    """
    Factory function for normalization layers.
    TransUNet practice: Use GroupNorm for better small-batch performance.
    
    Args:
        channels: Number of channels
        norm_type: 'batch' or 'group'
        num_groups: Number of groups for GroupNorm (default: 32)
    
    Returns:
        Normalization layer
    """
    if norm_type == 'group':
        # Ensure num_groups divides channels
        num_groups = min(num_groups, channels)
        while channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.BatchNorm2d(channels)


class ImprovedChannelAttention(nn.Module):
    """Channel Attention with GroupNorm."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class ImprovedSpatialAttention(nn.Module):
    """Spatial Attention Module."""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class ImprovedCBAM(nn.Module):
    """CBAM with GroupNorm support."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ImprovedChannelAttention(channels, reduction)
        self.spatial_attention = ImprovedSpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ImprovedSmartSkipConnection(nn.Module):
    """
    Smart Skip Connection with GroupNorm and positional embeddings.
    TransUNet best practice: Use GroupNorm for stability.
    """
    
    def __init__(self, encoder_channels, decoder_channels, fusion_type='concat', use_pos_embed=True):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.use_pos_embed = use_pos_embed
        
        # Align encoder features with GroupNorm
        self.align = nn.Sequential(
            nn.Conv2d(encoder_channels, decoder_channels, 1, bias=False),
            get_norm_layer(decoder_channels, 'group'),
            nn.ReLU(inplace=True)
        )
        
        # Attention for skip features
        self.attention = ImprovedCBAM(decoder_channels)
        
        if fusion_type == 'add':
            self.fuse = None
        elif fusion_type == 'concat':
            self.fuse = nn.Sequential(
                nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1, bias=False),
                get_norm_layer(decoder_channels, 'group'),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, encoder_feat, decoder_feat):
        # Align and enhance encoder features
        skip_feat = self.align(encoder_feat)
        skip_feat = self.attention(skip_feat)
        
        if self.fusion_type == 'add':
            return decoder_feat + skip_feat
        elif self.fusion_type == 'concat':
            fused = torch.cat([decoder_feat, skip_feat], dim=1)
            return self.fuse(fused)
        else:
            return skip_feat


# ============================================================================
# TIER 2.2: 2D Positional Embeddings
# ============================================================================

class PositionalEmbedding2D(nn.Module):
    """
    2D Learnable Positional Embeddings.
    TransUNet practice: Help decoder understand spatial structure.
    """
    
    def __init__(self, channels, height, width):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, channels, height, width) * 0.02)
        self.channels = channels
        self.height = height
        self.width = width
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Handle size mismatches
        if H != self.height or W != self.width:
            pos_embed = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        else:
            pos_embed = self.pos_embed
        
        return x + pos_embed


class SinusoidalPositionalEmbedding2D(nn.Module):
    """
    Sinusoidal 2D Positional Embeddings (no learnable parameters).
    Alternative to learned embeddings.
    """
    
    def __init__(self, channels, temperature=10000):
        super().__init__()
        self.channels = channels
        self.temperature = temperature
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Create coordinate grid
        y_embed = torch.arange(H, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device).unsqueeze(0).repeat(H, 1)
        
        # Normalize to [0, 1]
        y_embed = y_embed / H
        x_embed = x_embed / W
        
        # Create positional embeddings
        dim_t = torch.arange(C // 2, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (C // 2))
        
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        
        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1).unsqueeze(0)
        
        return x + pos


# ============================================================================
# TIER 2.3: Multi-Scale Aggregation
# ============================================================================

class MultiScaleAggregation(nn.Module):
    """
    Multi-Scale Feature Aggregation.
    TransUNet practice: Allow decoder to use multiple encoder scales simultaneously.
    """
    
    def __init__(self, encoder_channels_list, out_channels, target_size=None):
        super().__init__()
        self.target_size = target_size
        
        # Projection layers for each scale
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, 1, bias=False),
                get_norm_layer(out_channels, 'group'),
                nn.ReLU(inplace=True)
            )
            for ch in encoder_channels_list
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(encoder_channels_list), out_channels, 1, bias=False),
            get_norm_layer(out_channels, 'group'),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features_list):
        """
        Args:
            features_list: List of features at different scales [F1, F2, F3, F4]
        
        Returns:
            Aggregated features
        """
        if self.target_size is None:
            # Use the largest feature map size
            self.target_size = features_list[0].shape[2:]
        
        # Project and resize all features to target size
        projected = []
        for feat, proj in zip(features_list, self.projections):
            feat_proj = proj(feat)
            if feat_proj.shape[2:] != self.target_size:
                feat_proj = F.interpolate(feat_proj, size=self.target_size, mode='bilinear', align_corners=False)
            projected.append(feat_proj)
        
        # Concatenate and fuse
        concatenated = torch.cat(projected, dim=1)
        fused = self.fusion(concatenated)
        
        return fused


# ============================================================================
# TIER 3.1: Cross-Attention Bottleneck
# ============================================================================

class CrossAttentionBottleneck(nn.Module):
    """
    Cross-Attention Bottleneck for querying encoder tokens.
    TransUNet practice: Active feature querying for better context integration.
    """
    
    def __init__(self, decoder_dim, encoder_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        
        # Cross attention (decoder queries encoder)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            kdim=encoder_dim,
            vdim=encoder_dim,
            dropout=dropout,
            batch_first=False  # PyTorch expects (L, N, E)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(decoder_dim)
        self.norm2 = nn.LayerNorm(decoder_dim)
        
        # Feed-forward network
        mlp_hidden_dim = int(decoder_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(decoder_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, decoder_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, decoder_feat, encoder_tokens):
        """
        Args:
            decoder_feat: [B, C, H, W] - decoder features
            encoder_tokens: [B, N, D] - encoder tokens (from Swin)
        
        Returns:
            Enhanced decoder features [B, C, H, W]
        """
        B, C, H, W = decoder_feat.shape
        
        # Flatten decoder features to sequence: [B, C, H, W] -> [HW, B, C]
        decoder_seq = decoder_feat.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        
        # Reshape encoder tokens: [B, N, D] -> [N, B, D]
        encoder_seq = encoder_tokens.permute(1, 0, 2)  # [N, B, D]
        
        # Cross attention: decoder queries encoder
        attn_out, _ = self.cross_attn(
            query=decoder_seq,
            key=encoder_seq,
            value=encoder_seq
        )
        
        # Residual connection + norm
        decoder_seq = self.norm1(decoder_seq + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(decoder_seq)
        decoder_seq = self.norm2(decoder_seq + ffn_out)
        
        # Reshape back to spatial: [HW, B, C] -> [B, C, H, W]
        output = decoder_seq.permute(1, 2, 0).reshape(B, C, H, W)
        
        return output


class ImprovedDecoderBlockWithCrossAttn(nn.Module):
    """
    Decoder block with cross-attention, multi-scale aggregation, and positional embeddings.
    Combines all TransUNet best practices.
    """
    
    def __init__(self, in_channels, out_channels, encoder_channels_list=None, 
                 use_cross_attn=False, encoder_dim=None, use_pos_embed=True):
        super().__init__()
        
        self.use_cross_attn = use_cross_attn
        self.use_pos_embed = use_pos_embed
        
        # Main convolution blocks with GroupNorm
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            get_norm_layer(out_channels, 'group'),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            get_norm_layer(out_channels, 'group'),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # CBAM attention
        self.attention = ImprovedCBAM(out_channels)
        
        # Positional embedding
        if use_pos_embed:
            # Will be initialized dynamically based on input size
            self.pos_embed = None
        
        # Cross-attention bottleneck (optional)
        if use_cross_attn and encoder_dim is not None:
            self.cross_attn = CrossAttentionBottleneck(out_channels, encoder_dim)
        else:
            self.cross_attn = None
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x, encoder_tokens=None):
        identity = x
        
        # Main convolution path
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Residual connection
        out = out + self.residual(identity)
        
        # Attention
        out = self.attention(out)
        
        # Positional embedding (add spatial awareness)
        if self.use_pos_embed:
            if self.pos_embed is None or self.pos_embed.height != out.shape[2]:
                self.pos_embed = PositionalEmbedding2D(
                    out.shape[1], out.shape[2], out.shape[3]
                ).to(out.device)
            out = self.pos_embed(out)
        
        # Cross-attention (query encoder if available)
        if self.cross_attn is not None and encoder_tokens is not None:
            out = self.cross_attn(out, encoder_tokens)
        
        # Dropout
        out = self.dropout(out)
        
        return out


# ============================================================================
# ENHANCED EFFICIENTNET DECODER (CNN-based with TransUNet improvements)
# ============================================================================

class DeepDecoderBlock(nn.Module):
    """
    Deep CNN decoder block with residual connections and attention.
    Pure EfficientNet-style architecture with GroupNorm.
    """
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        
        # Main convolution blocks with GroupNorm
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            get_norm_layer(out_channels, 'group'),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            get_norm_layer(out_channels, 'group'),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # CBAM attention
        self.attention = ImprovedCBAM(out_channels)
        
        # Positional embedding (will be initialized dynamically)
        self.pos_embed = None
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x):
        identity = x
        
        # Convolution path
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Residual
        out = out + self.residual(identity)
        
        # Attention
        out = self.attention(out)
        
        # Positional embedding
        if self.pos_embed is None or self.pos_embed.height != out.shape[2]:
            self.pos_embed = PositionalEmbedding2D(
                out.shape[1], out.shape[2], out.shape[3]
            ).to(out.device)
        out = self.pos_embed(out)
        
        # Dropout
        out = self.dropout(out)
        
        return out


class EnhancedEfficientNetDecoder(nn.Module):
    """
    Enhanced EfficientNet-style CNN Decoder with ALL TransUNet best practices:
    - Deep Supervision (auxiliary outputs)
    - Cross-Attention Bottleneck (active encoder querying)
    - Multi-Scale Aggregation (combine all encoder scales)
    - GroupNorm (better for small batches)
    - 2D Positional Embeddings (spatial awareness)
    - CBAM Attention (channel + spatial)
    - Smart Skip Connections
    
    Architecture: Pure CNN (Conv + GroupNorm + ReLU) with attention mechanisms
    """
    
    def __init__(self, encoder_channels, num_classes=6, decoder_channels=[256, 128, 64, 32],
                 use_deep_supervision=True, use_cross_attn=True, use_multiscale_agg=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.encoder_channels = encoder_channels  # [96, 192, 384, 768] from Swin
        self.decoder_channels = decoder_channels  # [256, 128, 64, 32]
        self.use_deep_supervision = use_deep_supervision
        self.use_cross_attn = use_cross_attn
        self.use_multiscale_agg = use_multiscale_agg
        
        print("🚀 Enhanced EfficientNet Decoder Initialized:")
        print(f"  ✅ Pure CNN Architecture (EfficientNet-style)")
        print(f"  ✅ GroupNorm (better small-batch stability)")
        print(f"  ✅ CBAM Attention (channel + spatial)")
        print(f"  ✅ 2D Positional Embeddings")
        print(f"  ✅ Deep Supervision: {use_deep_supervision}")
        print(f"  ✅ Cross-Attention: {use_cross_attn}")
        print(f"  ✅ Multi-Scale Aggregation: {use_multiscale_agg}")
        
        # ==================================================================
        # FEATURE PROJECTION: Encoder -> Decoder channels
        # ==================================================================
        self.encoder_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(enc_ch, dec_ch, 1, bias=False),
                get_norm_layer(dec_ch, 'group'),
                nn.ReLU(inplace=True)
            )
            for enc_ch, dec_ch in zip(encoder_channels, decoder_channels)
        ])
        
        # ==================================================================
        # MULTI-SCALE AGGREGATION (Optional)
        # ==================================================================
        if use_multiscale_agg:
            self.multiscale_agg = MultiScaleAggregation(
                encoder_channels_list=encoder_channels,
                out_channels=decoder_channels[3],  # Bottleneck dimension
                target_size=None  # Will be set dynamically
            )
        
        # ==================================================================
        # BOTTLENECK with Cross-Attention
        # ==================================================================
        if use_cross_attn:
            self.bottleneck = ImprovedDecoderBlockWithCrossAttn(
                in_channels=decoder_channels[3],
                out_channels=decoder_channels[3],
                use_cross_attn=True,
                encoder_dim=encoder_channels[3],  # 768 from Swin
                use_pos_embed=True
            )
        else:
            self.bottleneck = DeepDecoderBlock(
                in_channels=decoder_channels[3],
                out_channels=decoder_channels[3]
            )
        
        # ==================================================================
        # DECODER STAGES (Pure CNN with Smart Skip Connections)
        # ==================================================================
        
        # Stage 1: H/32 -> H/16
        self.decoder1 = nn.Sequential(
            DeepDecoderBlock(decoder_channels[3], decoder_channels[2]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.skip1 = ImprovedSmartSkipConnection(
            encoder_channels=decoder_channels[2],
            decoder_channels=decoder_channels[2],
            fusion_type='concat'
        )
        
        # Stage 2: H/16 -> H/8
        self.decoder2 = nn.Sequential(
            DeepDecoderBlock(decoder_channels[2], decoder_channels[1]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.skip2 = ImprovedSmartSkipConnection(
            encoder_channels=decoder_channels[1],
            decoder_channels=decoder_channels[1],
            fusion_type='concat'
        )
        
        # Stage 3: H/8 -> H/4
        self.decoder3 = nn.Sequential(
            DeepDecoderBlock(decoder_channels[1], decoder_channels[0]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.skip3 = ImprovedSmartSkipConnection(
            encoder_channels=decoder_channels[0],
            decoder_channels=decoder_channels[0],
            fusion_type='concat'
        )
        
        # Stage 4: H/4 -> H (final)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(decoder_channels[0], 64, 3, padding=1, bias=False),
            get_norm_layer(64, 'group'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
        
        # ==================================================================
        # SEGMENTATION HEADS
        # ==================================================================
        
        # Main segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            get_norm_layer(64, 'group'),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1)
        )
        
        # Auxiliary heads for deep supervision
        if use_deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(decoder_channels[2], num_classes, 1),  # From stage 1
                nn.Conv2d(decoder_channels[1], num_classes, 1),  # From stage 2
                nn.Conv2d(decoder_channels[0], num_classes, 1),  # From stage 3
            ])
        
        print(f"  📊 Encoder channels: {encoder_channels}")
        print(f"  📊 Decoder channels: {decoder_channels}")
        print(f"  📊 Output classes: {num_classes}")
    
    def forward(self, encoder_features, encoder_tokens=None):
        """
        Args:
            encoder_features: List of 4 feature maps from encoder
                              [F1 (H/4), F2 (H/8), F3 (H/16), F4 (H/32)]
            encoder_tokens: Optional encoder tokens for cross-attention [B, N, D]
        
        Returns:
            If use_deep_supervision:
                (main_output, [aux_out1, aux_out2, aux_out3])
            Else:
                main_output
        """
        # Unpack encoder features
        f1, f2, f3, f4 = encoder_features
        
        # ==================================================================
        # PROJECT ENCODER FEATURES
        # ==================================================================
        p1 = self.encoder_projections[0](f1)  # H/4
        p2 = self.encoder_projections[1](f2)  # H/8
        p3 = self.encoder_projections[2](f3)  # H/16
        p4 = self.encoder_projections[3](f4)  # H/32
        
        # ==================================================================
        # MULTI-SCALE AGGREGATION (Optional)
        # ==================================================================
        if self.use_multiscale_agg:
            bottleneck_feat = self.multiscale_agg([f1, f2, f3, f4])
            if bottleneck_feat.shape[2:] != p4.shape[2:]:
                bottleneck_feat = F.interpolate(bottleneck_feat, size=p4.shape[2:], 
                                                mode='bilinear', align_corners=False)
            bottleneck_input = bottleneck_feat + p4
        else:
            bottleneck_input = p4
        
        # ==================================================================
        # BOTTLENECK with Cross-Attention
        # ==================================================================
        if self.use_cross_attn and encoder_tokens is not None:
            x = self.bottleneck(bottleneck_input, encoder_tokens)
        else:
            x = self.bottleneck(bottleneck_input)
        
        # ==================================================================
        # DECODER STAGES with Skip Connections
        # ==================================================================
        
        aux_outputs = [] if self.use_deep_supervision else None
        
        # Stage 1: H/32 -> H/16
        x = self.decoder1(x)
        x = self.skip1(p3, x)
        if self.use_deep_supervision:
            aux_out1 = self.aux_heads[0](x)
            aux_out1 = F.interpolate(aux_out1, scale_factor=16, mode='bilinear', align_corners=False)
            aux_outputs.append(aux_out1)
        
        # Stage 2: H/16 -> H/8
        x = self.decoder2(x)
        x = self.skip2(p2, x)
        if self.use_deep_supervision:
            aux_out2 = self.aux_heads[1](x)
            aux_out2 = F.interpolate(aux_out2, scale_factor=8, mode='bilinear', align_corners=False)
            aux_outputs.append(aux_out2)
        
        # Stage 3: H/8 -> H/4
        x = self.decoder3(x)
        x = self.skip3(p1, x)
        if self.use_deep_supervision:
            aux_out3 = self.aux_heads[2](x)
            aux_out3 = F.interpolate(aux_out3, scale_factor=4, mode='bilinear', align_corners=False)
            aux_outputs.append(aux_out3)
        
        # Stage 4: H/4 -> H (final)
        x = self.decoder4(x)
        
        # ==================================================================
        # SEGMENTATION HEAD
        # ==================================================================
        main_output = self.seg_head(x)
        
        if self.use_deep_supervision:
            return main_output, aux_outputs
        else:
            return main_output
    
    def get_model_info(self):
        """Get model information."""
        return {
            'decoder_type': 'EnhancedEfficientNet',
            'encoder_channels': self.encoder_channels,
            'decoder_channels': self.decoder_channels,
            'num_classes': self.num_classes,
            'use_deep_supervision': self.use_deep_supervision,
            'use_cross_attn': self.use_cross_attn,
            'use_multiscale_agg': self.use_multiscale_agg,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def create_enhanced_efficientnet_decoder(encoder_channels, num_classes=6,
                                         use_deep_supervision=True,
                                         use_cross_attn=True,
                                         use_multiscale_agg=True):
    """
    Factory function to create Enhanced EfficientNet Decoder.
    
    Args:
        encoder_channels: List of encoder channel dimensions [96, 192, 384, 768]
        num_classes: Number of segmentation classes
        use_deep_supervision: Enable auxiliary outputs
        use_cross_attn: Enable cross-attention bottleneck
        use_multiscale_agg: Enable multi-scale aggregation
    
    Returns:
        EnhancedEfficientNetDecoder instance
    """
    return EnhancedEfficientNetDecoder(
        encoder_channels=encoder_channels,
        num_classes=num_classes,
        decoder_channels=[256, 128, 64, 32],
        use_deep_supervision=use_deep_supervision,
        use_cross_attn=use_cross_attn,
        use_multiscale_agg=use_multiscale_agg
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_module_info(module, name="Module"):
    """Print information about a module."""
    total_params = count_parameters(module)
    print(f"{name}: {total_params:,} parameters")
    return total_params


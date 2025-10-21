"""
TransUNet-Enhanced Decoder for Hybrid2
Implements ALL TransUNet best practices:
- GroupNorm instead of BatchNorm
- 2D Positional Embeddings
- Multi-Scale Aggregation
- Cross-Attention Bottleneck
- Deep Supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transunet_improvements import (
    get_norm_layer,
    ImprovedSmartSkipConnection,
    PositionalEmbedding2D,
    MultiScaleAggregation,
    CrossAttentionBottleneck,
    ImprovedDecoderBlockWithCrossAttn,
    ImprovedCBAM
)


class TransUNetEnhancedDecoder(nn.Module):
    """
    Fully enhanced decoder with ALL TransUNet best practices.
    
    Improvements over original:
    - GroupNorm (better for small batches)
    - 2D Positional Embeddings (spatial awareness)
    - Multi-Scale Aggregation (use multiple encoder scales)
    - Cross-Attention Bottleneck (active encoder querying)
    - Deep Supervision (auxiliary outputs)
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
        
        print("🚀 TransUNet-Enhanced Decoder Initialized:")
        print(f"  ✅ GroupNorm (better small-batch stability)")
        print(f"  ✅ 2D Positional Embeddings (spatial awareness)")
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
        self.bottleneck = ImprovedDecoderBlockWithCrossAttn(
            in_channels=decoder_channels[3],
            out_channels=decoder_channels[3],
            use_cross_attn=use_cross_attn,
            encoder_dim=encoder_channels[3],  # 768 from Swin
            use_pos_embed=True
        )
        
        # ==================================================================
        # DECODER STAGES with Smart Skip Connections
        # ==================================================================
        
        # Stage 1: H/32 -> H/16 (decoder_channels[3]=32 -> decoder_channels[2]=64)
        self.decoder1 = nn.Sequential(
            ImprovedDecoderBlockWithCrossAttn(
                in_channels=decoder_channels[3],  # 32
                out_channels=decoder_channels[2],  # 64
                use_cross_attn=False,  # Only bottleneck uses cross-attn
                use_pos_embed=True
            ),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.skip1 = ImprovedSmartSkipConnection(
            encoder_channels=decoder_channels[2],  # p3 has 64 channels (already projected)
            decoder_channels=decoder_channels[2],  # decoder1 output has 64 channels
            fusion_type='concat'
        )
        
        # Stage 2: H/16 -> H/8 (decoder_channels[2]=64 -> decoder_channels[1]=128)
        self.decoder2 = nn.Sequential(
            ImprovedDecoderBlockWithCrossAttn(
                in_channels=decoder_channels[2],  # 64 (after skip fusion)
                out_channels=decoder_channels[1],  # 128
                use_cross_attn=False,
                use_pos_embed=True
            ),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.skip2 = ImprovedSmartSkipConnection(
            encoder_channels=decoder_channels[1],  # p2 has 128 channels
            decoder_channels=decoder_channels[1],  # decoder2 output has 128 channels
            fusion_type='concat'
        )
        
        # Stage 3: H/8 -> H/4 (decoder_channels[1]=128 -> decoder_channels[0]=256)
        self.decoder3 = nn.Sequential(
            ImprovedDecoderBlockWithCrossAttn(
                in_channels=decoder_channels[1],  # 128 (after skip fusion)
                out_channels=decoder_channels[0],  # 256
                use_cross_attn=False,
                use_pos_embed=True
            ),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.skip3 = ImprovedSmartSkipConnection(
            encoder_channels=decoder_channels[0],  # p1 has 256 channels
            decoder_channels=decoder_channels[0],  # decoder3 output has 256 channels
            fusion_type='concat'
        )
        
        # Stage 4: H/4 -> H (final upsampling)
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
        # Unpack encoder features (stride 4, 8, 16, 32)
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
            # Aggregate all scales at bottleneck
            bottleneck_feat = self.multiscale_agg([f1, f2, f3, f4])
            # Resize to match p4
            if bottleneck_feat.shape[2:] != p4.shape[2:]:
                bottleneck_feat = F.interpolate(bottleneck_feat, size=p4.shape[2:], 
                                                mode='bilinear', align_corners=False)
            # Combine with p4
            bottleneck_input = bottleneck_feat + p4
        else:
            bottleneck_input = p4
        
        # ==================================================================
        # BOTTLENECK with Cross-Attention
        # ==================================================================
        x = self.bottleneck(bottleneck_input, encoder_tokens)
        
        # ==================================================================
        # DECODER STAGES with Skip Connections
        # ==================================================================
        
        aux_outputs = [] if self.use_deep_supervision else None
        
        # Stage 1: H/32 -> H/16
        x = self.decoder1(x)  # Upsample first
        x = self.skip1(p3, x)  # Then apply skip connection (now same spatial size)
        if self.use_deep_supervision:
            aux_out1 = self.aux_heads[0](x)
            aux_out1 = F.interpolate(aux_out1, scale_factor=16, mode='bilinear', align_corners=False)
            aux_outputs.append(aux_out1)
        
        # Stage 2: H/16 -> H/8
        x = self.decoder2(x)  # Upsample first
        x = self.skip2(p2, x)  # Then apply skip connection
        if self.use_deep_supervision:
            aux_out2 = self.aux_heads[1](x)
            aux_out2 = F.interpolate(aux_out2, scale_factor=8, mode='bilinear', align_corners=False)
            aux_outputs.append(aux_out2)
        
        # Stage 3: H/8 -> H/4
        x = self.decoder3(x)  # Upsample first
        x = self.skip3(p1, x)  # Then apply skip connection
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
            'decoder_type': 'TransUNetEnhanced',
            'encoder_channels': self.encoder_channels,
            'decoder_channels': self.decoder_channels,
            'num_classes': self.num_classes,
            'use_deep_supervision': self.use_deep_supervision,
            'use_cross_attn': self.use_cross_attn,
            'use_multiscale_agg': self.use_multiscale_agg,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def create_transunet_enhanced_decoder(encoder_channels, num_classes=6, 
                                      use_deep_supervision=True, 
                                      use_cross_attn=True,
                                      use_multiscale_agg=True):
    """
    Factory function to create TransUNet-enhanced decoder.
    
    Args:
        encoder_channels: List of encoder channel dimensions [96, 192, 384, 768]
        num_classes: Number of segmentation classes
        use_deep_supervision: Enable auxiliary outputs
        use_cross_attn: Enable cross-attention bottleneck
        use_multiscale_agg: Enable multi-scale aggregation
    
    Returns:
        TransUNetEnhancedDecoder instance
    """
    return TransUNetEnhancedDecoder(
        encoder_channels=encoder_channels,
        num_classes=num_classes,
        decoder_channels=[256, 128, 64, 32],
        use_deep_supervision=use_deep_supervision,
        use_cross_attn=use_cross_attn,
        use_multiscale_agg=use_multiscale_agg
    )


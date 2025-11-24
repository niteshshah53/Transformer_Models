"""
Hybrid2 Components: All building blocks for Hybrid2 models
Contains: Swin Encoder, Improvements (CBAM, CrossAttention, etc.), Decoders (Baseline & Enhanced)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm


# ============================================================================
# SWIN TRANSFORMER COMPONENTS (Encoder)
# ============================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """Partition input into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition to reconstruct feature map."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self attention (W-MSA) module with relative position bias."""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer."""
    
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinEncoder(nn.Module):
    """
    Swin Transformer encoder that extracts multi-scale features for segmentation.
    
    Provides 4 feature levels at different scales (strides 4, 8, 16, 32) suitable for CNN decoders.
    """
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward pass through Swin encoder."""
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        features = []
        for layer in self.layers:
            features.append(x)
            x = layer(x)
        
        # Convert token sequences back to feature maps
        feature_maps = []
        for i, feat in enumerate(features):
            B, L, C = feat.shape
            H = W = int(L ** 0.5)
            feat_map = feat.transpose(1, 2).view(B, C, H, W)
            feature_maps.append(feat_map)
        
        return feature_maps
    
    def get_channels(self):
        """Get the number of channels for each feature level."""
        return [int(self.embed_dim * 2 ** i) for i in range(self.num_layers)]
    
    def get_strides(self):
        """Get the stride for each feature level."""
        return [4 * (2 ** i) for i in range(self.num_layers)]


# ============================================================================
# IMPROVEMENTS & ATTENTION MECHANISMS (CBAM, CrossAttention, etc.)
# ============================================================================

def get_norm_layer(channels, norm_type='group', num_groups=32):
    """Factory function for normalization layers."""
    if norm_type == 'group':
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
    """CBAM (Convolutional Block Attention Module) with GroupNorm support."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ImprovedChannelAttention(channels, reduction)
        self.spatial_attention = ImprovedSpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ImprovedSmartSkipConnection(nn.Module):
    """Smart Skip Connection with GroupNorm and CBAM attention."""
    
    def __init__(self, encoder_channels, decoder_channels, fusion_type='concat', use_pos_embed=True):
        super().__init__()
        self.fusion_type = fusion_type
        self.use_pos_embed = use_pos_embed
        
        self.align = nn.Sequential(
            nn.Conv2d(encoder_channels, decoder_channels, 1, bias=False),
            get_norm_layer(decoder_channels, 'group'),
            nn.ReLU(inplace=True)
        )
        
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
        skip_feat = self.align(encoder_feat)
        
        # Upsample encoder features to match decoder spatial size
        if skip_feat.shape[2:] != decoder_feat.shape[2:]:
            skip_feat = F.interpolate(skip_feat, size=decoder_feat.shape[2:], 
                                    mode='bilinear', align_corners=False)
        
        skip_feat = self.attention(skip_feat)
        
        if self.fusion_type == 'add':
            return decoder_feat + skip_feat
        elif self.fusion_type == 'concat':
            fused = torch.cat([decoder_feat, skip_feat], dim=1)
            return self.fuse(fused)
        else:
            return skip_feat


class PositionalEmbedding2D(nn.Module):
    """2D Learnable Positional Embeddings."""
    
    def __init__(self, channels, height, width):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, channels, height, width) * 0.02)
        self.channels = channels
        self.height = height
        self.width = width
    
    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.height or W != self.width:
            pos_embed = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        else:
            pos_embed = self.pos_embed
        return x + pos_embed


class MultiScaleAggregation(nn.Module):
    """Multi-Scale Feature Aggregation."""
    
    def __init__(self, encoder_channels_list, out_channels, target_size=None):
        super().__init__()
        self.target_size = target_size
        
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, 1, bias=False),
                get_norm_layer(out_channels, 'group'),
                nn.ReLU(inplace=True)
            )
            for ch in encoder_channels_list
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(encoder_channels_list), out_channels, 1, bias=False),
            get_norm_layer(out_channels, 'group'),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features_list):
        if self.target_size is None:
            self.target_size = features_list[0].shape[2:]
        
        projected = []
        for feat, proj in zip(features_list, self.projections):
            feat_proj = proj(feat)
            if feat_proj.shape[2:] != self.target_size:
                feat_proj = F.interpolate(feat_proj, size=self.target_size, mode='bilinear', align_corners=False)
            projected.append(feat_proj)
        
        concatenated = torch.cat(projected, dim=1)
        fused = self.fusion(concatenated)
        return fused


class CrossAttentionBottleneck(nn.Module):
    """Cross-Attention Bottleneck for querying encoder tokens."""
    
    def __init__(self, decoder_dim, encoder_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            kdim=encoder_dim,
            vdim=encoder_dim,
            dropout=dropout,
            batch_first=False
        )
        
        self.norm1 = nn.LayerNorm(decoder_dim)
        self.norm2 = nn.LayerNorm(decoder_dim)
        
        mlp_hidden_dim = int(decoder_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(decoder_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, decoder_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, decoder_feat, encoder_tokens):
        B, C, H, W = decoder_feat.shape
        decoder_seq = decoder_feat.flatten(2).permute(2, 0, 1)
        encoder_seq = encoder_tokens.permute(1, 0, 2)
        
        attn_out, _ = self.cross_attn(query=decoder_seq, key=encoder_seq, value=encoder_seq)
        decoder_seq = self.norm1(decoder_seq + attn_out)
        ffn_out = self.ffn(decoder_seq)
        decoder_seq = self.norm2(decoder_seq + ffn_out)
        
        output = decoder_seq.permute(1, 2, 0).reshape(B, C, H, W)
        return output


class ImprovedDecoderBlockWithCrossAttn(nn.Module):
    """Decoder block with cross-attention, multi-scale aggregation, and positional embeddings."""
    
    def __init__(self, in_channels, out_channels, encoder_channels_list=None, 
                 use_cross_attn=False, encoder_dim=None, use_pos_embed=True):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_pos_embed = use_pos_embed
        
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
        
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attention = ImprovedCBAM(out_channels)
        
        if use_pos_embed:
            self.pos_embed = None
        
        if use_cross_attn and encoder_dim is not None:
            self.cross_attn = CrossAttentionBottleneck(out_channels, encoder_dim)
        else:
            self.cross_attn = None
        
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x, encoder_tokens=None):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.residual(identity)
        out = self.attention(out)
        
        if self.use_pos_embed:
            if self.pos_embed is None or self.pos_embed.height != out.shape[2]:
                self.pos_embed = PositionalEmbedding2D(
                    out.shape[1], out.shape[2], out.shape[3]
                ).to(out.device)
            out = self.pos_embed(out)
        
        if self.cross_attn is not None and encoder_tokens is not None:
            out = self.cross_attn(out, encoder_tokens)
        
        out = self.dropout(out)
        return out


class DeepDecoderBlock(nn.Module):
    """Deep CNN decoder block with residual connections and attention."""
    
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        
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
        
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attention = ImprovedCBAM(out_channels)
        self.pos_embed = None
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.residual(identity)
        out = self.attention(out)
        
        if self.pos_embed is None or self.pos_embed.height != out.shape[2]:
            self.pos_embed = PositionalEmbedding2D(
                out.shape[1], out.shape[2], out.shape[3]
            ).to(out.device)
        out = self.pos_embed(out)
        out = self.dropout(out)
        return out


# ============================================================================
# BASELINE DECODER (Simple EfficientNet-style)
# ============================================================================

class SimpleSkipConnection(nn.Module):
    """Simple skip connection: Convert token features to CNN features and concatenate."""
    
    def __init__(self, encoder_channels, decoder_channels, use_groupnorm=True):
        super().__init__()
        norm_layer = get_norm_layer(decoder_channels, 'group' if use_groupnorm else 'batch')
        norm_layer_fuse = get_norm_layer(decoder_channels, 'group' if use_groupnorm else 'batch')
        
        self.proj = nn.Sequential(
            nn.Conv2d(encoder_channels, decoder_channels, 1, bias=False),
            norm_layer,
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1, bias=False),
            norm_layer_fuse,
            nn.ReLU(inplace=True)
        )
    
    def forward(self, encoder_feat, decoder_feat):
        encoder_proj = self.proj(encoder_feat)
        
        # Upsample encoder features to match decoder spatial size
        if encoder_proj.shape[2:] != decoder_feat.shape[2:]:
            encoder_proj = F.interpolate(encoder_proj, size=decoder_feat.shape[2:], 
                                       mode='bilinear', align_corners=False)
        
        fused = torch.cat([decoder_feat, encoder_proj], dim=1)
        fused = self.fuse(fused)
        return fused


class SimpleDecoderBlock(nn.Module):
    """Simple CNN decoder block with BatchNorm or GroupNorm support."""
    
    def __init__(self, in_channels, out_channels, use_batchnorm=True, use_groupnorm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        
        # Choose normalization layer
        if use_groupnorm:
            self.bn1 = get_norm_layer(out_channels, 'group')
        elif use_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.Identity()
            
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        # Choose normalization layer
        if use_groupnorm:
            self.bn2 = get_norm_layer(out_channels, 'group')
        elif use_batchnorm:
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.bn2 = nn.Identity()
            
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class BaselineHybrid2Decoder(nn.Module):
    """
    Baseline Hybrid2 Decoder:
    - Swin Bottleneck (2 Swin blocks)
    - EfficientNet-style CNN decoder
    - Simple skip connections
    - BatchNorm (not GroupNorm)
    - No attention mechanisms (optional via flags)
    
    All enhancements can be enabled via flags.
    """
    
    def __init__(self, encoder_channels, num_classes=6, 
                 efficientnet_variant='b4',
                 use_deep_supervision=False, use_cbam=False, 
                 use_smart_skip=False, use_cross_attn=False,
                 use_multiscale_agg=False, use_groupnorm=True,
                 use_pos_embed=True,  # Default True to match SwinUnet pattern
                 use_bottleneck_swin=True,
                 img_size=224):
        super().__init__()
        
        self.num_classes = num_classes
        self.encoder_channels = encoder_channels
        self.efficientnet_variant = efficientnet_variant
        self.use_deep_supervision = use_deep_supervision
        self.use_cbam = use_cbam
        self.use_smart_skip = use_smart_skip
        self.use_cross_attn = use_cross_attn
        self.use_multiscale_agg = use_multiscale_agg
        self.use_groupnorm = use_groupnorm
        self.use_pos_embed = use_pos_embed
        self.use_bottleneck_swin = use_bottleneck_swin
        
        if efficientnet_variant == 'b0':
            decoder_channels = [128, 64, 32, 16]
        elif efficientnet_variant == 'b4':
            decoder_channels = [256, 128, 64, 32]
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {efficientnet_variant}")
        
        self.decoder_channels = decoder_channels
        
        print("=" * 80)
        print("🚀 BASELINE Hybrid2 Decoder")
        print("=" * 80)
        print(f"  • EfficientNet Variant: {efficientnet_variant}")
        print(f"  • Decoder Channels: {decoder_channels}")
        print(f"  • Bottleneck Swin Blocks: {use_bottleneck_swin}")
        print(f"\n📊 Optional Features:")
        print(f"  • Deep Supervision: {use_deep_supervision}")
        print(f"  • CBAM Attention: {use_cbam}")
        print(f"  • Smart Skip Connections: {use_smart_skip}")
        print(f"  • Cross-Attention: {use_cross_attn}")
        print(f"  • Multi-Scale Aggregation: {use_multiscale_agg}")
        print(f"  • GroupNorm: {use_groupnorm} (else BatchNorm)")
        print(f"  • Positional Embeddings: {use_pos_embed}")
        print("=" * 80)
        
        # Feature projections
        self.encoder_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(enc_ch, dec_ch, 1, bias=False),
                self._get_norm_layer(dec_ch),
                nn.ReLU(inplace=True)
            )
            for enc_ch, dec_ch in zip(encoder_channels, decoder_channels)
        ])
        
        # Multi-scale aggregation (optional)
        if use_multiscale_agg:
            self.multiscale_agg = MultiScaleAggregation(
                encoder_channels_list=encoder_channels,
                out_channels=decoder_channels[3],
                target_size=None
            )
        else:
            self.multiscale_agg = None
        
        # Bottleneck: 2 Swin blocks (baseline)
        if use_bottleneck_swin:
            bottleneck_resolution = (img_size // 32, img_size // 32)
            # Use encoder dimension (768) instead of decoder dimension (32)
            encoder_dim = encoder_channels[3]  # 768
            bottleneck_dim = encoder_dim  # 768 (match encoder dimension)
            
            # Choose num_heads for 768 dimension (should be 24, matching encoder Stage 4)
            if bottleneck_dim == 768:
                bottleneck_num_heads = 24  # 768 / 24 = 32 per head (matches encoder Stage 4)
            elif bottleneck_dim % 24 == 0:
                bottleneck_num_heads = 24
            elif bottleneck_dim % 16 == 0:
                bottleneck_num_heads = 16
            elif bottleneck_dim % 8 == 0:
                bottleneck_num_heads = 8
            else:
                bottleneck_num_heads = 8  # Default fallback
            
            # No token projection needed - use encoder dimension directly
            self.token_projection = None
            
            # Feature map projection: project from bottleneck output (768) to decoder input (32)
            decoder_dim = decoder_channels[3]  # 32
            self.feature_projection = nn.Sequential(
                nn.Conv2d(encoder_dim, decoder_dim, 1, bias=False),
                self._get_norm_layer(decoder_dim),
                nn.ReLU(inplace=True)
            )
            
            # Projection layers for MSA/p4 to encoder dimension (for token processing)
            # Used when MSA is enabled or when encoder_tokens not available
            self.msa_to_encoder_proj = nn.Sequential(
                nn.Conv2d(decoder_dim, encoder_dim, 1, bias=False),
                self._get_norm_layer(encoder_dim),
                nn.ReLU(inplace=True)
            )
            self.p4_to_encoder_proj = self.msa_to_encoder_proj  # Same projection for p4
            
            self.bottleneck_layer = BasicLayer(
                dim=bottleneck_dim,  # 768
                input_resolution=bottleneck_resolution,
                depth=2,
                num_heads=bottleneck_num_heads,  # 24
                window_size=7,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False
            )
        else:
            self.bottleneck_layer = None
            self.token_projection = None
            self.feature_projection = None
        
        # Optional: Cross-Attention Bottleneck
        if use_cross_attn:
            self.cross_attn = CrossAttentionBottleneck(
                decoder_dim=decoder_channels[3],
                encoder_dim=encoder_channels[3],
                num_heads=8
            )
        else:
            self.cross_attn = None
        
        # Optional: CBAM for bottleneck
        if use_cbam:
            self.bottleneck_cbam = ImprovedCBAM(decoder_channels[3])
        else:
            self.bottleneck_cbam = None
        
        # Optional: Positional Embeddings
        if use_pos_embed:
            self.pos_embed = PositionalEmbedding2D(
                decoder_channels[3], img_size // 32, img_size // 32
            )
        else:
            self.pos_embed = None
        
        # Decoder stages
        self.decoder1 = nn.Sequential(
            self._create_decoder_block(decoder_channels[3], decoder_channels[2]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        if use_smart_skip:
            self.skip1 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[3],  # p4 (e4) → decoder1
                decoder_channels=decoder_channels[2],
                fusion_type='concat'
            )
        else:
            self.skip1 = SimpleSkipConnection(
                encoder_channels=decoder_channels[3],  # p4 (e4) → decoder1
                decoder_channels=decoder_channels[2],
                use_groupnorm=use_groupnorm
            )
        
        self.decoder2 = nn.Sequential(
            self._create_decoder_block(decoder_channels[2], decoder_channels[1]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        if use_smart_skip:
            self.skip2 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[2],  # p3 (e3) → decoder2
                decoder_channels=decoder_channels[1],
                fusion_type='concat'
            )
        else:
            self.skip2 = SimpleSkipConnection(
                encoder_channels=decoder_channels[2],  # p3 (e3) → decoder2
                decoder_channels=decoder_channels[1],
                use_groupnorm=use_groupnorm
            )
        
        self.decoder3 = nn.Sequential(
            self._create_decoder_block(decoder_channels[1], decoder_channels[0]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        if use_smart_skip:
            self.skip3 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[1],  # p2 (e2) → decoder3
                decoder_channels=decoder_channels[0],
                fusion_type='concat'
            )
        else:
            self.skip3 = SimpleSkipConnection(
                encoder_channels=decoder_channels[1],  # p2 (e2) → decoder3
                decoder_channels=decoder_channels[0],
                use_groupnorm=use_groupnorm
            )
        
        # Skip connection for decoder4: e1 (p1) → decoder4
        if use_smart_skip:
            self.skip4 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[0],  # p1 (e1) → decoder4
                decoder_channels=64,  # decoder4 output is 64 channels
                fusion_type='concat'
            )
        else:
            self.skip4 = SimpleSkipConnection(
                encoder_channels=decoder_channels[0],  # p1 (e1) → decoder4
                decoder_channels=64,  # decoder4 output is 64 channels
                use_groupnorm=use_groupnorm
            )
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(decoder_channels[0], 64, 3, padding=1, bias=False),
            self._get_norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            self._get_norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1)
        )
        
        # Auxiliary heads for deep supervision (MSAGHNet-style: simple OutConv, 3x3 conv)
        # Outputs at native resolutions (NO upsampling) - matching Network model multi-resolution approach
        if use_deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(decoder_channels[0], num_classes, 3, padding=1),  # Stage 1: H/16, 256 channels
                nn.Conv2d(decoder_channels[1], num_classes, 3, padding=1),  # Stage 2: H/8, 128 channels
                nn.Conv2d(decoder_channels[2], num_classes, 3, padding=1),  # Stage 3: H/4, 64 channels
            ])
        else:
            self.aux_heads = None
    
    def _get_norm_layer(self, channels):
        """Get normalization layer based on flag."""
        if self.use_groupnorm:
            return get_norm_layer(channels, 'group')
        else:
            return nn.BatchNorm2d(channels)
    
    def _create_decoder_block(self, in_channels, out_channels):
        """Create decoder block with optional CBAM."""
        block = SimpleDecoderBlock(
            in_channels, 
            out_channels, 
            use_batchnorm=not self.use_groupnorm,
            use_groupnorm=self.use_groupnorm
        )
        
        if self.use_cbam:
            cbam = ImprovedCBAM(out_channels)
            return nn.Sequential(block, cbam)
        else:
            return block
    
    def forward(self, encoder_features, encoder_tokens=None):
        f1, f2, f3, f4 = encoder_features
        
        p1 = self.encoder_projections[0](f1)
        p2 = self.encoder_projections[1](f2)
        p3 = self.encoder_projections[2](f3)
        p4 = self.encoder_projections[3](f4)
        
        if self.use_multiscale_agg:
            # If MSA enabled: bottleneck receives input ONLY from MSA output (not MSA + p4)
            bottleneck_feat = self.multiscale_agg([f1, f2, f3, f4])
            if bottleneck_feat.shape[2:] != p4.shape[2:]:
                bottleneck_feat = F.interpolate(bottleneck_feat, size=p4.shape[2:], 
                                              mode='bilinear', align_corners=False)
            x = bottleneck_feat  # MSA output ONLY (no addition with p4)
        else:
            # If MSA disabled: bottleneck receives input ONLY from e4 (p4)
            x = p4
        
        # Bottleneck: 2 Swin blocks
        # If MSA enabled: bottleneck receives input ONLY from MSA output (convert to tokens)
        # If MSA disabled: bottleneck receives input from e4 (use encoder_tokens if available, else p4)
        if self.use_bottleneck_swin:
            if self.use_multiscale_agg:
                # MSA enabled: use MSA output (x) - convert feature maps to tokens
                # x is MSA output at decoder_dim (32), need to project to encoder_dim (768) for token processing
                B, C, H, W = x.shape
                encoder_dim = self.encoder_channels[3]  # 768
                
                # Project MSA output to encoder dimension for bottleneck token processing
                if C != encoder_dim:
                    x = self.msa_to_encoder_proj(x)  # (B, 768, 7, 7)
                
                # Convert feature maps to tokens
                x_tokens = x.flatten(2).permute(0, 2, 1)  # (B, 49, 768)
                x_tokens = self.bottleneck_layer(x_tokens)  # (B, 49, 768)
                x = x_tokens.transpose(1, 2).view(B, encoder_dim, H, W)  # (B, 768, 7, 7)
                
                # Project from encoder dimension (768) to decoder dimension (32)
                x = self.feature_projection(x)  # (B, 32, 7, 7)
            elif encoder_tokens is not None:
                # MSA disabled AND encoder_tokens available: use encoder_tokens directly
                stage4_tokens = encoder_tokens  # (B, 49, 768)
                x_tokens = self.bottleneck_layer(stage4_tokens)  # (B, 49, 768)
                
                # Convert tokens back to feature maps
                B, L, C = x_tokens.shape
                H = W = int(L ** 0.5)  # H = W = 7
                x = x_tokens.transpose(1, 2).view(B, C, H, W)  # (B, 768, 7, 7)
                
                # Project from encoder dimension (768) to decoder dimension (32)
                x = self.feature_projection(x)  # (B, 32, 7, 7)
            else:
                # MSA disabled AND no encoder_tokens: use p4, convert to tokens
                B, C, H, W = x.shape  # x is p4 at decoder_dim (32)
                encoder_dim = self.encoder_channels[3]  # 768
                
                # Project p4 to encoder dimension for token processing
                if C != encoder_dim:
                    x = self.p4_to_encoder_proj(x)  # (B, 768, 7, 7)
                
                # Convert feature maps to tokens
                x_tokens = x.flatten(2).permute(0, 2, 1)  # (B, 49, 768)
                x_tokens = self.bottleneck_layer(x_tokens)  # (B, 49, 768)
                x = x_tokens.transpose(1, 2).view(B, encoder_dim, H, W)  # (B, 768, 7, 7)
                
                # Project from encoder dimension (768) to decoder dimension (32)
                x = self.feature_projection(x)  # (B, 32, 7, 7)
        
        if self.bottleneck_cbam is not None:
            x = self.bottleneck_cbam(x)
        
        if self.pos_embed is not None:
            x = self.pos_embed(x)
        
        if self.use_cross_attn and encoder_tokens is not None:
            x = self.cross_attn(x, encoder_tokens)
        
        aux_outputs = [] if self.use_deep_supervision else None
        
        # Stage 1: H/32 -> H/16
        x = self.decoder1(x)
        x = self.skip1(p4, x)  # e4 (p4) → decoder1
        if self.use_deep_supervision:
            # Output at native resolution (H/16) - NO upsampling (multi-resolution deep supervision)
            aux_out1 = self.aux_heads[0](x)  # [B, num_classes, H/16, W/16]
            aux_outputs.append(aux_out1)
        
        # Stage 2: H/16 -> H/8
        x = self.decoder2(x)
        x = self.skip2(p3, x)  # e3 (p3) → decoder2
        if self.use_deep_supervision:
            # Output at native resolution (H/8) - NO upsampling (multi-resolution deep supervision)
            aux_out2 = self.aux_heads[1](x)  # [B, num_classes, H/8, W/8]
            aux_outputs.append(aux_out2)
        
        # Stage 3: H/8 -> H/4
        x = self.decoder3(x)
        x = self.skip3(p2, x)  # e2 (p2) → decoder3
        if self.use_deep_supervision:
            # Output at native resolution (H/4) - NO upsampling (multi-resolution deep supervision)
            aux_out3 = self.aux_heads[2](x)  # [B, num_classes, H/4, W/4]
            aux_outputs.append(aux_out3)
        
        # Stage 4: H/4 -> H
        x = self.decoder4(x)
        x = self.skip4(p1, x)  # e1 (p1) → decoder4
        main_output = self.seg_head(x)
        
        if self.use_deep_supervision:
            return main_output, aux_outputs
        else:
            return main_output
    
    def get_model_info(self):
        """Get model information."""
        return {
            'decoder_type': 'BaselineHybrid2',
            'encoder_channels': self.encoder_channels,
            'decoder_channels': self.decoder_channels,
            'num_classes': self.num_classes,
            'efficientnet_variant': self.efficientnet_variant,
            'use_deep_supervision': self.use_deep_supervision,
            'use_cbam': self.use_cbam,
            'use_smart_skip': self.use_smart_skip,
            'use_cross_attn': self.use_cross_attn,
            'use_multiscale_agg': self.use_multiscale_agg,
            'use_groupnorm': self.use_groupnorm,
            'use_pos_embed': self.use_pos_embed,
            'use_bottleneck_swin': self.use_bottleneck_swin,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# ============================================================================
# SIMPLE DECODER (Baseline CNN Decoder with Flag-Controlled Enhancements)
# ============================================================================

class SimpleDecoder(nn.Module):
    """
    Simple CNN Decoder for Hybrid2 model.
    
    All enhancements are controlled by flags:
    - use_deep_supervision: Deep supervision (auxiliary outputs)
    - use_cbam: CBAM attention modules
    - use_smart_skip: Smart skip connections (attention-based)
    - use_cross_attn: Cross-attention bottleneck
    - use_multiscale_agg: Multi-scale aggregation
    - use_groupnorm: GroupNorm (default) or BatchNorm
    - use_pos_embed: Positional embeddings
    
    Architecture: Simple CNN decoder (can be replaced with ResNet50 or other backbones)
    """
    
    def __init__(self, encoder_channels, num_classes=6, 
                 efficientnet_variant='b4',
                 use_deep_supervision=False, use_cbam=False, 
                 use_smart_skip=False, use_cross_attn=False,
                 use_multiscale_agg=False, use_groupnorm=True,
                 use_pos_embed=True,
                 use_bottleneck_swin=True,
                 img_size=224):
        super().__init__()
        
        self.num_classes = num_classes
        self.encoder_channels = encoder_channels
        self.efficientnet_variant = efficientnet_variant
        self.use_deep_supervision = use_deep_supervision
        self.use_cbam = use_cbam
        self.use_smart_skip = use_smart_skip
        self.use_cross_attn = use_cross_attn
        self.use_multiscale_agg = use_multiscale_agg
        self.use_groupnorm = use_groupnorm
        self.use_pos_embed = use_pos_embed
        self.use_bottleneck_swin = use_bottleneck_swin
        self.img_size = img_size
        
        if efficientnet_variant == 'b0':
            decoder_channels = [128, 64, 32, 16]
        elif efficientnet_variant == 'b4':
            decoder_channels = [256, 128, 64, 32]
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {efficientnet_variant}")
        
        self.decoder_channels = decoder_channels
        
        print("=" * 80)
        print("🚀 SIMPLE DECODER (Flag-Controlled Enhancements)")
        print("=" * 80)
        print(f"  • EfficientNet Variant: {efficientnet_variant}")
        print(f"  • Decoder Channels: {decoder_channels}")
        print(f"  • Bottleneck Swin Blocks: {use_bottleneck_swin}")
        print(f"\n📊 Enhancement Flags:")
        print(f"  • Deep Supervision: {use_deep_supervision}")
        print(f"  • CBAM Attention: {use_cbam}")
        print(f"  • Smart Skip Connections: {use_smart_skip}")
        print(f"  • Cross-Attention: {use_cross_attn}")
        print(f"  • Multi-Scale Aggregation: {use_multiscale_agg}")
        print(f"  • GroupNorm: {use_groupnorm} (else BatchNorm)")
        print(f"  • Positional Embeddings: {use_pos_embed}")
        print("=" * 80)
        
        # Feature projections
        self.encoder_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(enc_ch, dec_ch, 1, bias=False),
                self._get_norm_layer(dec_ch),
                nn.ReLU(inplace=True)
            )
            for enc_ch, dec_ch in zip(encoder_channels, decoder_channels)
        ])
        
        # Multi-scale aggregation (optional)
        if use_multiscale_agg:
            self.multiscale_agg = MultiScaleAggregation(
                encoder_channels_list=encoder_channels,
                out_channels=decoder_channels[3],
                target_size=None
            )
        else:
            self.multiscale_agg = None
        
        # Bottleneck: 2 Swin blocks (baseline)
        if use_bottleneck_swin:
            bottleneck_resolution = (img_size // 32, img_size // 32)
            # Use encoder dimension (768) instead of decoder dimension (32)
            encoder_dim = encoder_channels[3]  # 768
            bottleneck_dim = encoder_dim  # 768 (match encoder dimension)
            
            # Choose num_heads for 768 dimension (should be 24, matching encoder Stage 4)
            if bottleneck_dim == 768:
                bottleneck_num_heads = 24  # 768 / 24 = 32 per head (matches encoder Stage 4)
            elif bottleneck_dim % 24 == 0:
                bottleneck_num_heads = 24
            elif bottleneck_dim % 16 == 0:
                bottleneck_num_heads = 16
            elif bottleneck_dim % 8 == 0:
                bottleneck_num_heads = 8
            else:
                bottleneck_num_heads = 8  # Default fallback
            
            # No token projection needed - use encoder dimension directly
            self.token_projection = None
            
            # Feature map projection: project from bottleneck output (768) to decoder input (32)
            decoder_dim = decoder_channels[3]  # 32
            self.feature_projection = nn.Sequential(
                nn.Conv2d(encoder_dim, decoder_dim, 1, bias=False),
                self._get_norm_layer(decoder_dim),
                nn.ReLU(inplace=True)
            )
            
            # Projection layers for MSA/p4 to encoder dimension (for token processing)
            # Used when MSA is enabled or when encoder_tokens not available
            self.msa_to_encoder_proj = nn.Sequential(
                nn.Conv2d(decoder_dim, encoder_dim, 1, bias=False),
                self._get_norm_layer(encoder_dim),
                nn.ReLU(inplace=True)
            )
            self.p4_to_encoder_proj = self.msa_to_encoder_proj  # Same projection for p4
            
            self.bottleneck_layer = BasicLayer(
                dim=bottleneck_dim,  # 768
                input_resolution=bottleneck_resolution,
                depth=2,
                num_heads=bottleneck_num_heads,  # 24
                window_size=7,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False
            )
        else:
            self.bottleneck_layer = None
            self.token_projection = None
            self.feature_projection = None
        
        # Optional: Cross-Attention Bottleneck
        if use_cross_attn:
            self.cross_attn = CrossAttentionBottleneck(
                decoder_dim=decoder_channels[3],
                encoder_dim=encoder_channels[3],
                num_heads=8
            )
        else:
            self.cross_attn = None
        
        # Optional: CBAM for bottleneck
        if use_cbam:
            self.bottleneck_cbam = ImprovedCBAM(decoder_channels[3])
        else:
            self.bottleneck_cbam = None
        
        # Optional: Positional Embeddings
        if use_pos_embed:
            self.pos_embed = PositionalEmbedding2D(
                decoder_channels[3], img_size // 32, img_size // 32
            )
        else:
            self.pos_embed = None
        
        # Decoder stages
        self.decoder1 = nn.Sequential(
            self._create_decoder_block(decoder_channels[3], decoder_channels[2]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        if use_smart_skip:
            self.skip1 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[3],  # p4 (e4) → decoder1
                decoder_channels=decoder_channels[2],
                fusion_type='concat'
            )
        else:
            self.skip1 = SimpleSkipConnection(
                encoder_channels=decoder_channels[3],  # p4 (e4) → decoder1
                decoder_channels=decoder_channels[2],
                use_groupnorm=use_groupnorm
            )
        
        self.decoder2 = nn.Sequential(
            self._create_decoder_block(decoder_channels[2], decoder_channels[1]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        if use_smart_skip:
            self.skip2 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[2],  # p3 (e3) → decoder2
                decoder_channels=decoder_channels[1],
                fusion_type='concat'
            )
        else:
            self.skip2 = SimpleSkipConnection(
                encoder_channels=decoder_channels[2],  # p3 (e3) → decoder2
                decoder_channels=decoder_channels[1],
                use_groupnorm=use_groupnorm
            )
        
        self.decoder3 = nn.Sequential(
            self._create_decoder_block(decoder_channels[1], decoder_channels[0]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        if use_smart_skip:
            self.skip3 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[1],  # p2 (e2) → decoder3
                decoder_channels=decoder_channels[0],
                fusion_type='concat'
            )
        else:
            self.skip3 = SimpleSkipConnection(
                encoder_channels=decoder_channels[1],  # p2 (e2) → decoder3
                decoder_channels=decoder_channels[0],
                use_groupnorm=use_groupnorm
            )
        
        # Skip connection for decoder4: e1 (p1) → decoder4
        if use_smart_skip:
            self.skip4 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[0],  # p1 (e1) → decoder4
                decoder_channels=64,  # decoder4 output is 64 channels
                fusion_type='concat'
            )
        else:
            self.skip4 = SimpleSkipConnection(
                encoder_channels=decoder_channels[0],  # p1 (e1) → decoder4
                decoder_channels=64,  # decoder4 output is 64 channels
                use_groupnorm=use_groupnorm
            )
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(decoder_channels[0], 64, 3, padding=1, bias=False),
            self._get_norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            self._get_norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1)
        )
        
        # Auxiliary heads for deep supervision (MSAGHNet-style: simple OutConv, 3x3 conv)
        # Outputs at native resolutions (NO upsampling) - matching Network model multi-resolution approach
        if use_deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(decoder_channels[0], num_classes, 3, padding=1),  # Stage 1: H/16, 256 channels
                nn.Conv2d(decoder_channels[1], num_classes, 3, padding=1),  # Stage 2: H/8, 128 channels
                nn.Conv2d(decoder_channels[2], num_classes, 3, padding=1),  # Stage 3: H/4, 64 channels
            ])
        else:
            self.aux_heads = None
    
    def _get_norm_layer(self, channels):
        """Get normalization layer based on flag."""
        if self.use_groupnorm:
            return get_norm_layer(channels, 'group')
        else:
            return nn.BatchNorm2d(channels)
    
    def _create_decoder_block(self, in_channels, out_channels):
        """Create decoder block with optional CBAM."""
        block = SimpleDecoderBlock(
            in_channels, 
            out_channels, 
            use_batchnorm=not self.use_groupnorm,
            use_groupnorm=self.use_groupnorm
        )
        
        if self.use_cbam:
            cbam = ImprovedCBAM(out_channels)
            return nn.Sequential(block, cbam)
        else:
            return block
    
    def forward(self, encoder_features, encoder_tokens=None):
        f1, f2, f3, f4 = encoder_features
        
        p1 = self.encoder_projections[0](f1)
        p2 = self.encoder_projections[1](f2)
        p3 = self.encoder_projections[2](f3)
        p4 = self.encoder_projections[3](f4)
        
        if self.use_multiscale_agg:
            # If MSA enabled: bottleneck receives input ONLY from MSA output (not MSA + p4)
            bottleneck_feat = self.multiscale_agg([f1, f2, f3, f4])
            if bottleneck_feat.shape[2:] != p4.shape[2:]:
                bottleneck_feat = F.interpolate(bottleneck_feat, size=p4.shape[2:], 
                                              mode='bilinear', align_corners=False)
            x = bottleneck_feat  # MSA output ONLY (no addition with p4)
        else:
            # If MSA disabled: bottleneck receives input ONLY from e4 (p4)
            x = p4
        
        # Bottleneck: 2 Swin blocks
        # If MSA enabled: bottleneck receives input ONLY from MSA output (convert to tokens)
        # If MSA disabled: bottleneck receives input from e4 (use encoder_tokens if available, else p4)
        if self.use_bottleneck_swin:
            if self.use_multiscale_agg:
                # MSA enabled: use MSA output (x) - convert feature maps to tokens
                # x is MSA output at decoder_dim (32), need to project to encoder_dim (768) for token processing
                B, C, H, W = x.shape
                encoder_dim = self.encoder_channels[3]  # 768
                
                # Project MSA output to encoder dimension for bottleneck token processing
                if C != encoder_dim:
                    x = self.msa_to_encoder_proj(x)  # (B, 768, 7, 7)
                
                # Convert feature maps to tokens
                x_tokens = x.flatten(2).permute(0, 2, 1)  # (B, 49, 768)
                x_tokens = self.bottleneck_layer(x_tokens)  # (B, 49, 768)
                x = x_tokens.transpose(1, 2).view(B, encoder_dim, H, W)  # (B, 768, 7, 7)
                
                # Project from encoder dimension (768) to decoder dimension (32)
                x = self.feature_projection(x)  # (B, 32, 7, 7)
            elif encoder_tokens is not None:
                # MSA disabled AND encoder_tokens available: use encoder_tokens directly
                stage4_tokens = encoder_tokens  # (B, 49, 768)
                x_tokens = self.bottleneck_layer(stage4_tokens)  # (B, 49, 768)
                
                # Convert tokens back to feature maps
                B, L, C = x_tokens.shape
                H = W = int(L ** 0.5)  # H = W = 7
                x = x_tokens.transpose(1, 2).view(B, C, H, W)  # (B, 768, 7, 7)
                
                # Project from encoder dimension (768) to decoder dimension (32)
                x = self.feature_projection(x)  # (B, 32, 7, 7)
            else:
                # MSA disabled AND no encoder_tokens: use p4, convert to tokens
                B, C, H, W = x.shape  # x is p4 at decoder_dim (32)
                encoder_dim = self.encoder_channels[3]  # 768
                
                # Project p4 to encoder dimension for token processing
                if C != encoder_dim:
                    x = self.p4_to_encoder_proj(x)  # (B, 768, 7, 7)
                
                # Convert feature maps to tokens
                x_tokens = x.flatten(2).permute(0, 2, 1)  # (B, 49, 768)
                x_tokens = self.bottleneck_layer(x_tokens)  # (B, 49, 768)
                x = x_tokens.transpose(1, 2).view(B, encoder_dim, H, W)  # (B, 768, 7, 7)
                
                # Project from encoder dimension (768) to decoder dimension (32)
                x = self.feature_projection(x)  # (B, 32, 7, 7)
        
        if self.bottleneck_cbam is not None:
            x = self.bottleneck_cbam(x)
        
        if self.pos_embed is not None:
            x = self.pos_embed(x)
        
        if self.use_cross_attn and encoder_tokens is not None:
            x = self.cross_attn(x, encoder_tokens)
        
        aux_outputs = [] if self.use_deep_supervision else None
        
        # Stage 1: H/32 -> H/16
        x = self.decoder1(x)
        x = self.skip1(p4, x)  # e4 (p4) → decoder1
        if self.use_deep_supervision:
            # Output at native resolution (H/16) - NO upsampling (multi-resolution deep supervision)
            aux_out1 = self.aux_heads[0](x)  # [B, num_classes, H/16, W/16]
            aux_outputs.append(aux_out1)
        
        # Stage 2: H/16 -> H/8
        x = self.decoder2(x)
        x = self.skip2(p3, x)  # e3 (p3) → decoder2
        if self.use_deep_supervision:
            # Output at native resolution (H/8) - NO upsampling (multi-resolution deep supervision)
            aux_out2 = self.aux_heads[1](x)  # [B, num_classes, H/8, W/8]
            aux_outputs.append(aux_out2)
        
        # Stage 3: H/8 -> H/4
        x = self.decoder3(x)
        x = self.skip3(p2, x)  # e2 (p2) → decoder3
        if self.use_deep_supervision:
            # Output at native resolution (H/4) - NO upsampling (multi-resolution deep supervision)
            aux_out3 = self.aux_heads[2](x)  # [B, num_classes, H/4, W/4]
            aux_outputs.append(aux_out3)
        
        # Stage 4: H/4 -> H
        x = self.decoder4(x)
        x = self.skip4(p1, x)  # e1 (p1) → decoder4
        main_output = self.seg_head(x)
        
        if self.use_deep_supervision:
            return main_output, aux_outputs
        else:
            return main_output
    
    def get_model_info(self):
        """Get model information."""
        return {
            'decoder_type': 'SimpleDecoder',
            'encoder_channels': self.encoder_channels,
            'decoder_channels': self.decoder_channels,
            'num_classes': self.num_classes,
            'efficientnet_variant': self.efficientnet_variant,
            'use_deep_supervision': self.use_deep_supervision,
            'use_cbam': self.use_cbam,
            'use_smart_skip': self.use_smart_skip,
            'use_cross_attn': self.use_cross_attn,
            'use_multiscale_agg': self.use_multiscale_agg,
            'use_groupnorm': self.use_groupnorm,
            'use_pos_embed': self.use_pos_embed,
            'use_bottleneck_swin': self.use_bottleneck_swin,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# ============================================================================
# RESNET50 DECODER (ResNet50-style decoder with bottleneck blocks)
# ============================================================================

class ResNet50BottleneckBlock(nn.Module):
    """ResNet50-style bottleneck block for decoder."""
    
    def __init__(self, in_channels, out_channels, stride=1, use_groupnorm=True):
        super().__init__()
        self.expansion = 4  # ResNet50 expansion factor
        
        # 1x1 conv to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = get_norm_layer(out_channels, 'group' if use_groupnorm else 'batch')
        
        # 3x3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=stride, bias=False)
        self.bn2 = get_norm_layer(out_channels, 'group' if use_groupnorm else 'batch')
        
        # 1x1 conv to expand channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = get_norm_layer(out_channels * self.expansion, 'group' if use_groupnorm else 'batch')
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False),
                get_norm_layer(out_channels * self.expansion, 'group' if use_groupnorm else 'batch')
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class ResNet50Decoder(nn.Module):
    """
    ResNet50-style Decoder for Hybrid2 model.
    
    Uses ResNet50 bottleneck blocks with residual connections.
    All enhancements are controlled by flags (same as SimpleDecoder).
    """
    
    def __init__(self, encoder_channels, num_classes=6,
                 use_deep_supervision=False, use_cbam=False,
                 use_smart_skip=False, use_cross_attn=False,
                 use_multiscale_agg=False, use_groupnorm=True,
                 use_pos_embed=True,
                 use_bottleneck_swin=True,
                 img_size=224):
        super().__init__()
        
        self.num_classes = num_classes
        self.encoder_channels = encoder_channels
        self.use_deep_supervision = use_deep_supervision
        self.use_cbam = use_cbam
        self.use_smart_skip = use_smart_skip
        self.use_cross_attn = use_cross_attn
        self.use_multiscale_agg = use_multiscale_agg
        self.use_groupnorm = use_groupnorm
        self.use_pos_embed = use_pos_embed
        self.use_bottleneck_swin = use_bottleneck_swin
        self.img_size = img_size
        
        # ResNet50 decoder channels: [512, 256, 128, 64] (expansion factor 4)
        decoder_channels = [512, 256, 128, 64]
        self.decoder_channels = decoder_channels
        
        print("=" * 80)
        print("🚀 RESNET50 DECODER (Flag-Controlled Enhancements)")
        print("=" * 80)
        print(f"  • Decoder Channels: {decoder_channels}")
        print(f"  • Bottleneck Swin Blocks: {use_bottleneck_swin}")
        print(f"\n📊 Enhancement Flags:")
        print(f"  • Deep Supervision: {use_deep_supervision}")
        print(f"  • CBAM Attention: {use_cbam}")
        print(f"  • Smart Skip Connections: {use_smart_skip}")
        print(f"  • Cross-Attention: {use_cross_attn}")
        print(f"  • Multi-Scale Aggregation: {use_multiscale_agg}")
        print(f"  • GroupNorm: {use_groupnorm} (else BatchNorm)")
        print(f"  • Positional Embeddings: {use_pos_embed}")
        print("=" * 80)
        
        # Feature projections
        self.encoder_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(enc_ch, dec_ch, 1, bias=False),
                self._get_norm_layer(dec_ch),
                nn.ReLU(inplace=True)
            )
            for enc_ch, dec_ch in zip(encoder_channels, decoder_channels)
        ])
        
        # Multi-scale aggregation (optional)
        if use_multiscale_agg:
            self.multiscale_agg = MultiScaleAggregation(
                encoder_channels_list=encoder_channels,
                out_channels=decoder_channels[3],
                target_size=None
            )
        else:
            self.multiscale_agg = None
        
        # Bottleneck: 2 Swin blocks (same as SimpleDecoder)
        if use_bottleneck_swin:
            bottleneck_resolution = (img_size // 32, img_size // 32)
            encoder_dim = encoder_channels[3]  # 768
            bottleneck_dim = encoder_dim  # 768
            
            if bottleneck_dim == 768:
                bottleneck_num_heads = 24
            elif bottleneck_dim % 24 == 0:
                bottleneck_num_heads = 24
            elif bottleneck_dim % 16 == 0:
                bottleneck_num_heads = 16
            elif bottleneck_dim % 8 == 0:
                bottleneck_num_heads = 8
            else:
                bottleneck_num_heads = 8
            
            self.token_projection = None
            
            decoder_dim = decoder_channels[3]  # 64
            self.feature_projection = nn.Sequential(
                nn.Conv2d(encoder_dim, decoder_dim, 1, bias=False),
                self._get_norm_layer(decoder_dim),
                nn.ReLU(inplace=True)
            )
            
            # Projection layers for MSA/p4 to encoder dimension (for token processing)
            # Used when MSA is enabled or when encoder_tokens not available
            self.msa_to_encoder_proj = nn.Sequential(
                nn.Conv2d(decoder_dim, encoder_dim, 1, bias=False),
                self._get_norm_layer(encoder_dim),
                nn.ReLU(inplace=True)
            )
            self.p4_to_encoder_proj = self.msa_to_encoder_proj  # Same projection for p4
            
            self.bottleneck_layer = BasicLayer(
                dim=bottleneck_dim,
                input_resolution=bottleneck_resolution,
                depth=2,
                num_heads=bottleneck_num_heads,
                window_size=7,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False
            )
        else:
            self.bottleneck_layer = None
            self.token_projection = None
            self.feature_projection = None
        
        # Optional: Cross-Attention Bottleneck
        if use_cross_attn:
            self.cross_attn = CrossAttentionBottleneck(
                decoder_dim=decoder_channels[3],
                encoder_dim=encoder_channels[3],
                num_heads=8
            )
        else:
            self.cross_attn = None
        
        # Optional: CBAM for bottleneck
        if use_cbam:
            self.bottleneck_cbam = ImprovedCBAM(decoder_channels[3])
        else:
            self.bottleneck_cbam = None
        
        # Optional: Positional Embeddings
        if use_pos_embed:
            self.pos_embed = PositionalEmbedding2D(
                decoder_channels[3], img_size // 32, img_size // 32
            )
        else:
            self.pos_embed = None
        
        # Decoder stages with ResNet50 bottleneck blocks
        self.decoder1 = nn.Sequential(
            ResNet50BottleneckBlock(decoder_channels[3], decoder_channels[3] // 4, use_groupnorm=use_groupnorm),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        if use_smart_skip:
            self.skip1 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[3],  # p4 (e4) → decoder1
                decoder_channels=decoder_channels[2],
                fusion_type='concat'
            )
        else:
            self.skip1 = SimpleSkipConnection(
                encoder_channels=decoder_channels[3],  # p4 (e4) → decoder1
                decoder_channels=decoder_channels[2],
                use_groupnorm=use_groupnorm
            )
        
        self.decoder2 = nn.Sequential(
            ResNet50BottleneckBlock(decoder_channels[2], decoder_channels[2] // 4, use_groupnorm=use_groupnorm),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        if use_smart_skip:
            self.skip2 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[2],  # p3 (e3) → decoder2
                decoder_channels=decoder_channels[1],
                fusion_type='concat'
            )
        else:
            self.skip2 = SimpleSkipConnection(
                encoder_channels=decoder_channels[2],  # p3 (e3) → decoder2
                decoder_channels=decoder_channels[1],
                use_groupnorm=use_groupnorm
            )
        
        self.decoder3 = nn.Sequential(
            ResNet50BottleneckBlock(decoder_channels[1], decoder_channels[1] // 4, use_groupnorm=use_groupnorm),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        if use_smart_skip:
            self.skip3 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[1],  # p2 (e2) → decoder3
                decoder_channels=decoder_channels[0],
                fusion_type='concat'
            )
        else:
            self.skip3 = SimpleSkipConnection(
                encoder_channels=decoder_channels[1],  # p2 (e2) → decoder3
                decoder_channels=decoder_channels[0],
                use_groupnorm=use_groupnorm
            )
        
        # Skip connection for decoder4: e1 (p1) → decoder4
        if use_smart_skip:
            self.skip4 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[0],  # p1 (e1) → decoder4
                decoder_channels=64,  # decoder4 output is 64 channels
                fusion_type='concat'
            )
        else:
            self.skip4 = SimpleSkipConnection(
                encoder_channels=decoder_channels[0],  # p1 (e1) → decoder4
                decoder_channels=64,  # decoder4 output is 64 channels
                use_groupnorm=use_groupnorm
            )
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(decoder_channels[0], 64, 3, padding=1, bias=False),
            self._get_norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            self._get_norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1)
        )
        
        # Auxiliary heads for deep supervision (MSAGHNet-style: simple OutConv, 3x3 conv)
        # Outputs at native resolutions (NO upsampling) - matching Network model multi-resolution approach
        if use_deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(decoder_channels[0], num_classes, 3, padding=1),  # Stage 1: H/16, 256 channels
                nn.Conv2d(decoder_channels[1], num_classes, 3, padding=1),  # Stage 2: H/8, 128 channels
                nn.Conv2d(decoder_channels[2], num_classes, 3, padding=1),  # Stage 3: H/4, 64 channels
            ])
        else:
            self.aux_heads = None
    
    def _get_norm_layer(self, channels):
        """Get normalization layer based on flag."""
        if self.use_groupnorm:
            return get_norm_layer(channels, 'group')
        else:
            return nn.BatchNorm2d(channels)
    
    def forward(self, encoder_features, encoder_tokens=None):
        f1, f2, f3, f4 = encoder_features
        
        p1 = self.encoder_projections[0](f1)
        p2 = self.encoder_projections[1](f2)
        p3 = self.encoder_projections[2](f3)
        p4 = self.encoder_projections[3](f4)
        
        if self.use_multiscale_agg:
            bottleneck_feat = self.multiscale_agg([f1, f2, f3, f4])
            if bottleneck_feat.shape[2:] != p4.shape[2:]:
                bottleneck_feat = F.interpolate(bottleneck_feat, size=p4.shape[2:],
                                              mode='bilinear', align_corners=False)
            x = bottleneck_feat + p4
        else:
            x = p4
        
        # Bottleneck: 2 Swin blocks
        if self.use_bottleneck_swin:
            if encoder_tokens is not None:
                stage4_tokens = encoder_tokens
                x_tokens = self.bottleneck_layer(stage4_tokens)
                B, L, C = x_tokens.shape
                H = W = int(L ** 0.5)
                x = x_tokens.transpose(1, 2).view(B, C, H, W)
                x = self.feature_projection(x)
            else:
                B, C, H, W = x.shape
                x_tokens = x.flatten(2).permute(0, 2, 1)
                x_tokens = self.bottleneck_layer(x_tokens)
                x = x_tokens.permute(0, 2, 1).reshape(B, C, H, W)
                if self.feature_projection is not None and C != self.decoder_channels[3]:
                    x = self.feature_projection(x)
        
        if self.bottleneck_cbam is not None:
            x = self.bottleneck_cbam(x)
        
        if self.pos_embed is not None:
            x = self.pos_embed(x)
        
        if self.use_cross_attn and encoder_tokens is not None:
            x = self.cross_attn(x, encoder_tokens)
        
        aux_outputs = [] if self.use_deep_supervision else None
        
        # Stage 1: H/32 -> H/16
        x = self.decoder1(x)
        x = self.skip1(p4, x)  # e4 (p4) → decoder1
        if self.use_deep_supervision:
            # Output at native resolution (H/16) - NO upsampling (multi-resolution deep supervision)
            aux_out1 = self.aux_heads[0](x)  # [B, num_classes, H/16, W/16]
            aux_outputs.append(aux_out1)
        
        # Stage 2: H/16 -> H/8
        x = self.decoder2(x)
        x = self.skip2(p3, x)  # e3 (p3) → decoder2
        if self.use_deep_supervision:
            # Output at native resolution (H/8) - NO upsampling (multi-resolution deep supervision)
            aux_out2 = self.aux_heads[1](x)  # [B, num_classes, H/8, W/8]
            aux_outputs.append(aux_out2)
        
        # Stage 3: H/8 -> H/4
        x = self.decoder3(x)
        x = self.skip3(p2, x)  # e2 (p2) → decoder3
        if self.use_deep_supervision:
            # Output at native resolution (H/4) - NO upsampling (multi-resolution deep supervision)
            aux_out3 = self.aux_heads[2](x)  # [B, num_classes, H/4, W/4]
            aux_outputs.append(aux_out3)
        
        # Stage 4: H/4 -> H
        x = self.decoder4(x)
        x = self.skip4(p1, x)  # e1 (p1) → decoder4
        main_output = self.seg_head(x)
        
        if self.use_deep_supervision:
            return main_output, aux_outputs
        else:
            return main_output
    
    def get_model_info(self):
        """Get model information."""
        return {
            'decoder_type': 'ResNet50Decoder',
            'encoder_channels': self.encoder_channels,
            'decoder_channels': self.decoder_channels,
            'num_classes': self.num_classes,
            'use_deep_supervision': self.use_deep_supervision,
            'use_cbam': self.use_cbam,
            'use_smart_skip': self.use_smart_skip,
            'use_cross_attn': self.use_cross_attn,
            'use_multiscale_agg': self.use_multiscale_agg,
            'use_groupnorm': self.use_groupnorm,
            'use_pos_embed': self.use_pos_embed,
            'use_bottleneck_swin': self.use_bottleneck_swin,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# ============================================================================
# EFFICIENTNET-B4 DECODER (Real MBConv Blocks from timm)
# ============================================================================

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv) block.
    Used in EfficientNet architectures.
    Structure: Expand -> Depthwise -> SE -> Project
    """
    def __init__(self, in_channels, out_channels, expansion_ratio=6, 
                 kernel_size=3, stride=1, se_ratio=0.25, use_groupnorm=False):
        super().__init__()
        self.stride = stride
        self.expanded_channels = int(in_channels * expansion_ratio)
        
        # Expansion phase (1x1 conv)
        if expansion_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, self.expanded_channels, 1, bias=False)
            if use_groupnorm:
                self.expand_bn = get_norm_layer(self.expanded_channels, 'group')
            else:
                self.expand_bn = nn.BatchNorm2d(self.expanded_channels)
            self.expand_act = nn.SiLU()  # Swish activation
        else:
            self.expand_conv = None
        
        # Depthwise convolution (3x3 or 5x5)
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv2d(
            self.expanded_channels, self.expanded_channels, 
            kernel_size, stride=stride, padding=padding, 
            groups=self.expanded_channels, bias=False
        )
        if use_groupnorm:
            self.depthwise_bn = get_norm_layer(self.expanded_channels, 'group')
        else:
            self.depthwise_bn = nn.BatchNorm2d(self.expanded_channels)
        self.depthwise_act = nn.SiLU()
        
        # Squeeze-and-Excitation (SE) module
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.expanded_channels, se_channels, 1),
            nn.SiLU(),
            nn.Conv2d(se_channels, self.expanded_channels, 1),
            nn.Sigmoid()
        )
        
        # Projection phase (1x1 conv)
        self.project_conv = nn.Conv2d(self.expanded_channels, out_channels, 1, bias=False)
        if use_groupnorm:
            self.project_bn = get_norm_layer(out_channels, 'group')
        else:
            self.project_bn = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.use_residual = (stride == 1 and in_channels == out_channels)
    
    def forward(self, x):
        identity = x
        
        # Expansion
        if self.expand_conv is not None:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.expand_act(x)
        
        # Depthwise
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_act(x)
        
        # SE
        x = x * self.se(x)
        
        # Projection
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        # Skip connection
        if self.use_residual:
            x = x + identity
        
        return x


class EfficientNetB4Decoder(nn.Module):
    """
    Real EfficientNet-B4 Decoder using MBConv blocks from timm architecture.
    Uses actual MBConv blocks (not simple Conv blocks) for decoder stages.
    """
    
    def __init__(self, encoder_channels, num_classes=6,
                 use_deep_supervision=False, use_cbam=False,
                 use_smart_skip=False, use_cross_attn=False,
                 use_multiscale_agg=False, use_groupnorm=True,
                 use_pos_embed=True, use_bottleneck_swin=True,
                 img_size=224):
        super().__init__()
        
        self.num_classes = num_classes
        self.encoder_channels = encoder_channels
        self.use_deep_supervision = use_deep_supervision
        self.use_cbam = use_cbam
        self.use_smart_skip = use_smart_skip
        self.use_cross_attn = use_cross_attn
        self.use_multiscale_agg = use_multiscale_agg
        self.use_groupnorm = use_groupnorm
        self.use_pos_embed = use_pos_embed
        self.use_bottleneck_swin = use_bottleneck_swin
        self.img_size = img_size
        
        # EfficientNet-B4 decoder channels (matching encoder output scales)
        decoder_channels = [256, 128, 64, 32]
        self.decoder_channels = decoder_channels
        
        print("=" * 80)
        print("🚀 EFFICIENTNET-B4 DECODER (Real MBConv Blocks)")
        print("=" * 80)
        print(f"  • Decoder Channels: {decoder_channels}")
        print(f"  • Using MBConv blocks (NOT simple Conv)")
        print(f"  • Bottleneck Swin Blocks: {use_bottleneck_swin}")
        print(f"\n📊 Optional Features:")
        print(f"  • Deep Supervision: {use_deep_supervision}")
        print(f"  • CBAM Attention: {use_cbam}")
        print(f"  • Smart Skip Connections: {use_smart_skip}")
        print(f"  • Cross-Attention: {use_cross_attn}")
        print(f"  • Multi-Scale Aggregation: {use_multiscale_agg}")
        print(f"  • GroupNorm: {use_groupnorm} (else BatchNorm)")
        print(f"  • Positional Embeddings: {use_pos_embed}")
        print("=" * 80)
        
        # Feature projections
        self.encoder_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(enc_ch, dec_ch, 1, bias=False),
                self._get_norm_layer(dec_ch),
                nn.ReLU(inplace=True)
            )
            for enc_ch, dec_ch in zip(encoder_channels, decoder_channels)
        ])
        
        # Multi-scale aggregation (optional)
        if use_multiscale_agg:
            self.multiscale_agg = MultiScaleAggregation(
                encoder_channels_list=encoder_channels,
                out_channels=decoder_channels[3],
                target_size=None
            )
        else:
            self.multiscale_agg = None
        
        # Bottleneck: 2 Swin blocks (same as SimpleDecoder)
        if use_bottleneck_swin:
            bottleneck_resolution = (img_size // 32, img_size // 32)
            encoder_dim = encoder_channels[3]  # 768
            bottleneck_dim = encoder_dim  # 768
            
            if bottleneck_dim == 768:
                bottleneck_num_heads = 24
            elif bottleneck_dim % 24 == 0:
                bottleneck_num_heads = 24
            elif bottleneck_dim % 16 == 0:
                bottleneck_num_heads = 16
            elif bottleneck_dim % 8 == 0:
                bottleneck_num_heads = 8
            else:
                bottleneck_num_heads = 8
            
            self.token_projection = None
            
            decoder_dim = decoder_channels[3]  # 32
            self.feature_projection = nn.Sequential(
                nn.Conv2d(encoder_dim, decoder_dim, 1, bias=False),
                self._get_norm_layer(decoder_dim),
                nn.ReLU(inplace=True)
            )
            
            # Projection layers for MSA/p4 to encoder dimension (for token processing)
            # Used when MSA is enabled or when encoder_tokens not available
            self.msa_to_encoder_proj = nn.Sequential(
                nn.Conv2d(decoder_dim, encoder_dim, 1, bias=False),
                self._get_norm_layer(encoder_dim),
                nn.ReLU(inplace=True)
            )
            self.p4_to_encoder_proj = self.msa_to_encoder_proj  # Same projection for p4
            
            self.bottleneck_layer = BasicLayer(
                dim=bottleneck_dim,
                input_resolution=bottleneck_resolution,
                depth=2,
                num_heads=bottleneck_num_heads,
                window_size=7,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False
            )
        else:
            self.bottleneck_layer = None
            self.token_projection = None
            self.feature_projection = None
        
        # Optional: Cross-Attention Bottleneck
        if use_cross_attn:
            self.cross_attn = CrossAttentionBottleneck(
                decoder_dim=decoder_channels[3],
                encoder_dim=encoder_channels[3],
                num_heads=8
            )
        else:
            self.cross_attn = None
        
        # Optional: CBAM for bottleneck
        if use_cbam:
            self.bottleneck_cbam = ImprovedCBAM(decoder_channels[3])
        else:
            self.bottleneck_cbam = None
        
        # Optional: Positional Embeddings
        if use_pos_embed:
            self.pos_embed = PositionalEmbedding2D(
                decoder_channels[3], img_size // 32, img_size // 32
            )
        else:
            self.pos_embed = None
        
        # Decoder stages with MBConv blocks
        # Stage 1: 32 -> 256 (with upsampling)
        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            MBConvBlock(32, 256, expansion_ratio=6, kernel_size=3, stride=1, use_groupnorm=use_groupnorm),
            MBConvBlock(256, 256, expansion_ratio=6, kernel_size=3, stride=1, use_groupnorm=use_groupnorm)
        )
        
        if use_smart_skip:
            self.skip1 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[3],  # p4 (e4) → decoder1
                decoder_channels=decoder_channels[0],  # decoder1 outputs 256 channels
                fusion_type='concat'
            )
        else:
            self.skip1 = SimpleSkipConnection(
                encoder_channels=decoder_channels[3],  # p4 (e4) → decoder1
                decoder_channels=decoder_channels[0],  # decoder1 outputs 256 channels
                use_groupnorm=use_groupnorm
            )
        
        # Stage 2: 256 -> 128 (with upsampling)
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            MBConvBlock(256, 128, expansion_ratio=6, kernel_size=3, stride=1, use_groupnorm=use_groupnorm),
            MBConvBlock(128, 128, expansion_ratio=6, kernel_size=3, stride=1, use_groupnorm=use_groupnorm)
        )
        
        if use_smart_skip:
            self.skip2 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[2],  # p3 (e3) → decoder2
                decoder_channels=decoder_channels[1],  # decoder2 outputs 128 channels
                fusion_type='concat'
            )
        else:
            self.skip2 = SimpleSkipConnection(
                encoder_channels=decoder_channels[2],  # p3 (e3) → decoder2
                decoder_channels=decoder_channels[1],  # decoder2 outputs 128 channels
                use_groupnorm=use_groupnorm
            )
        
        # Stage 3: 128 -> 64 (with upsampling)
        self.decoder3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            MBConvBlock(128, 64, expansion_ratio=6, kernel_size=3, stride=1, use_groupnorm=use_groupnorm),
            MBConvBlock(64, 64, expansion_ratio=6, kernel_size=3, stride=1, use_groupnorm=use_groupnorm)
        )
        
        if use_smart_skip:
            self.skip3 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[1],  # p2 (e2) → decoder3
                decoder_channels=decoder_channels[2],  # decoder3 outputs 64 channels
                fusion_type='concat'
            )
        else:
            self.skip3 = SimpleSkipConnection(
                encoder_channels=decoder_channels[1],  # p2 (e2) → decoder3
                decoder_channels=decoder_channels[2],  # decoder3 outputs 64 channels
                use_groupnorm=use_groupnorm
            )
        
        # Skip connection for decoder4: e1 (p1) → decoder4
        if use_smart_skip:
            self.skip4 = ImprovedSmartSkipConnection(
                encoder_channels=decoder_channels[0],  # p1 (e1) → decoder4
                decoder_channels=64,  # decoder4 output is 64 channels
                fusion_type='concat'
            )
        else:
            self.skip4 = SimpleSkipConnection(
                encoder_channels=decoder_channels[0],  # p1 (e1) → decoder4
                decoder_channels=64,  # decoder4 output is 64 channels
                use_groupnorm=use_groupnorm
            )
        
        # Stage 4: 64 -> 64 (with progressive upsampling: 2x → MBConv → 2x)
        # This matches U-Net design principles: progressive 2x upsampling is better than single 4x
        # Consistent with other decoder stages which all use 2x upsampling
        self.decoder4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # First 2x upsampling
            MBConvBlock(64, 64, expansion_ratio=6, kernel_size=3, stride=1, use_groupnorm=use_groupnorm),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)   # Second 2x upsampling
        )
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            self._get_norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1)
        )
        
        # Auxiliary heads for deep supervision (MSAGHNet-style: simple OutConv, 3x3 conv)
        # Outputs at native resolutions (NO upsampling) - matching Network model multi-resolution approach
        if use_deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(decoder_channels[0], num_classes, 3, padding=1),  # Stage 1: H/16, 256 channels
                nn.Conv2d(decoder_channels[1], num_classes, 3, padding=1),  # Stage 2: H/8, 128 channels
                nn.Conv2d(decoder_channels[2], num_classes, 3, padding=1),  # Stage 3: H/4, 64 channels
            ])
        else:
            self.aux_heads = None
    
    def _get_norm_layer(self, channels):
        """Get normalization layer based on flag."""
        if self.use_groupnorm:
            return get_norm_layer(channels, 'group')
        else:
            return nn.BatchNorm2d(channels)
    
    def forward(self, encoder_features, encoder_tokens=None):
        f1, f2, f3, f4 = encoder_features
        
        p1 = self.encoder_projections[0](f1)
        p2 = self.encoder_projections[1](f2)
        p3 = self.encoder_projections[2](f3)
        p4 = self.encoder_projections[3](f4)
        
        if self.use_multiscale_agg:
            # If MSA enabled: bottleneck receives input ONLY from MSA output (not MSA + p4)
            bottleneck_feat = self.multiscale_agg([f1, f2, f3, f4])
            if bottleneck_feat.shape[2:] != p4.shape[2:]:
                bottleneck_feat = F.interpolate(bottleneck_feat, size=p4.shape[2:], 
                                              mode='bilinear', align_corners=False)
            x = bottleneck_feat  # MSA output ONLY (no addition with p4)
        else:
            # If MSA disabled: bottleneck receives input ONLY from e4 (p4)
            x = p4
        
        # Bottleneck: 2 Swin blocks
        if self.use_bottleneck_swin:
            if encoder_tokens is not None:
                stage4_tokens = encoder_tokens
                x_tokens = self.bottleneck_layer(stage4_tokens)
                B, L, C = x_tokens.shape
                H = W = int(L ** 0.5)
                x = x_tokens.transpose(1, 2).view(B, C, H, W)
                x = self.feature_projection(x)
            else:
                B, C, H, W = x.shape
                x_tokens = x.flatten(2).permute(0, 2, 1)
                x_tokens = self.bottleneck_layer(x_tokens)
                x = x_tokens.permute(0, 2, 1).reshape(B, C, H, W)
                if self.feature_projection is not None and C != self.decoder_channels[3]:
                    x = self.feature_projection(x)
        
        if self.bottleneck_cbam is not None:
            x = self.bottleneck_cbam(x)
        
        if self.pos_embed is not None:
            x = self.pos_embed(x)
        
        if self.use_cross_attn and encoder_tokens is not None:
            x = self.cross_attn(x, encoder_tokens)
        
        aux_outputs = [] if self.use_deep_supervision else None
        
        # Stage 1: H/32 -> H/16 (Upsample then MBConv)
        x = self.decoder1(x)
        x = self.skip1(p4, x)  # e4 (p4) → decoder1
        if self.use_deep_supervision:
            # Output at native resolution (H/16) - NO upsampling (multi-resolution deep supervision)
            aux_out1 = self.aux_heads[0](x)  # [B, num_classes, H/16, W/16]
            aux_outputs.append(aux_out1)
        
        # Stage 2: H/16 -> H/8 (Upsample then MBConv)
        x = self.decoder2(x)
        x = self.skip2(p3, x)  # e3 (p3) → decoder2
        if self.use_deep_supervision:
            # Output at native resolution (H/8) - NO upsampling (multi-resolution deep supervision)
            aux_out2 = self.aux_heads[1](x)  # [B, num_classes, H/8, W/8]
            aux_outputs.append(aux_out2)
        
        # Stage 3: H/8 -> H/4 (Upsample then MBConv)
        x = self.decoder3(x)
        x = self.skip3(p2, x)  # e2 (p2) → decoder3
        if self.use_deep_supervision:
            # Output at native resolution (H/4) - NO upsampling (multi-resolution deep supervision)
            aux_out3 = self.aux_heads[2](x)  # [B, num_classes, H/4, W/4]
            aux_outputs.append(aux_out3)
        
        # Stage 4: H/4 -> H (Upsample then MBConv)
        x = self.decoder4(x)
        x = self.skip4(p1, x)  # e1 (p1) → decoder4
        main_output = self.seg_head(x)
        
        if self.use_deep_supervision:
            return main_output, aux_outputs
        else:
            return main_output
    
    def get_model_info(self):
        """Get model information."""
        return {
            'decoder_type': 'EfficientNetB4Decoder',
            'encoder_channels': self.encoder_channels,
            'decoder_channels': self.decoder_channels,
            'num_classes': self.num_classes,
            'use_deep_supervision': self.use_deep_supervision,
            'use_cbam': self.use_cbam,
            'use_smart_skip': self.use_smart_skip,
            'use_cross_attn': self.use_cross_attn,
            'use_multiscale_agg': self.use_multiscale_agg,
            'use_groupnorm': self.use_groupnorm,
            'use_pos_embed': self.use_pos_embed,
            'use_bottleneck_swin': self.use_bottleneck_swin,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


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
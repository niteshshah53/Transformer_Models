import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
import copy


def get_norm_layer(channels, norm_type='group', num_groups=32):
    """Factory function for normalization layers."""
    if norm_type == 'group':
        num_groups = min(num_groups, channels)
        while channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.BatchNorm2d(channels)


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
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

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


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage (encoder-style, no upsample).
    
    This matches SwinUnet's BasicLayer implementation.
    """
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


class BasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
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

        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class CNNFeatureAdapter(nn.Module):
    """Adapts CNN features to transformer format with re-embedding"""
    def __init__(self, in_channels, out_channels, spatial_size, use_groupnorm=False):
        super().__init__()
        self.spatial_size = spatial_size
        self.out_channels = out_channels
        self.use_groupnorm = use_groupnorm
        
        # Projection layer to match transformer dimensions
        if use_groupnorm:
            # For CNN features, use GroupNorm
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                get_norm_layer(out_channels, 'group'),
                nn.GELU()
            )
        else:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GELU()
            )
        
        # Additional fully connected layer for feature refinement
        self.fc_refine = nn.Linear(out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)  # Always LayerNorm for tokens
        
        # Initialize weights properly (Vision Transformer style)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following Vision Transformer convention."""
        # Truncated normal initialization for Linear layers (ViT style)
        trunc_normal_(self.fc_refine.weight, std=.02)
        if self.fc_refine.bias is not None:
            nn.init.constant_(self.fc_refine.bias, 0)
        
        # LayerNorm initialization
        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)
        
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Project to target channels
        x = self.projection(x)  # (B, out_channels, H, W)
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, out_channels)
        
        # Refine features with FC layer
        x = self.fc_refine(x)
        x = self.norm(x)
        
        return x


class FourierFeatureFusion(nn.Module):
    """
    Fourier-based feature fusion for CNN-Transformer hybrid models.
    Combines features using FFT in frequency domain with proper dimension handling.
    """
    def __init__(self, in_dim1, in_dim2, out_dim):
        super().__init__()
        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim2
        self.out_dim = out_dim
        
        # Project to common dimension first
        self.proj1 = nn.Linear(in_dim1, out_dim)
        self.proj2 = nn.Linear(in_dim2, out_dim)
        
        # Final projection after fusion
        self.fusion_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, feat1, feat2):
        """
        Args:
            feat1: (B, L1, dim1) - First feature map (decoder)
            feat2: (B, L2, dim2) - Second feature map (encoder skip)
            
        Returns:
            fused: (B, L1, out_dim) - Fused features matching feat1 spatial size
        """
        B, L1, dim1 = feat1.shape
        _, L2, dim2 = feat2.shape
        
        # Project both features to output dimension
        feat1_proj = self.proj1(feat1)  # (B, L1, out_dim)
        feat2_proj = self.proj2(feat2)  # (B, L2, out_dim)
        
        # Get spatial dimensions
        H1 = W1 = int(L1 ** 0.5)
        H2 = W2 = int(L2 ** 0.5)
        
        # Reshape to 2D spatial format
        feat1_2d = feat1_proj.view(B, H1, W1, self.out_dim)  # (B, H1, W1, C)
        feat2_2d = feat2_proj.view(B, H2, W2, self.out_dim)  # (B, H2, W2, C)
        
        # If spatial sizes differ, interpolate feat2 to match feat1
        if H1 != H2 or W1 != W2:
            feat2_2d = feat2_2d.permute(0, 3, 1, 2)  # (B, C, H2, W2)
            feat2_2d = torch.nn.functional.interpolate(
                feat2_2d, size=(H1, W1), mode='bilinear', align_corners=False
            )
            feat2_2d = feat2_2d.permute(0, 2, 3, 1)  # (B, H1, W1, C)
        
        # Permute for FFT: (B, H, W, C) -> (B, C, H, W)
        feat1_2d = feat1_2d.permute(0, 3, 1, 2)  # (B, C, H1, W1)
        feat2_2d = feat2_2d.permute(0, 3, 1, 2)  # (B, C, H1, W1)
        
        # Store original dtype for conversion back
        original_dtype = feat1_2d.dtype
        
        # Convert to float32 for FFT (cuFFT requires float32 for non-power-of-2 dimensions)
        # This is especially important when using mixed precision training
        feat1_2d_fp32 = feat1_2d.float()
        feat2_2d_fp32 = feat2_2d.float()
        
        # Transform to frequency domain (apply FFT across spatial dimensions)
        # rfft2 always operates on the last two dimensions (H, W) - no dim parameter needed
        feat1_fft = torch.fft.rfft2(feat1_2d_fp32, norm='ortho')  # (B, C, H1, W1//2+1)
        feat2_fft = torch.fft.rfft2(feat2_2d_fp32, norm='ortho')  # (B, C, H1, W1//2+1)
        
        # Extract magnitude and phase
        feat1_mag = torch.abs(feat1_fft)
        feat1_phase = torch.angle(feat1_fft)
        feat2_mag = torch.abs(feat2_fft)
        feat2_phase = torch.angle(feat2_fft)
        
        # Fuse in frequency domain (weighted average)
        # You can adjust the weights (0.5, 0.5) for different fusion strategies
        fused_mag = 0.5 * feat1_mag + 0.5 * feat2_mag
        fused_phase = 0.5 * feat1_phase + 0.5 * feat2_phase
        
        # Reconstruct complex representation
        fused_complex = fused_mag * torch.exp(1j * fused_phase)
        
        # Transform back to spatial domain
        # CRITICAL: Specify the output size (s parameter) to ensure correct dimensions
        # irfft2 always operates on the last two dimensions - no dim parameter needed
        fused_spatial = torch.fft.irfft2(fused_complex, s=(H1, W1), norm='ortho')
        
        # Convert back to original dtype (float16 if using mixed precision)
        fused_spatial = fused_spatial.to(original_dtype)
        
        # Permute back: (B, C, H1, W1) -> (B, H1, W1, C)
        fused_spatial = fused_spatial.permute(0, 2, 3, 1)
        
        # Reshape to tokens: (B, H1, W1, C) -> (B, L1, C)
        fused_tokens = fused_spatial.reshape(B, H1 * W1, self.out_dim)
        
        # Verify the shape is correct
        assert fused_tokens.shape == (B, L1, self.out_dim), \
            f"Shape mismatch: expected ({B}, {L1}, {self.out_dim}), got {fused_tokens.shape}"
        
        # Final projection and normalization
        output = self.fusion_proj(fused_tokens)
        output = self.norm(output)
        
        return output


# ============================================================================
# MSFA + MCT Hybrid Bottleneck Components
# ============================================================================
# Adapted from components.py to work with token-based bottleneck

def pair(t):
    """Helper function to ensure tuple format"""
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    """Pre-normalization for Transformer layers"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Feed-forward network for Transformer"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention for Transformer"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        from einops import rearrange
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer with PreNorm and residual connections"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MSFA(nn.Module):
    """
    Multi-Scale Feature Aggregation (MSFA) module.
    Adapted from MSAGHNet (components.py) for bottleneck processing.
    
    Architecture:
    - 5 branches: 4 atrous convolutions (dilations 1, 2, 4, 8) + 1 avg pool branch
    - Attention mechanism for atrous branches
    - Concat all branches + 1x1 conv fusion
    """
    def __init__(self, ch_in=768, out_channel=768):
        super(MSFA, self).__init__()

        depth = out_channel // 4
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.conv = nn.Conv2d(ch_in, depth, 1, 1)

        self.atrous_block1 = nn.Conv2d(ch_in, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(ch_in, depth, 3, 1, padding=1, dilation=1)
        self.atrous_block12 = nn.Conv2d(ch_in, depth, 3, 1, padding=2, dilation=2)
        self.atrous_block18 = nn.Conv2d(ch_in, depth, 3, 1, padding=4, dilation=4)
        self.atrous_block21 = nn.Conv2d(ch_in, depth, 3, 1, padding=8, dilation=8)
        self.attention = nn.Conv2d(depth * 4, 4, 1, padding=0, groups=4, bias=False)
        self.conv_1x1_output = nn.Conv2d(depth * 6, out_channel, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]

        p1 = self.pool(x)
        p1 = self.conv(p1)

        d0 = self.atrous_block1(x)
        d1 = self.atrous_block6(x)
        d2 = self.atrous_block12(x)
        d3 = self.atrous_block18(x)
        d4 = self.atrous_block21(x)
        
        # Attention mechanism: apply sigmoid attention to atrous branches
        att = torch.sigmoid(self.attention(torch.cat([d1, d2, d3, d4], 1)))
        d1 = d1 + d1 * att[:, 0].unsqueeze(1)
        d2 = d2 + d2 * att[:, 1].unsqueeze(1)
        d3 = d3 + d3 * att[:, 2].unsqueeze(1)
        d4 = d4 + d4 * att[:, 3].unsqueeze(1)

        net = self.conv_1x1_output(torch.cat([d0, d1, d2, d3, d4, p1], dim=1))
        out = self.relu(net)

        return out


class MCT(nn.Module):
    """
    Multi-scale Convolutional Transformer (MCT) module.
    Adapted from MSAGHNet (components.py) for bottleneck processing.
    
    Architecture:
    - Linear projection (patch embedding)
    - Positional embedding + CLS token
    - 12 stacked Transformer layers
    - Reshape back to spatial (no upsampling - maintains input size)
    
    Note: For bottleneck use, we don't do upsampling (unlike original MCT which upsamples).
    """
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=768, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        from einops import rearrange, repeat
        from einops.layers.torch import Rearrange
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Reshape back to spatial (no upsampling for bottleneck)
        self.out = Rearrange("b (h w) c->b c h w", h=image_height // patch_height, w=image_width // patch_width)

    def forward(self, img):
        from einops import repeat
        # Linear projection: convert spatial to tokens
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        # Process through Transformer
        x = self.transformer(x)

        # Remove CLS token
        output = x[:, 1:, :]
        # Reshape back to spatial (maintains input spatial size)
        output = self.out(output)

        return output


class MSFAMCTHybridBottleneck(nn.Module):
    """
    Hybrid MSFA + MCT Bottleneck for token-based processing.
    Adapted from MSAGHNet paper (Fig. 3).
    
    Architecture:
    1. Convert tokens to spatial format
    2. Process in parallel:
       - MSFA: Multi-scale atrous convolutions
       - MCT: Multi-scale Convolutional Transformer
    3. Concat both branches + ReLU
    4. Convert back to tokens
    
    Input: [B, L, C] tokens where L = (H/32) * (W/32), C = 768
    Output: [B, L, C] tokens
    """
    def __init__(self, img_size=224, dim=768, use_groupnorm=False):
        super().__init__()
        
        # Spatial size at bottleneck (H/32, W/32)
        spatial_size = img_size // 32
        
        # MSFA module (expects spatial input [B, C, H, W])
        self.msfa = MSFA(ch_in=dim, out_channel=dim)
        
        # MCT module (expects spatial input [B, C, H, W])
        # MCT uses depth=12, heads=16, mlp_dim=1024 (from MSAGHNet)
        # For bottleneck, use patch_size=1 (each pixel is a patch) since spatial_size is small (e.g., 7x7)
        # This ensures spatial_size is divisible by patch_size
        patch_size = 1 if spatial_size < 8 else 8  # Use patch_size=1 for small spatial sizes
        
        self.mct = MCT(
            image_size=spatial_size,
            patch_size=patch_size,
            dim=dim,
            depth=12,
            heads=16,
            mlp_dim=1024,
            channels=dim,
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.1
        )
        
        # Fusion: concat + ReLU
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, tokens):
        """
        Forward pass: tokens -> spatial -> MSFA + MCT -> spatial -> tokens
        
        Args:
            tokens: [B, L, C] where L = (H/32) * (W/32), C = 768
            
        Returns:
            tokens: [B, L, C] processed tokens
        """
        B, L, C = tokens.shape
        
        # Convert tokens to spatial format
        h = w = int(L ** 0.5)
        spatial = tokens.view(B, h, w, C)
        spatial = spatial.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Process in parallel: MSFA and MCT
        msfa_out = self.msfa(spatial)  # [B, C, H, W]
        mct_out = self.mct(spatial)    # [B, C, H, W]
        
        # Concat both branches
        concat = torch.cat([msfa_out, mct_out], dim=1)  # [B, 2*C, H, W]
        
        # Fusion: 1x1 conv + ReLU
        fused = self.fusion(concat)  # [B, C, H, W]
        
        # Convert back to tokens
        fused = fused.permute(0, 2, 3, 1)  # [B, H, W, C]
        tokens_out = fused.reshape(B, L, C)  # [B, L, C]
        
        return tokens_out


# ============================================================================
# SE-MSFE (Squeeze-and-Excitation Multi-Scale Feature Extraction) Components
# ============================================================================
# Adapted from components.py to work with EfficientNet encoder

class ChannelShuffle(nn.Module):
    """Channel Shuffle for SE-MSFE - adapted from components.py"""
    def __init__(self, groups=4):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        num_channels_per_group = num_channels // self.groups
        # Reshape input
        x = x.view(batch_size, self.groups, num_channels_per_group, height, width)
        # Transpose and reshape
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x


class SEModule(nn.Module):
    """Squeeze-and-Excitation Module for SE-MSFE - adapted from components.py"""
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class SE_MSFE(nn.Module):
    """
    SE-MSFE (Squeeze-and-Excitation Multi-Scale Feature Extraction) Module.
    Adapted from MSAGHNet paper (components.py) to replace standard conv operations in MBConv blocks.
    
    Architecture (from MSAGHNet paper Fig. 2):
    1. W_e: 3x3 conv + BN + ReLU
    2. Split to 5 branches:
       - con1: 1x1 conv (residual branch)
       - con3: 3x3 atrous conv (dilation=1) + 1x1
       - con5: 3x3 atrous conv (dilation=3) + 1x1
       - con7: 3x3 atrous conv (dilation=5) + 1x1
       - con9: 3x3 atrous conv (dilation=7) + 1x1
    3. Concat atrous branches (c3, c5, c7, c9) + channel shuffle
    4. Add residual (con1)
    5. SE block (global avg pool + FC-ReLU-FC-Sigmoid scale)
    """
    def __init__(self, ch_in, ch_out, use_groupnorm=False):
        super(SE_MSFE, self).__init__()
        
        # W_e: Initial projection (3x3 conv + BN + ReLU)
        norm_layer = get_norm_layer(ch_out, 'group') if use_groupnorm else nn.BatchNorm2d(ch_out)
        self.W_e = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            norm_layer,
            nn.ReLU(inplace=True)
        )
        
        # Residual branch (1x1 conv)
        norm_layer_res = get_norm_layer(ch_out, 'group') if use_groupnorm else nn.BatchNorm2d(ch_out)
        self.con1 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0),
            norm_layer_res,
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale atrous branches (4 branches with dilations 1, 3, 5, 7)
        # Each branch: 3x3 atrous conv + 1x1 conv
        norm_layer_3 = get_norm_layer(ch_out // 4, 'group') if use_groupnorm else nn.BatchNorm2d(ch_out // 4)
        self.con3 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out // 4, kernel_size=3, stride=1, padding=1, dilation=1),
            norm_layer_3,
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=1, stride=1),
            get_norm_layer(ch_out // 4, 'group') if use_groupnorm else nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(inplace=True)
        )
        
        norm_layer_5 = get_norm_layer(ch_out // 4, 'group') if use_groupnorm else nn.BatchNorm2d(ch_out // 4)
        self.con5 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out // 4, kernel_size=3, stride=1, padding=3, dilation=3),
            norm_layer_5,
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=1, stride=1),
            get_norm_layer(ch_out // 4, 'group') if use_groupnorm else nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(inplace=True)
        )
        
        norm_layer_7 = get_norm_layer(ch_out // 4, 'group') if use_groupnorm else nn.BatchNorm2d(ch_out // 4)
        self.con7 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out // 4, kernel_size=3, stride=1, padding=5, dilation=5),
            norm_layer_7,
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=1, stride=1),
            get_norm_layer(ch_out // 4, 'group') if use_groupnorm else nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(inplace=True)
        )
        
        norm_layer_9 = get_norm_layer(ch_out // 4, 'group') if use_groupnorm else nn.BatchNorm2d(ch_out // 4)
        self.con9 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out // 4, kernel_size=3, stride=1, padding=7, dilation=7),
            norm_layer_9,
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=1, stride=1),
            get_norm_layer(ch_out // 4, 'group') if use_groupnorm else nn.BatchNorm2d(ch_out // 4),
            nn.ReLU(inplace=True)
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.shuffle = ChannelShuffle(4)
        self.se = SEModule(ch_out)

    def forward(self, x1):
        x1 = self.W_e(x1)
        c1 = self.con1(x1)  # Residual branch
        c3 = self.con3(x1)  # Dilation 1
        c5 = self.con5(x1)  # Dilation 3
        c7 = self.con7(x1)  # Dilation 5
        c9 = self.con9(x1)  # Dilation 7
        # Concat atrous branches, shuffle, add residual, then SE
        out = self.se(self.shuffle(torch.cat([c3, c5, c7, c9], dim=1)) + c1)
        return out


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet encoder with feature extraction at multiple scales using timm.
    Supports optional SE-MSFE replacement of MBConv block conv operations.
    
    NOTE: EfficientNet-B4 Architecture Comparison:
    - Network Model Encoder: Uses actual EfficientNet-B4 architecture from timm
      (tf_efficientnet_b4_ns with MBConv blocks, depthwise separable convolutions, etc.)
      This is the REAL EfficientNet-B4 encoder architecture.
    
    - Hybrid2 Model Decoder: Uses "EfficientNet-B4" channel configuration but NOT the architecture.
      The decoder uses SimpleDecoderBlock (Conv2d + BatchNorm + ReLU), which is a simple CNN decoder.
      The "B4" name refers to the channel configuration (decoder_channels=[256, 128, 64, 32]),
      not the EfficientNet architecture itself.
    
    They are NOT the same - Network uses real EfficientNet-B4 encoder, Hybrid2 uses simple CNN decoder
    with EfficientNet-B4 channel configuration.
    """
    def __init__(self, model_name='tf_efficientnet_b4_ns', pretrained=True, freeze_bn=False, 
                 use_se_msfe=False, use_groupnorm=False):
        super().__init__()
        
        self.use_se_msfe = use_se_msfe
        self.use_groupnorm = use_groupnorm
        
        # Load pretrained EfficientNet using timm
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=(1, 2, 3, 4)
        )
        
        # Get channel dimensions for different stages from timm
        self.stage_channels = self.backbone.feature_info.channels()
        
        # Replace MBConv blocks with SE-MSFE if requested
        if use_se_msfe:
            self._replace_mbconv_with_se_msfe()
        
        # Freeze batch normalization layers if specified
        if freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.track_running_stats = False
    
    def _replace_mbconv_with_se_msfe(self):
        """
        Replace MBConv block conv operations with SE-MSFE modules.
        This modifies the encoder in-place to use SE-MSFE instead of standard MBConv operations.
        
        Strategy: Replace the conv_dw (depthwise conv) in each DepthwiseSeparableConv module
        with SE-MSFE, which provides multi-scale feature extraction with atrous convolutions.
        
        Note: timm's EfficientNet uses DepthwiseSeparableConv which has conv_dw and conv_pw.
        """
        # Recursively find all DepthwiseSeparableConv modules (timm's MBConv equivalent)
        def find_depthwise_separable_convs(module, conv_modules):
            """Recursively find all DepthwiseSeparableConv modules"""
            module_type = type(module).__name__
            if 'DepthwiseSeparableConv' in module_type and hasattr(module, 'conv_dw'):
                conv_modules.append(module)
            for child in module.children():
                find_depthwise_separable_convs(child, conv_modules)
        
        depthwise_convs = []
        find_depthwise_separable_convs(self.backbone, depthwise_convs)
        
        if len(depthwise_convs) == 0:
            print("⚠️  Warning: No DepthwiseSeparableConv modules found in EfficientNet. SE-MSFE not applied.")
            return
        
        # Replace each DepthwiseSeparableConv's conv_dw with SE-MSFE
        # Structure: conv_dw (depthwise) -> conv_pw (pointwise)
        replaced_count = 0
        for conv_module in depthwise_convs:
            # Get channels from conv_dw
            conv_dw = conv_module.conv_dw
            if isinstance(conv_dw, nn.Conv2d):
                in_channels = conv_dw.in_channels
                out_channels = conv_dw.out_channels
                original_stride = conv_dw.stride[0] if isinstance(conv_dw.stride, (tuple, list)) else conv_dw.stride
            elif isinstance(conv_dw, nn.Sequential):
                # Find the conv layer in Sequential
                for layer in conv_dw:
                    if isinstance(layer, nn.Conv2d):
                        in_channels = layer.in_channels
                        out_channels = layer.out_channels
                        original_stride = layer.stride[0] if isinstance(layer.stride, (tuple, list)) else layer.stride
                        break
                else:
                    continue
            else:
                continue
            
            # Create SE-MSFE module to replace conv_dw
            # SE-MSFE maintains same input/output channels
            se_msfe = SE_MSFE(ch_in=in_channels, ch_out=out_channels, 
                             use_groupnorm=self.use_groupnorm)
            
            # Handle stride: SE-MSFE W_e needs to match original stride for downsampling
            if original_stride > 1:
                # Modify W_e's first conv to have the correct stride
                se_msfe.W_e[0].stride = (original_stride, original_stride)
                # Adjust padding for stride > 1 (maintain output size)
                if original_stride == 2:
                    se_msfe.W_e[0].padding = (1, 1)
            
            # Replace conv_dw with SE-MSFE
            conv_module.conv_dw = se_msfe
            replaced_count += 1
        
        if replaced_count > 0:
            print(f"✓ Replaced {replaced_count} DepthwiseSeparableConv.conv_dw operations with SE-MSFE")
        else:
            print("⚠️  Warning: No DepthwiseSeparableConv modules were replaced. SE-MSFE may not be applied correctly.")
    
    def forward(self, x):
        # Extract features at different stages
        features = self.backbone(x)
        return features  # Return 4 feature maps


class ResNet50Encoder(nn.Module):
    """
    ResNet-50 encoder with feature extraction at multiple scales using timm.
    
    ResNet-50 is a standard CNN encoder that provides good feature representations
    for segmentation tasks. It uses residual connections and has been widely used
    in computer vision tasks.
    """
    def __init__(self, model_name='resnet50', pretrained=True, freeze_bn=False):
        super().__init__()
        
        # Load pretrained ResNet-50 using timm
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=(1, 2, 3, 4)
        )
        
        # Freeze batch normalization layers if specified
        if freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.track_running_stats = False
        
        # Get channel dimensions for different stages from timm
        self.stage_channels = self.backbone.feature_info.channels()
    
    def forward(self, x):
        # Extract features at different stages
        features = self.backbone(x)
        return features  # Return 4 feature maps


class SmartSkipConnectionTransformer(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.align = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.GELU()
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=decoder_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(decoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.fuse = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim * 4),
            nn.LayerNorm(decoder_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim * 4, decoder_dim),
            nn.LayerNorm(decoder_dim)
        )

    def forward(self, encoder_tokens, decoder_tokens):
        B_enc, L_enc, _ = encoder_tokens.shape
        B_dec, L_dec, dec_dim = decoder_tokens.shape
        skip_tokens = self.align(encoder_tokens)
        skip_enhanced, _ = self.attention(skip_tokens, skip_tokens, skip_tokens)
        skip_tokens = self.norm_attn(skip_tokens + self.dropout(skip_enhanced))
        if L_enc != L_dec:
            h_enc = w_enc = int(L_enc ** 0.5)
            h_dec = w_dec = int(L_dec ** 0.5)
            skip_tokens_2d = skip_tokens.view(B_enc, h_enc, w_enc, dec_dim).permute(0, 3, 1, 2)
            skip_tokens_2d = torch.nn.functional.interpolate(skip_tokens_2d, size=(h_dec, w_dec), mode='bilinear', align_corners=False)
            skip_tokens = skip_tokens_2d.permute(0, 2, 3, 1).reshape(B_dec, L_dec, dec_dim)
        fused = torch.cat([decoder_tokens, skip_tokens], dim=-1)
        fused_output = self.fuse(fused)
        # Add residual connection for stable gradient flow (ResNet/Transformer style)
        return decoder_tokens + fused_output


class SimpleSkipConnectionTransformer(nn.Module):
    """
    Simple skip connection for transformer tokens: Project encoder tokens and concatenate.
    Matches hybrid2's SimpleSkipConnection pattern but works with tokens instead of CNN features.
    
    Pattern (matching hybrid2):
    1. Project encoder tokens: Linear → LayerNorm → ReLU
    2. Concatenate: [decoder_tokens, encoder_proj_tokens]
    3. Fuse: Linear (equivalent to Conv3x3) → LayerNorm → ReLU
    """
    def __init__(self, encoder_dim, decoder_dim, use_groupnorm=False):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        # Project encoder tokens to decoder dimension (equivalent to Conv1x1 in hybrid2)
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim, bias=False),
            nn.LayerNorm(decoder_dim),
            nn.ReLU(inplace=True)
        )
        
        # Fuse concatenated tokens (equivalent to Conv3x3 in hybrid2)
        self.fuse = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim, bias=False),
            nn.LayerNorm(decoder_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, encoder_tokens, decoder_tokens):
        """
        Args:
            encoder_tokens: (B, L_enc, encoder_dim) - Encoder tokens
            decoder_tokens: (B, L_dec, decoder_dim) - Decoder tokens
            
        Returns:
            fused: (B, L_dec, decoder_dim) - Fused tokens
        """
        B_enc, L_enc, enc_dim = encoder_tokens.shape
        B_dec, L_dec, dec_dim = decoder_tokens.shape
        
        # Project encoder tokens to decoder dimension
        encoder_proj = self.proj(encoder_tokens)  # (B_enc, L_enc, decoder_dim)
        
        # Handle spatial size mismatch (if encoder and decoder have different resolutions)
        if L_enc != L_dec:
            h_enc = w_enc = int(L_enc ** 0.5)
            h_dec = w_dec = int(L_dec ** 0.5)
            
            # Reshape to spatial format
            encoder_proj_2d = encoder_proj.view(B_enc, h_enc, w_enc, dec_dim)
            encoder_proj_2d = encoder_proj_2d.permute(0, 3, 1, 2)  # (B, C, H, W)
            
            # Interpolate to match decoder spatial size
            encoder_proj_2d = torch.nn.functional.interpolate(
                encoder_proj_2d, size=(h_dec, w_dec), mode='bilinear', align_corners=False
            )
            
            # Reshape back to tokens
            encoder_proj_2d = encoder_proj_2d.permute(0, 2, 3, 1)  # (B, H, W, C)
            encoder_proj = encoder_proj_2d.reshape(B_dec, L_dec, dec_dim)
        
        # Concatenate along feature dimension (equivalent to channel dim in CNN)
        fused = torch.cat([decoder_tokens, encoder_proj], dim=-1)  # (B, L_dec, decoder_dim * 2)
        
        # Fuse with Linear → LayerNorm → ReLU (equivalent to Conv3x3 → Norm → ReLU)
        fused = self.fuse(fused)  # (B, L_dec, decoder_dim)
        
        return fused


# ============================================================================
# GCFF (Global Context Feature Fusion) Module Components
# ============================================================================
# These are adapted from components.py to work with the token-based decoder

class ContextBlock(nn.Module):
    """Global Context Block for GCFF - adapted from components.py"""
    def __init__(self, inplanes, ratio, pooling_type='att',
                 fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class CAMLayer(nn.Module):
    """Channel Attention Module for GCFF - adapted from components.py"""
    def __init__(self, channel, reduction=8):
        super(CAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = x
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        channel_x = channel_out * x

        return channel_x


class GCFFSkipConnectionTransformer(nn.Module):
    """
    GCFF (Global Context Feature Fusion) Skip Connection for transformer tokens.
    Adapted from MSAGHNet's GCFF_block (components.py) to work with token-based decoder.
    
    Architecture (from MSAGHNet paper Fig. 4):
    1. Project encoder (W_e: 1x1 conv) and decoder (W_d: Upsample + 1x1 conv) features
    2. Add + ReLU → F
    3. Global context path: ContextBlock (attention pooling + MLP)
    4. Channel attention path: CAMLayer (max/avg pool + MLP)
    5. Concat both paths → psi (1x1 conv + ReLU)
    6. Add back to encoder: out = psi(...) + e1
    
    This module handles token-to-spatial conversion internally.
    """
    def __init__(self, encoder_dim, decoder_dim, ch_out, use_groupnorm=False):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.ch_out = ch_out
        
        # W_e: Project encoder tokens to output dimension (1x1 conv equivalent)
        # Convert to spatial, apply conv, convert back
        self.W_e = nn.Sequential(
            nn.Conv2d(encoder_dim, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out) if not use_groupnorm else get_norm_layer(ch_out, 'group')
        )
        
        # W_d: Project decoder tokens (1x1 conv)
        # Note: We handle upsampling separately to match encoder resolution
        self.W_d_conv = nn.Conv2d(decoder_dim, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_d_norm = nn.BatchNorm2d(ch_out) if not use_groupnorm else get_norm_layer(ch_out, 'group')
        
        self.relu = nn.ReLU(inplace=True)
        
        # Global Context Block (attention-based pooling + MLP)
        self.gc = ContextBlock(inplanes=ch_out, ratio=1. / 8., pooling_type='att')
        
        # Channel Attention Module
        self.ca = CAMLayer(ch_out)
        
        # Psi: Final fusion (1x1 conv + ReLU)
        self.psi = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
    
    def forward(self, encoder_tokens, decoder_tokens):
        """
        GCFF forward pass: Fuses encoder feature Ei with upsampled decoder feature Di-1.
        
        Args:
            encoder_tokens: (B, L_enc, encoder_dim) - Encoder tokens (Ei) at resolution H/2^i
            decoder_tokens: (B, L_dec, decoder_dim) - Decoder tokens (Di-1) at resolution H/2^(i-1), already upsampled
            
        Returns:
            fused: (B, L_enc, ch_out) - Fused tokens at encoder resolution (H/2^i)
        """
        B_enc, L_enc, enc_dim = encoder_tokens.shape
        B_dec, L_dec, dec_dim = decoder_tokens.shape
        
        # Convert encoder tokens to spatial format
        h_enc = w_enc = int(L_enc ** 0.5)
        encoder_spatial = encoder_tokens.view(B_enc, h_enc, w_enc, enc_dim)
        encoder_spatial = encoder_spatial.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Convert decoder tokens to spatial format
        h_dec = w_dec = int(L_dec ** 0.5)
        decoder_spatial = decoder_tokens.view(B_dec, h_dec, w_dec, dec_dim)
        decoder_spatial = decoder_spatial.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Apply GCFF in spatial domain (following MSAGHNet GCFF_block from components.py)
        # Step 1: Project encoder (W_e: 1x1 conv)
        e1 = self.W_e(encoder_spatial)  # (B, ch_out, H_enc, W_enc)
        
        # Step 1b: Handle decoder - if decoder is at higher resolution, downsample first
        # Then project (W_d: 1x1 conv) - matching original GCFF_block where W_d upsamples
        # but in our case decoder is already upsampled, so we just need to match encoder size
        if h_dec != h_enc or w_dec != w_enc:
            # Decoder is at higher resolution, downsample to encoder resolution
            decoder_spatial = F.interpolate(
                decoder_spatial, size=(h_enc, w_enc), mode='bilinear', align_corners=False
            )
        
        # Project decoder features
        d1 = self.W_d_conv(decoder_spatial)  # (B, ch_out, H_enc, W_enc)
        d1 = self.W_d_norm(d1)
        
        # Step 2: Add + ReLU → F
        x1 = self.relu(e1 + d1)  # (B, ch_out, H, W)
        
        # Step 3: Global context path (ContextBlock: attention pooling + MLP)
        gc_out = self.gc(x1)  # (B, ch_out, H, W)
        
        # Step 4: Channel attention path (CAMLayer: max/avg pool + MLP)
        ca_out = self.ca(x1)  # (B, ch_out, H, W)
        
        # Step 5: Combine both paths → psi (following components.py line 337: gc(x1) + ca(x1))
        combined = gc_out + ca_out  # (B, ch_out, H, W)
        psi_out = self.psi(combined)  # (B, ch_out, H, W)
        
        # Step 6: Add back to encoder feature (residual connection)
        out = psi_out + e1  # (B, ch_out, H, W)
        
        # Convert back to token format (at encoder resolution)
        out = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        out = out.reshape(B_enc, L_enc, self.ch_out)  # (B, L_enc, ch_out)
        
        return out


class EfficientNetSwinUNet(nn.Module):
    """
    Hybrid CNN-Transformer UNet with EfficientNet or ResNet-50 encoder and Swin Transformer decoder
    """
    def __init__(self, img_size=224, num_classes=6, efficientnet_model='tf_efficientnet_b4_ns',
                 encoder_type='efficientnet', pretrained=True, embed_dim=96, depths_decoder=[2, 2, 2, 2], 
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, use_checkpoint=False,
                 use_deep_supervision=False, fusion_method='simple', use_bottleneck=False, 
                 adapter_mode='external', use_multiscale_agg=False, use_groupnorm=False,
                 use_se_msfe=False, use_msfa_mct_bottleneck=False):
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.use_deep_supervision = use_deep_supervision
        self.fusion_method = fusion_method  # 'simple' or 'fourier' or 'smart'
        self.use_bottleneck = use_bottleneck
        self.adapter_mode = adapter_mode  # 'external' or 'streaming'
        self.use_multiscale_agg = use_multiscale_agg
        self.use_groupnorm = use_groupnorm
        self.encoder_type = encoder_type  # 'efficientnet' or 'resnet50'
        self.use_se_msfe = use_se_msfe
        self.use_msfa_mct_bottleneck = use_msfa_mct_bottleneck
        
        # Encoder selection: EfficientNet or ResNet-50
        if encoder_type == 'resnet50':
            self.encoder = ResNet50Encoder(model_name='resnet50', pretrained=pretrained)
        else:
            self.encoder = EfficientNetEncoder(
                model_name=efficientnet_model, 
                pretrained=pretrained,
                use_se_msfe=use_se_msfe,
                use_groupnorm=use_groupnorm
            )
        
        # Target dimensions for decoder
        # Stage resolutions: 56x56, 28x28, 14x14, 7x7
        # Channel dimensions calculated from embed_dim: [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
        # Swin Tiny (default): [96, 192, 384, 768]
        # Swin Base (SimMIM): [128, 256, 512, 1024]
        self.decoder_dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        
        # Feature adapters (external mode) to convert CNN features to transformer format
        encoder_spatial_sizes = [img_size // 4, img_size // 8, img_size // 16, img_size // 32]
        if self.adapter_mode == 'external':
            self.feature_adapters = nn.ModuleList([
                CNNFeatureAdapter(
                    in_channels=self.encoder.stage_channels[i],
                    out_channels=self.decoder_dims[i],
                    spatial_size=encoder_spatial_sizes[i],
                    use_groupnorm=use_groupnorm  # Pass GroupNorm flag
                ) for i in range(4)
            ])
        else:
            self.feature_adapters = None  # streaming mode handles adaptation inline
            # Register streaming adapters (Hybrid1-like): 1x1 conv per stage + GELU (like hybrid1)
            if use_groupnorm:
                self.streaming_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(self.encoder.stage_channels[i], self.decoder_dims[i], kernel_size=1, bias=False),
                        get_norm_layer(self.decoder_dims[i], 'group'),
                        nn.GELU()
                    ) for i in range(4)
                ])
            else:
                self.streaming_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(self.encoder.stage_channels[i], self.decoder_dims[i], kernel_size=1, bias=False),
                        nn.GELU()
                    ) for i in range(4)
                ])
        
        # Bottleneck processing
        # Aligned to SwinUnet: dim=768, heads=24, BasicLayer, stochastic drop_path
        bottleneck_dim = self.decoder_dims[-1]  # 768 (matches SwinUnet encoder stage 4)
        bottleneck_num_heads = num_heads[-1]  # 24 (matches SwinUnet encoder stage 4)
        
        self.norm = norm_layer(self.decoder_dims[-1])
        if self.use_bottleneck:
            if self.use_msfa_mct_bottleneck:
                # MSFA + MCT Hybrid Bottleneck (from MSAGHNet)
                self.bottleneck_layer = MSFAMCTHybridBottleneck(
                    img_size=img_size,
                    dim=bottleneck_dim,
                    use_groupnorm=use_groupnorm
                )
                print("🚀 Bottleneck: MSFA + MCT Hybrid (from MSAGHNet)")
            else:
                # Standard 2 Swin Transformer blocks
                # Calculate stochastic drop_path using decoder depths for consistency
                # The bottleneck is part of the decoder path, so use decoder depths
                # dpr = linspace(0, drop_path_rate, sum(depths_decoder))
                dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))]
                # Bottleneck uses the last 2 blocks (since bottleneck depth=2)
                bottleneck_drop_path = dpr[-2:]
                
                # 2 Swin blocks at 7x7 with dim=768, heads=24 (matching SwinUnet)
                self.bottleneck_layer = BasicLayer(
                    dim=bottleneck_dim,
                    input_resolution=(img_size // 32, img_size // 32),
                    depth=2,
                    num_heads=bottleneck_num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=bottleneck_drop_path,  # Stochastic depth matching SwinUnet
                    norm_layer=norm_layer,
                    downsample=None,  # No downsampling in bottleneck
                    use_checkpoint=use_checkpoint
                )
            
            # Multi-Scale Aggregation support
            if use_multiscale_agg:
                # Projection layers to combine multi-scale encoder features
                # Project all 4 scales to bottleneck dim (768, matching SwinUnet)
                self.multiscale_proj = nn.ModuleList([
                    nn.Linear(self.decoder_dims[i], bottleneck_dim)
                    for i in range(4)  # Project all 4 scales to bottleneck dim (768)
                ])
                self.multiscale_fusion = nn.Linear(bottleneck_dim * 4, bottleneck_dim)
                print("🚀 Multi-Scale Aggregation enabled in bottleneck")
        
        # Fusion method for decoder
        if fusion_method == 'fourier':
            # Fourier fusion for skip connections
            # After layer 0: x=[384] decoder + C3=[384] encoder → [384]
            # After layer 1: x=[192] decoder + C2=[192] encoder → [192]
            # After layer 2: x=[96] decoder + C1=[96] encoder → [96]
            self.skip_fusions = nn.ModuleList([
                FourierFeatureFusion(
                    in_dim1=self.decoder_dims[2],  # Layer 1 input dimension
                    in_dim2=self.decoder_dims[2],  # C3 encoder dimension
                    out_dim=self.decoder_dims[2]   # Layer 1 output dimension
                ),
                FourierFeatureFusion(
                    in_dim1=self.decoder_dims[1],  # Layer 2 input dimension
                    in_dim2=self.decoder_dims[1],  # C2 encoder dimension
                    out_dim=self.decoder_dims[1]   # Layer 2 output dimension
                ),
                FourierFeatureFusion(
                    in_dim1=self.decoder_dims[0],   # Layer 3 input dimension
                    in_dim2=self.decoder_dims[0],   # C1 encoder dimension
                    out_dim=self.decoder_dims[0]    # Layer 3 output dimension
                )
            ])
        elif fusion_method == 'smart':
            # Attention-based smart skip connections (like hybrid1)
            self.smart_skips = nn.ModuleList([
                SmartSkipConnectionTransformer(encoder_dim=self.decoder_dims[2], decoder_dim=self.decoder_dims[2], num_heads=num_heads[2]),
                SmartSkipConnectionTransformer(encoder_dim=self.decoder_dims[1], decoder_dim=self.decoder_dims[1], num_heads=num_heads[1]),
                SmartSkipConnectionTransformer(encoder_dim=self.decoder_dims[0], decoder_dim=self.decoder_dims[0], num_heads=num_heads[0]),
            ])
            self.skip_fusions = None
        elif fusion_method == 'simple':
            # Simple skip connections matching hybrid2's SimpleSkipConnection pattern
            self.simple_skips = nn.ModuleList([
                SimpleSkipConnectionTransformer(encoder_dim=self.decoder_dims[2], decoder_dim=self.decoder_dims[2]),
                SimpleSkipConnectionTransformer(encoder_dim=self.decoder_dims[1], decoder_dim=self.decoder_dims[1]),
                SimpleSkipConnectionTransformer(encoder_dim=self.decoder_dims[0], decoder_dim=self.decoder_dims[0]),
            ])
            self.skip_fusions = None
            self.smart_skips = None
            self.gcff_skips = None
        elif fusion_method == 'gcff':
            # GCFF (Global Context Feature Fusion) skip connections from MSAGHNet
            # ch_out matches decoder_dim for each stage
            self.gcff_skips = nn.ModuleList([
                GCFFSkipConnectionTransformer(encoder_dim=self.decoder_dims[2], decoder_dim=self.decoder_dims[2], ch_out=self.decoder_dims[2], use_groupnorm=use_groupnorm),
                GCFFSkipConnectionTransformer(encoder_dim=self.decoder_dims[1], decoder_dim=self.decoder_dims[1], ch_out=self.decoder_dims[1], use_groupnorm=use_groupnorm),
                GCFFSkipConnectionTransformer(encoder_dim=self.decoder_dims[0], decoder_dim=self.decoder_dims[0], ch_out=self.decoder_dims[0], use_groupnorm=use_groupnorm),
            ])
            self.skip_fusions = None
            self.smart_skips = None
            self.simple_skips = None
        else:
            # Default: no special fusion (fallback to original simple concat)
            self.skip_fusions = None
            self.smart_skips = None
            self.simple_skips = None
            self.gcff_skips = None
        
        # Swin Transformer Decoder
        self.num_layers = 4
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(
                2 * self.decoder_dims[self.num_layers - 1 - i_layer],
                self.decoder_dims[self.num_layers - 1 - i_layer]
            ) if i_layer > 0 else nn.Identity()
            
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(img_size // (2 ** (self.num_layers + 1 - i_layer)),
                                     img_size // (2 ** (self.num_layers + 1 - i_layer))),
                    dim=self.decoder_dims[self.num_layers - 1 - i_layer],
                    dim_scale=2,
                    norm_layer=norm_layer
                )
            else:
                layer_up = BasicLayer_up(
                    dim=self.decoder_dims[self.num_layers - 1 - i_layer],
                    input_resolution=(img_size // (2 ** (self.num_layers + 1 - i_layer)),
                                     img_size // (2 ** (self.num_layers + 1 - i_layer))),
                    depth=depths_decoder[self.num_layers - 1 - i_layer],
                    num_heads=num_heads[self.num_layers - 1 - i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.,
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint
                )
            
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        
        self.norm_up = norm_layer(self.embed_dim)
        
        # Final upsampling and segmentation head
        self.up = FinalPatchExpand_X4(
            input_resolution=(img_size // 4, img_size // 4),
            dim_scale=4,
            dim=embed_dim
        )
        
        # Projection layer to reduce channels from embed_dim to 64 (matching Hybrid2's decoder4)
        # This matches Hybrid2 baseline: decoder4 reduces channels to 64 before seg_head
        if use_groupnorm:
            self.final_proj = nn.Sequential(
                nn.Conv2d(in_channels=embed_dim, out_channels=64, kernel_size=3, padding=1, bias=False),
                get_norm_layer(64, 'group'),
                nn.ReLU(inplace=True)
            )
        else:
            self.final_proj = nn.Sequential(
                nn.Conv2d(in_channels=embed_dim, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        
        # Segmentation head - Baseline style (exactly matching hybrid2 baseline)
        if use_groupnorm:
            # Baseline segmentation head with GroupNorm (matching Hybrid2 baseline)
            self.output = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1, bias=False),  # 64 -> 64 (matching Hybrid2)
                get_norm_layer(64, 'group'),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(64, num_classes, 1)  # bias=True by default (matching Hybrid2)
            )
        else:
            # Original segmentation head (fallback for non-groupnorm mode)
            self.output = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(64, num_classes, 1)
            )
        
        # Deep Supervision: Auxiliary heads for intermediate outputs
        # Using MSAGHNet-style multi-resolution deep supervision (simple OutConv, no upsampling)
        if use_deep_supervision:
            # Stages 1, 2, 3 have dimensions: [384, 192, 96]
            aux_dims = [int(embed_dim * 2 ** (3-i)) for i in range(1, 4)]  # [384, 192, 96]
            # Use simple OutConv heads (matching MSAGHNet style: single Conv2d, no BN/ReLU)
            # This keeps outputs at native resolutions without upsampling
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(dim, num_classes, 3, padding=1) for dim in aux_dims
            ])
            print("🚀 Deep Supervision enabled: 3 auxiliary outputs (MSAGHNet-style multi-resolution)")
            print(f"   Aux dims: {aux_dims}")
            print("   Style: Simple OutConv (single Conv2d), outputs at native resolutions (H/16, H/8, H/4)")
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def freeze_encoder(self):
        """Freeze encoder parameters for training decoder first"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Handle both adapter types
        if self.adapter_mode == 'external':
            if hasattr(self, 'feature_adapters') and self.feature_adapters is not None:
                for param in self.feature_adapters.parameters():
                    param.requires_grad = False
        elif self.adapter_mode == 'streaming':
            if hasattr(self, 'streaming_proj') and self.streaming_proj is not None:
                for param in self.streaming_proj.parameters():
                    param.requires_grad = False
        
        print("Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        # Handle both adapter types
        if self.adapter_mode == 'external':
            if hasattr(self, 'feature_adapters') and self.feature_adapters is not None:
                for param in self.feature_adapters.parameters():
                    param.requires_grad = True
        elif self.adapter_mode == 'streaming':
            if hasattr(self, 'streaming_proj') and self.streaming_proj is not None:
                for param in self.streaming_proj.parameters():
                    param.requires_grad = True
        
        print("Encoder unfrozen")
    
    def unfreeze_layer_by_layer(self, layer_idx):
        """Unfreeze encoder layer by layer for progressive training"""
        # EfficientNet has blocks, unfreeze from top to bottom
        blocks = list(self.encoder.backbone._blocks)
        total_blocks = len(blocks)
        blocks_per_layer = total_blocks // 4
        
        start_idx = max(0, total_blocks - (layer_idx + 1) * blocks_per_layer)
        end_idx = total_blocks - layer_idx * blocks_per_layer
        
        for idx in range(start_idx, end_idx):
            for param in blocks[idx].parameters():
                param.requires_grad = True
        
        print(f"Unfroze encoder blocks {start_idx} to {end_idx-1}")
    
    def forward(self, x):
        # Handle single channel input
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Extract features from EfficientNet encoder
        encoder_features = self.encoder(x)  # List of 4 feature maps
        
        # Adapt features to transformer tokens
        adapted_features = []
        if self.adapter_mode == 'external':
            for i, feat in enumerate(encoder_features):
                adapted = self.feature_adapters[i](feat)
                adapted_features.append(adapted)
        else:
            # Streaming-style adaptation (Hybrid1-like): Conv1x1 + GELU, then tokenization
            for i, feat in enumerate(encoder_features):
                y = self.streaming_proj[i](feat)  # (B, target_dim, H, W) - Conv1x1 + GELU
                y = y.flatten(2).transpose(1, 2)  # (B, H*W, target_dim) - Tokenization
                adapted_features.append(y)
        
        # Bottleneck with optional multi-scale aggregation
        x = self.norm(adapted_features[-1])
        if self.use_bottleneck:
            if self.use_multiscale_agg:
                # Multi-scale aggregation matching SwinUnet pattern
                import torch.nn.functional as F
                
                B, L, C = x.shape  # C is bottleneck_dim (768)
                h = w = int(L ** 0.5)
                
                # Project and resize all features to bottleneck size
                projected = []
                for i, feat in enumerate(adapted_features):
                    # feat: [B, L_i, C_i]
                    B_f, L_f, C_f = feat.shape
                    h_f = w_f = int(L_f ** 0.5)
                    
                    # Project to bottleneck dim (768)
                    proj_feat = self.multiscale_proj[i](feat)  # [B, L_i, bottleneck_dim]
                    
                    # Reshape and resize to bottleneck spatial size
                    proj_feat = proj_feat.view(B_f, h_f, w_f, C)  # C is bottleneck_dim (768)
                    proj_feat = proj_feat.permute(0, 3, 1, 2)  # [B, C, H, W]
                    proj_feat = F.interpolate(proj_feat, size=(h, w), mode='bilinear', align_corners=False)
                    proj_feat = proj_feat.permute(0, 2, 3, 1)  # [B, H, W, C] - interpolate output is already contiguous
                    proj_feat = proj_feat.reshape(B, -1, C)  # [B, L, bottleneck_dim] - reshape handles non-contiguous
                    
                    projected.append(proj_feat)
                
                # Concatenate along feature dimension
                aggregated = torch.cat(projected, dim=-1)  # [B, L, bottleneck_dim*4]
                fused = self.multiscale_fusion(aggregated)  # [B, L, bottleneck_dim]
                
                # Add to original features (residual connection)
                x = x + fused
            
            x = self.bottleneck_layer(x)  # [B, L, 768] - aligned to SwinUnet, no projection needed
        
        # Decoder with skip connections
        aux_features = [] if self.use_deep_supervision else None
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                # Skip connection: fuse decoder with encoder features
                skip_features = adapted_features[3 - inx]
                
                if self.fusion_method == 'fourier':
                    # Use Fourier feature fusion
                    x = self.skip_fusions[inx - 1](x, skip_features)
                elif self.fusion_method == 'smart' and self.smart_skips is not None:
                    x = self.smart_skips[inx - 1](skip_features, x)
                elif self.fusion_method == 'simple' and self.simple_skips is not None:
                    # Use simple skip connection matching hybrid2 pattern
                    x = self.simple_skips[inx - 1](skip_features, x)
                elif self.fusion_method == 'gcff' and self.gcff_skips is not None:
                    # Use GCFF (Global Context Feature Fusion) from MSAGHNet
                    # GCFF fuses encoder (Ei) and decoder (Di-1) features
                    # Output is at encoder resolution, need to upsample to decoder resolution
                    gcff_out = self.gcff_skips[inx - 1](skip_features, x)
                    
                    # GCFF outputs at encoder resolution, but decoder needs its own resolution
                    # Upsample GCFF output to match decoder resolution
                    B, L_enc, ch_out = gcff_out.shape
                    _, L_dec, dec_dim = x.shape
                    
                    if L_enc != L_dec:
                        # Reshape to spatial and upsample
                        h_enc = w_enc = int(L_enc ** 0.5)
                        h_dec = w_dec = int(L_dec ** 0.5)
                        gcff_spatial = gcff_out.view(B, h_enc, w_enc, ch_out).permute(0, 3, 1, 2)
                        gcff_spatial = F.interpolate(
                            gcff_spatial, size=(h_dec, w_dec), mode='bilinear', align_corners=False
                        )
                        gcff_out = gcff_spatial.permute(0, 2, 3, 1).reshape(B, L_dec, ch_out)
                    
                    # Project to decoder dimension if needed (ch_out should equal dec_dim, but handle mismatch)
                    if ch_out != dec_dim:
                        # Create a simple linear projection layer on-the-fly
                        proj_weight = torch.eye(min(dec_dim, ch_out), device=gcff_out.device, dtype=gcff_out.dtype)
                        if dec_dim > ch_out:
                            # Pad with zeros
                            padding = torch.zeros(dec_dim - ch_out, ch_out, device=gcff_out.device, dtype=gcff_out.dtype)
                            proj_weight = torch.cat([proj_weight, padding], dim=0)
                        elif ch_out > dec_dim:
                            # Truncate
                            proj_weight = proj_weight[:dec_dim, :]
                        gcff_out = F.linear(gcff_out, weight=proj_weight)
                    
                    x = gcff_out
                else:
                    # Fallback: Simple concatenation fusion (original)
                    x = torch.cat([x, skip_features], -1)
                    x = self.concat_back_dim[inx](x)
                
                # Collect intermediate features BEFORE upsampling for deep supervision
                if self.use_deep_supervision and inx > 0:
                    aux_features.append(x)
                
                x = layer_up(x)
        
        x = self.norm_up(x)
        
        # Final upsampling to original resolution
        H, W = self.img_size // 4, self.img_size // 4
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"
        
        x = self.up(x)
        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        
        # Project from embed_dim to 64 channels (matching Hybrid2's decoder4)
        x = self.final_proj(x)
        
        # Segmentation head (matching Hybrid2 baseline)
        x = self.output(x)
        
        # Process auxiliary outputs for deep supervision
        if self.use_deep_supervision:
            aux_outputs = self.process_aux_outputs(aux_features)
            return x, aux_outputs
        
        return x
    
    def process_aux_outputs(self, aux_features):
        """
        Process auxiliary features for deep supervision.
        MSAGHNet-style: Simple OutConv heads, outputs at native resolutions (NO upsampling).
        
        Args:
            aux_features: List of 3 intermediate features [stage1, stage2, stage3]
            Stage 1: H/16 resolution (after layer 0) - tokens at H/16 x W/16
            Stage 2: H/8 resolution (after layer 1) - tokens at H/8 x W/8
            Stage 3: H/4 resolution (after layer 2) - tokens at H/4 x W/4
        
        Returns:
            List of 3 auxiliary outputs at native resolutions:
            - aux1: [B, num_classes, H/16, W/16]
            - aux2: [B, num_classes, H/8, W/8]
            - aux3: [B, num_classes, H/4, W/4]
        """
        aux_outputs = []
        
        for i, aux_feat in enumerate(aux_features):
            # aux_feat: [B, L, C] - token format
            B, L, C = aux_feat.shape
            
            # Reshape tokens to spatial format
            h = w = int(L ** 0.5)
            aux_feat_spatial = aux_feat.view(B, h, w, C)
            aux_feat_spatial = aux_feat_spatial.permute(0, 3, 1, 2)  # [B, C, h, w]
            
            # Apply simple OutConv head (MSAGHNet style: single Conv2d)
            # Output remains at native resolution (NO upsampling)
            aux_out = self.aux_heads[i](aux_feat_spatial)  # [B, num_classes, h, w]
            aux_outputs.append(aux_out)
        
        return aux_outputs


# Example usage and training utilities
def create_model(config):
    """
    Create model based on config
    
    Example config structure:
    config = {
        'img_size': 224,
        'num_classes': 6,
        'efficientnet_model': 'tf_efficientnet_b4_ns',  # EfficientNet-B4 (same as Hybrid1)
        'pretrained': True,
        'embed_dim': 96,
        'depths_decoder': [2, 2, 2, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
        'batch_size': 8,
        'phase_epochs': 20,  # Epochs per phase
        'phase0_lr': 1e-3,  # Learning rate for phase 0
        'phase1_lr': 5e-4,  # Learning rate for phase 1
        'phase2_lr': 1e-4,  # Learning rate for phase 2
        'weight_decay': 0.01,
        'checkpoint_dir': './checkpoints',
        'save_freq': 5  # Save checkpoint every N epochs
    }
    """
    model = EfficientNetSwinUNet(
        img_size=config.get('img_size', 224),
        num_classes=config.get('num_classes', 6),
        efficientnet_model=config.get('efficientnet_model', 'tf_efficientnet_b4_ns'),
        encoder_type=config.get('encoder_type', 'efficientnet'),
        pretrained=config.get('pretrained', True),
        embed_dim=config.get('embed_dim', 96),
        depths_decoder=config.get('depths_decoder', [2, 2, 2, 2]),
        num_heads=config.get('num_heads', [3, 6, 12, 24]),
        window_size=config.get('window_size', 7),
        drop_rate=config.get('drop_rate', 0.0),
        drop_path_rate=config.get('drop_path_rate', 0.1),
        use_deep_supervision=config.get('use_deep_supervision', False),
        fusion_method=config.get('fusion_method', 'simple'),
        use_bottleneck=config.get('use_bottleneck', False),
        adapter_mode=config.get('adapter_mode', 'external'),
        use_multiscale_agg=config.get('use_multiscale_agg', False),
        use_groupnorm=config.get('use_groupnorm', False),
        use_se_msfe=config.get('use_se_msfe', False),
        use_msfa_mct_bottleneck=config.get('use_msfa_mct_bottleneck', False),
    )
    return model


# Progressive training strategy
class ProgressiveTrainer:
    """
    Training strategy following supervisor's suggestions:
    1. Freeze encoder, train decoder first
    2. Unfreeze layer by layer
    3. Fine-tune entire model
    """
    def __init__(self, model, optimizer_config):
        self.model = model
        self.optimizer_config = optimizer_config
        self.phase = 0
    
    def get_optimizer(self):
        """Get optimizer for current training phase"""
        if self.phase == 0:  # Decoder only
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            params = self.model.parameters()
        
        return torch.optim.AdamW(
            params,
            lr=self.optimizer_config['lr'],
            weight_decay=self.optimizer_config['weight_decay']
        )
    
    def start_phase_0(self):
        """Phase 0: Train decoder with frozen encoder"""
        print("=== Phase 0: Training decoder with frozen encoder ===")
        self.model.freeze_encoder()
        self.phase = 0
    
    def start_phase_1(self, layer_idx=0):
        """Phase 1: Progressive unfreezing"""
        print(f"=== Phase 1: Progressive unfreezing - Layer {layer_idx} ===")
        self.model.unfreeze_layer_by_layer(layer_idx)
        self.phase = 1
    
    def start_phase_2(self):
        """Phase 2: Fine-tune entire model"""
        print("=== Phase 2: Fine-tuning entire model ===")
        self.model.unfreeze_encoder()
        self.phase = 2


if __name__ == '__main__':
    # Example usage code - commented out as it requires external dependencies
    # Uncomment and modify if you want to test the model directly
    """
    # Configuration
    config = {
        'img_size': 224,
        'num_classes': 6,
        'efficientnet_model': 'tf_efficientnet_b4_ns',
        'pretrained': True,
        'embed_dim': 96,
        'depths_decoder': [2, 2, 2, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
        'batch_size': 8,
        'phase_epochs': 20,
        'phase0_lr': 1e-3,
        'phase1_lr': 5e-4,
        'phase2_lr': 1e-4,
        'weight_decay': 0.01,
        'checkpoint_dir': './checkpoints',
        'save_freq': 5
    }
    
    # Create model
    model = create_model(config)
    print("Model created successfully!")
    """
    pass
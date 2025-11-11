import torch
import torch.nn as nn
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
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
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
        # Using rfft2 on the last two dimensions (H, W)
        feat1_fft = torch.fft.rfft2(feat1_2d_fp32, dim=(-2, -1), norm='ortho')  # (B, C, H1, W1//2+1)
        feat2_fft = torch.fft.rfft2(feat2_2d_fp32, dim=(-2, -1), norm='ortho')  # (B, C, H1, W1//2+1)
        
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
        fused_spatial = torch.fft.irfft2(fused_complex, s=(H1, W1), dim=(-2, -1), norm='ortho')
        
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


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet encoder with feature extraction at multiple scales using timm.
    
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
    def __init__(self, model_name='tf_efficientnet_b4_ns', pretrained=True, freeze_bn=False):
        super().__init__()
        
        # Load pretrained EfficientNet using timm (same as Hybrid1)
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
        return self.fuse(fused)


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


class EfficientNetSwinUNet(nn.Module):
    """
    Hybrid CNN-Transformer UNet with EfficientNet encoder and Swin Transformer decoder
    """
    def __init__(self, img_size=224, num_classes=6, efficientnet_model='tf_efficientnet_b4_ns',
                 pretrained=True, embed_dim=96, depths_decoder=[2, 2, 2, 2], 
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, use_checkpoint=False,
                 use_deep_supervision=False, fusion_method='simple', use_bottleneck=False, 
                 adapter_mode='external', use_multiscale_agg=False, use_groupnorm=False):
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
        
        # EfficientNet Encoder
        self.encoder = EfficientNetEncoder(model_name=efficientnet_model, pretrained=pretrained)
        
        # Target dimensions for decoder (matching your diagram exactly)
        # Stage resolutions: 56x56, 28x28, 14x14, 7x7
        # Channel dimensions: 96, 192, 384, 768 (as shown in your diagram)
        self.decoder_dims = [96, 192, 384, 768]  # Exact dimensions from your diagram
        
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
            # Calculate stochastic drop_path matching SwinUnet
            # SwinUnet uses depths=[2, 2, 2, 2] for encoder, bottleneck is last 2 blocks
            # dpr = linspace(0, drop_path_rate, sum(depths)) = linspace(0, 0.1, 8)
            # Bottleneck uses last 2 values: dpr[6:8] ≈ [0.086, 0.1]
            depths_encoder = [2, 2, 2, 2]  # Matching SwinUnet encoder depths
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_encoder))]
            # Bottleneck is the last encoder layer (layer 3), uses last 2 blocks
            bottleneck_drop_path = dpr[sum(depths_encoder[:3]):sum(depths_encoder[:4])]
            
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
                    in_dim1=384,  # Layer 1 input dimension
                    in_dim2=384,  # C3 encoder dimension
                    out_dim=384   # Layer 1 output dimension
                ),
                FourierFeatureFusion(
                    in_dim1=192,  # Layer 2 input dimension
                    in_dim2=192,  # C2 encoder dimension
                    out_dim=192   # Layer 2 output dimension
                ),
                FourierFeatureFusion(
                    in_dim1=96,   # Layer 3 input dimension
                    in_dim2=96,   # C1 encoder dimension
                    out_dim=96    # Layer 3 output dimension
                )
            ])
        elif fusion_method == 'smart':
            # Attention-based smart skip connections (like hybrid1)
            self.smart_skips = nn.ModuleList([
                SmartSkipConnectionTransformer(encoder_dim=384, decoder_dim=384, num_heads=num_heads[2]),
                SmartSkipConnectionTransformer(encoder_dim=192, decoder_dim=192, num_heads=num_heads[1]),
                SmartSkipConnectionTransformer(encoder_dim=96,  decoder_dim=96,  num_heads=num_heads[0]),
            ])
            self.skip_fusions = None
        elif fusion_method == 'simple':
            # Simple skip connections matching hybrid2's SimpleSkipConnection pattern
            self.simple_skips = nn.ModuleList([
                SimpleSkipConnectionTransformer(encoder_dim=384, decoder_dim=384),
                SimpleSkipConnectionTransformer(encoder_dim=192, decoder_dim=192),
                SimpleSkipConnectionTransformer(encoder_dim=96,  decoder_dim=96),
            ])
            self.skip_fusions = None
            self.smart_skips = None
        else:
            # Default: no special fusion (fallback to original simple concat)
            self.skip_fusions = None
            self.smart_skips = None
            self.simple_skips = None
        
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
        if use_deep_supervision:
            # Stages 1, 2, 3 have dimensions: [384, 192, 96]
            aux_dims = [int(embed_dim * 2 ** (3-i)) for i in range(1, 4)]  # [384, 192, 96]
            self.aux_heads = nn.ModuleList([
                nn.Sequential(
                    norm_layer(dim),
                    nn.Linear(dim, num_classes)
                ) for dim in aux_dims
            ])
            print("🚀 Deep Supervision enabled: 3 auxiliary outputs")
            print(f"   Aux dims: {aux_dims}")
        
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
                    proj_feat = proj_feat.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
                    proj_feat = proj_feat.view(B, -1, C)  # [B, L, bottleneck_dim]
                    
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
        Matches hybrid2's pattern: uses scale_factor for upsampling.
        
        Args:
            aux_features: List of 3 intermediate features [stage1, stage2, stage3]
            Stage 1: H/16 resolution (after layer 0)
            Stage 2: H/8 resolution (after layer 1)
            Stage 3: H/4 resolution (after layer 2)
        
        Returns:
            List of 3 auxiliary outputs [B, num_classes, H, W]
        """
        import torch.nn.functional as F
        
        aux_outputs = []
        
        # Scale factors matching hybrid2: [16, 8, 4] for resolutions [H/16, H/8, H/4] -> H
        scale_factors = [16, 8, 4]
        
        for i, aux_feat in enumerate(aux_features):
            # aux_feat: [B, L, C]
            B, L, C = aux_feat.shape
            
            # Apply auxiliary head
            aux_out = self.aux_heads[i](aux_feat)  # [B, L, num_classes]
            
            # Reshape to spatial
            h = w = int(L ** 0.5)
            aux_out = aux_out.view(B, h, w, self.num_classes)
            aux_out = aux_out.permute(0, 3, 1, 2)  # [B, num_classes, h, w]
            
            # Upsample using scale_factor (matching hybrid2 pattern)
            aux_out = F.interpolate(aux_out, scale_factor=scale_factors[i], 
                                   mode='bilinear', align_corners=False)
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

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ============================================================================
# SWIN TRANSFORMER COMPONENTS
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
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
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
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

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
            # if window size is larger than input resolution, we don't partition windows
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
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
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

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
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

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
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
        """
        x: B, H*W, C
        """
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
        """
        x: B, H*W, C
        """
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


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
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

        # patch merging layer
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


# ============================================================================
# SMART SKIP CONNECTION FOR TRANSFORMER DECODER
# ============================================================================

class SmartSkipConnectionTransformer(nn.Module):
    """
    Smart skip connection for transformer decoder (token space).
    Equivalent to Hybrid2's ImprovedSmartSkipConnection but operates on tokens.
    
    Features:
    - Linear alignment of encoder tokens to decoder dimension
    - Multi-head self-attention for skip feature enhancement
    - MLP fusion of decoder and enhanced skip features
    """
    
    def __init__(self, encoder_dim, decoder_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        # 1. Align encoder tokens to decoder dimension
        self.align = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.GELU()
        )
        
        # 2. Multi-head self-attention for skip feature enhancement
        # This is similar to CBAM but for token space
        self.attention = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_attn = nn.LayerNorm(decoder_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 3. Fusion MLP (similar to Hybrid2's fusion convolution)
        self.fuse = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim * 4),
            nn.LayerNorm(decoder_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim * 4, decoder_dim),
            nn.LayerNorm(decoder_dim)
        )
    
    def forward(self, encoder_tokens, decoder_tokens):
        """
        Smart fusion of encoder skip tokens with decoder tokens.
        
        Args:
            encoder_tokens: Skip features from encoder [B, L, C_enc]
            decoder_tokens: Features from decoder path [B, L, C_dec]
        
        Returns:
            Fused tokens [B, L, C_dec]
        """
        B_enc, L_enc, C_enc = encoder_tokens.shape
        B_dec, L_dec, C_dec = decoder_tokens.shape
        
        # Step 1: Align encoder tokens to decoder dimension
        skip_tokens = self.align(encoder_tokens)  # [B, L, decoder_dim]
        
        # Step 2: Self-attention on skip tokens to enhance important features
        # This is analogous to CBAM's channel and spatial attention
        skip_enhanced, _ = self.attention(skip_tokens, skip_tokens, skip_tokens)
        skip_tokens = self.norm_attn(skip_tokens + self.dropout(skip_enhanced))
        
        # Step 3: Ensure spatial alignment
        if L_enc != L_dec:
            # Interpolate skip tokens to match decoder resolution
            # Reshape to 2D, interpolate, then back to tokens
            h_enc = w_enc = int(L_enc ** 0.5)
            h_dec = w_dec = int(L_dec ** 0.5)
            
            skip_tokens_2d = skip_tokens.view(B_enc, h_enc, w_enc, -1).permute(0, 3, 1, 2)
            skip_tokens_2d = torch.nn.functional.interpolate(
                skip_tokens_2d, size=(h_dec, w_dec), 
                mode='bilinear', align_corners=False
            )
            skip_tokens = skip_tokens_2d.permute(0, 2, 3, 1).reshape(B_dec, L_dec, -1)
        
        # Step 4: Concatenate and fuse
        fused = torch.cat([decoder_tokens, skip_tokens], dim=-1)  # [B, L, 2*decoder_dim]
        output = self.fuse(fused)  # [B, L, decoder_dim]
        
        return output


# ============================================================================
# SWIN DECODER
# ============================================================================

class SwinDecoder(nn.Module):
    """
    Enhanced Swin-Unet decoder with TransUNet best practices.
    
    Enhancements:
    - Deep Supervision (auxiliary outputs)
    - Multi-Scale Aggregation support (bottleneck)
    - Better skip connections
    """
    
    def __init__(self, num_classes: int = 5, img_size: int = 224, embed_dim: int = 96, 
                 use_deep_supervision: bool = False, use_multiscale_agg: bool = False,
                 use_smart_skip: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patches_resolution = (img_size // 4, img_size // 4)  # Base resolution for 224 input
        self.use_deep_supervision = use_deep_supervision
        self.use_multiscale_agg = use_multiscale_agg
        self.use_smart_skip = use_smart_skip
        
        # Add bottleneck layer with 2 SwinBlocks (missing from original implementation!)
        self.bottleneck_depth = 2  # 2 SwinBlocks for bottleneck processing
        self.bottleneck_dim = int(embed_dim * 2 ** 3)  # 768 for embed_dim=96
        self.bottleneck_resolution = (self.patches_resolution[0] // 8, self.patches_resolution[1] // 8)  # 7x7
        
        # Bottleneck layer (equivalent to deepest encoder layer with 2 SwinBlocks)
        self.bottleneck = BasicLayer_up(
            dim=self.bottleneck_dim,
            input_resolution=self.bottleneck_resolution,
            depth=self.bottleneck_depth,
            num_heads=24,  # Same as deepest layer
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True, qk_scale=None,
            drop=0.0, attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            upsample=None,  # No upsampling in bottleneck
            use_checkpoint=False)
        
        # Build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        
        # Decoder configuration (matching Swin-Tiny)
        depths_decoder = [1, 2, 2, 2]
        num_heads = [3, 6, 12, 24]
        
        for i_layer in range(4):  # 4 decoder stages
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (3 - i_layer)),
                                      int(embed_dim * 2 ** (3 - i_layer))) if i_layer > 0 else nn.Identity()
            
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(self.patches_resolution[0] // (2 ** (3 - i_layer)),
                                      self.patches_resolution[1] // (2 ** (3 - i_layer))),
                    dim=int(embed_dim * 2 ** (3 - i_layer)), dim_scale=2, norm_layer=nn.LayerNorm)
            else:
                layer_up = BasicLayer_up(
                    dim=int(embed_dim * 2 ** (3 - i_layer)),
                    input_resolution=(self.patches_resolution[0] // (2 ** (3 - i_layer)),
                                      self.patches_resolution[1] // (2 ** (3 - i_layer))),
                    depth=depths_decoder[3 - i_layer],
                    num_heads=num_heads[3 - i_layer],
                    window_size=7,
                    mlp_ratio=4.0,
                    qkv_bias=True, qk_scale=None,
                    drop=0.0, attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=nn.LayerNorm,
                    upsample=PatchExpand if (i_layer < 3) else None,
                    use_checkpoint=False)
            
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        
        self.norm_up = nn.LayerNorm(self.embed_dim)
        
        # Final upsampling and output
        self.up = FinalPatchExpand_X4(
            input_resolution=(img_size // 4, img_size // 4),
            dim_scale=4, dim=embed_dim)
        
        # REFERENCE ARCHITECTURE: Conv3×3 → ReLU → Conv1×1
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, 
                     kernel_size=3, padding=1, bias=False),  # 3x3 conv for feature refinement
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, 
                     kernel_size=1, bias=False)  # 1x1 conv for classification
        )
        
        # Deep Supervision: Auxiliary heads for intermediate outputs
        if use_deep_supervision:
            # Stages 1, 2, 3 have dimensions: [384, 192, 96]
            aux_dims = [int(embed_dim * 2 ** (3-i)) for i in range(1, 4)]  # [384, 192, 96]
            self.aux_heads = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, num_classes)
                ) for dim in aux_dims
            ])
            print("🚀 Deep Supervision enabled: 3 auxiliary outputs")
            print(f"   Aux dims: {aux_dims}")
        
        # Multi-Scale Aggregation support
        if use_multiscale_agg:
            # Projection layers to combine multi-scale CNN features
            self.multiscale_proj = nn.ModuleList([
                nn.Linear(int(embed_dim * 2 ** i), self.bottleneck_dim)
                for i in range(4)  # Project all 4 scales to bottleneck dim
            ])
            self.multiscale_fusion = nn.Linear(self.bottleneck_dim * 4, self.bottleneck_dim)
            print("🚀 Multi-Scale Aggregation enabled in bottleneck")
        
        # Smart Skip Connections (OPTIONAL enhancement, disabled by default for baseline)
        if use_smart_skip:
            # Create smart skip connections for stages 1, 2, 3
            skip_dims = [
                (int(embed_dim * 2 ** (3-i)), int(embed_dim * 2 ** (3-i)))
                for i in range(1, 4)  # [(384,384), (192,192), (96,96)]
            ]
            self.smart_skips = nn.ModuleList([
                SmartSkipConnectionTransformer(
                    encoder_dim=enc_dim,
                    decoder_dim=dec_dim,
                    num_heads=num_heads[3 - (i+1)],  # Match attention heads
                    dropout=0.1
                ) for i, (enc_dim, dec_dim) in enumerate(skip_dims)
            ])
            print("🚀 Smart Skip Connections enabled (attention-based fusion)")
            print(f"   Skip dims: {[(enc, dec) for enc, dec in skip_dims]}")
        else:
            # BASELINE: Use naive concatenation (REFERENCE ARCHITECTURE)
            self.smart_skips = None
            print("✅ Using BASELINE skip connections (naive concatenation)")
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_bottleneck(self, x, all_features=None):
        """
        Forward through bottleneck layer with 2 SwinBlocks.
        Optionally applies multi-scale aggregation if enabled.
        
        Args:
            x: Deepest feature tokens (B, L, C) where C=768, L=7*7=49
            all_features: Optional list of all 4 encoder features for multi-scale aggregation
            
        Returns:
            Processed bottleneck features (B, L, C)
        """
        if self.use_multiscale_agg and all_features is not None:
            # Apply multi-scale aggregation
            import torch.nn.functional as F
            
            B, L, C = x.shape
            h = w = int(L ** 0.5)
            
            # Project and resize all features to bottleneck size
            projected = []
            for i, feat in enumerate(all_features):
                # feat: [B, L_i, C_i]
                B_f, L_f, C_f = feat.shape
                h_f = w_f = int(L_f ** 0.5)
                
                # Project to bottleneck dim
                proj_feat = self.multiscale_proj[i](feat)  # [B, L_i, bottleneck_dim]
                
                # Reshape and resize to bottleneck spatial size
                proj_feat = proj_feat.view(B_f, h_f, w_f, self.bottleneck_dim)
                proj_feat = proj_feat.permute(0, 3, 1, 2)  # [B, C, H, W]
                proj_feat = F.interpolate(proj_feat, size=(h, w), mode='bilinear', align_corners=False)
                proj_feat = proj_feat.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
                proj_feat = proj_feat.view(B, -1, self.bottleneck_dim)  # [B, L, C]
                
                projected.append(proj_feat)
            
            # Concatenate and fuse
            aggregated = torch.cat(projected, dim=-1)  # [B, L, bottleneck_dim*4]
            fused = self.multiscale_fusion(aggregated)  # [B, L, bottleneck_dim]
            
            # Add to original features (residual)
            x = x + fused
        
        return self.bottleneck(x)
    
    def forward_up_features(self, x, x_downsample):
        """
        Forward through decoder with SMART skip connections.
        
        Returns:
            If use_deep_supervision:
                (final_features, [aux_feat1, aux_feat2, aux_feat3])
            Else:
                final_features
        """
        aux_features = [] if self.use_deep_supervision else None
        
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                # SMART SKIP CONNECTION (attention-based fusion) or fallback to naive
                if self.smart_skips is not None:
                    # Use attention-enhanced skip connection
                    encoder_skip = x_downsample[3 - inx]
                    x = self.smart_skips[inx - 1](encoder_skip, x)
                else:
                    # Fallback to naive concatenation (baseline)
                    x = torch.cat([x, x_downsample[3 - inx]], -1)
                    x = self.concat_back_dim[inx](x)
                
                # Collect intermediate features BEFORE upsampling for deep supervision
                if self.use_deep_supervision:
                    aux_features.append(x)
                
                x = layer_up(x)
        
        x = self.norm_up(x)  # B L C
        
        if self.use_deep_supervision:
            return x, aux_features
        return x
    
    def up_x4(self, x):
        """Final upsampling to original resolution."""
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"
        
        x = self.up(x)
        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        x = self.output(x)
        return x
    
    def process_aux_outputs(self, aux_features):
        """
        Process auxiliary features for deep supervision.
        
        Args:
            aux_features: List of 3 intermediate features [stage1, stage2, stage3]
        
        Returns:
            List of 3 auxiliary outputs [B, num_classes, H, W]
        """
        import torch.nn.functional as F
        
        aux_outputs = []
        H, W = self.patches_resolution
        
        for i, aux_feat in enumerate(aux_features):
            # aux_feat: [B, L, C]
            B, L, C = aux_feat.shape
            
            # Apply auxiliary head
            aux_out = self.aux_heads[i](aux_feat)  # [B, L, num_classes]
            
            # Reshape to spatial
            h = w = int(L ** 0.5)
            aux_out = aux_out.view(B, h, w, self.num_classes)
            aux_out = aux_out.permute(0, 3, 1, 2)  # [B, num_classes, h, w]
            
            # Upsample to full resolution
            aux_out = F.interpolate(aux_out, size=(self.img_size, self.img_size), 
                                   mode='bilinear', align_corners=False)
            aux_outputs.append(aux_out)
        
        return aux_outputs

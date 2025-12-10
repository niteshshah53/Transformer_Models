#!/usr/bin/env python3
"""
CNN-Transformer Vision Transformer for Historical Document Segmentation
Based on EfficientNetSwinUNet from cnn_transformer.py
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from cnn_transformer import EfficientNetSwinUNet, create_model


class CNNTransformerUnet(nn.Module):
    """
    CNN-Transformer U-Net wrapper for historical document segmentation.
    
    This class wraps the EfficientNetSwinUNet model from cnn_transformer.py
    to provide a consistent interface with the existing training/testing pipeline.
    """
    
    def __init__(self, config, img_size=224, num_classes=6, zero_head=False, vis=False, 
                 use_deep_supervision=False, fusion_method='simple', use_bottleneck=False, 
                 adapter_mode='external', use_multiscale_agg=False, use_groupnorm=False,
                 encoder_type='efficientnet', use_se_msfe=False, use_msfa_mct_bottleneck=False,
                 decoder_depths=None):
        """
        Initialize CNN-Transformer U-Net model.
        
        Args:
            config: Configuration object (used for decoder_depths if provided)
            img_size: Input image size (default: 224)
            num_classes: Number of segmentation classes (default: 6)
            zero_head: Whether to zero initialize the head (not used, kept for compatibility)
            vis: Whether to enable visualization (not used, kept for compatibility)
            use_deep_supervision: Whether to enable deep supervision (default: False)
            fusion_method: Feature fusion method - 'simple', 'fourier', or 'smart' (default: 'simple')
            use_bottleneck: Whether to use bottleneck (default: False)
            adapter_mode: Adapter mode - 'external' or 'streaming' (default: 'external')
            use_multiscale_agg: Whether to use multi-scale aggregation (default: False)
            use_groupnorm: Whether to use GroupNorm in adapters (default: False)
            encoder_type: Encoder type - 'efficientnet' or 'resnet50' (default: 'efficientnet')
            decoder_depths: List of decoder depths for each stage (default: [2, 2, 2, 2])
        """
        super(CNNTransformerUnet, self).__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_deep_supervision = use_deep_supervision
        self.fusion_method = fusion_method
        self.use_bottleneck = use_bottleneck
        self.adapter_mode = adapter_mode
        self.use_multiscale_agg = use_multiscale_agg
        self.use_groupnorm = use_groupnorm
        self.encoder_type = encoder_type
        
        # Get decoder depths from parameter, config, or default
        if decoder_depths is not None:
            depths_decoder = decoder_depths
        elif config is not None and hasattr(config.MODEL, 'SWIN') and hasattr(config.MODEL.SWIN, 'DECODER_DEPTHS'):
            depths_decoder = config.MODEL.SWIN.DECODER_DEPTHS
        else:
            depths_decoder = [2, 2, 2, 2]  # Default
        
        # Get encoder parameters from config if available
        if config is not None and hasattr(config.MODEL, 'ENCODER'):
            efficientnet_model = getattr(config.MODEL.ENCODER, 'EFFICIENTNET_MODEL', 'tf_efficientnet_b4_ns')
            pretrained = getattr(config.MODEL.ENCODER, 'PRETRAINED', True)
        else:
            efficientnet_model = 'tf_efficientnet_b4_ns'  # EfficientNet-B4 (same as Hybrid1)
            pretrained = True
        
        # Check if SimMIM pretrained checkpoint is specified - if so, use Swin Base dimensions EXACTLY
        use_simmim_dims = False
        if config is not None and hasattr(config.MODEL, 'PRETRAIN_CKPT'):
            pretrained_path = config.MODEL.PRETRAIN_CKPT
            if pretrained_path and 'simmim' in pretrained_path.lower():
                # Accept any SimMIM checkpoint (including fine-tuned DivaHisDB) and treat as Swin-Base
                use_simmim_dims = True
                print("   Detected SimMIM checkpoint - forcing Swin-Base decoder geometry")
                if 'swin_base' in pretrained_path.lower():
                    print("   Decoder will match encoder: dims=[128,256,512,1024], heads=[4,8,16,32], window=6")
                else:
                    print(f"   Checkpoint: {pretrained_path} (assumed Swin-Base SimMIM); overriding to decoder depths [2,18,2,2]")
        
        # Get Swin parameters from config if available, or use SimMIM dimensions if detected
        if use_simmim_dims:
            # SimMIM Swin Base dimensions (EXACT match)
            # SimMIM encoder stages: [2, 2, 18, 2] blocks with dims [128, 256, 512, 1024] and heads [4, 8, 16, 32]
            # Decoder stages (reversed order): [2, 18, 2, 2] blocks with dims [1024, 512, 256, 128] and heads [32, 16, 8, 4]
            embed_dim = 128
            num_heads = [4, 8, 16, 32]  # Will be reversed in decoder layers: [32, 16, 8, 4]
            # Use window_size=7 (interpolate from SimMIM's window_size=6 → model's window_size=7, matching SwinUnet/Hybrid)
            window_size = getattr(config.MODEL.SWIN, 'WINDOW_SIZE', 7) if config and hasattr(config.MODEL, 'SWIN') else 7
            # Override decoder depths to match SimMIM encoder depths (reversed)
            # SimMIM encoder: [2, 2, 18, 2] → Decoder (reversed): [2, 18, 2, 2]
            original_depths = depths_decoder.copy() if isinstance(depths_decoder, list) else depths_decoder
            depths_decoder = [2, 18, 2, 2]  # Match SimMIM encoder depths (reversed)
            print(f"   Overriding decoder depths: {original_depths} → {depths_decoder} (matches SimMIM encoder reversed)")
            print(f"   SimMIM Base config: embed_dim={embed_dim}, num_heads={num_heads}, window_size={window_size} (will interpolate from SimMIM's window_size=6)")
        elif config is not None and hasattr(config.MODEL, 'SWIN'):
            embed_dim = getattr(config.MODEL.SWIN, 'EMBED_DIM', 96)
            num_heads = getattr(config.MODEL.SWIN, 'NUM_HEADS', [3, 6, 12, 24])
            window_size = getattr(config.MODEL.SWIN, 'WINDOW_SIZE', 7)
        else:
            embed_dim = 96
            num_heads = [3, 6, 12, 24]
            window_size = 7
        
        # Get drop path rate from config if available
        if config is not None and hasattr(config.MODEL, 'DROP_PATH_RATE'):
            drop_path_rate = config.MODEL.DROP_PATH_RATE
        else:
            drop_path_rate = 0.1
        
        # Create model configuration
        model_config = {
            'img_size': img_size,
            'num_classes': num_classes,
            'encoder_type': encoder_type,  # 'efficientnet' or 'resnet50'
            'efficientnet_model': efficientnet_model,
            'pretrained': pretrained,
            'embed_dim': embed_dim,
            'depths_decoder': depths_decoder,  # Now from config or parameter
            'num_heads': num_heads,
            'window_size': window_size,
            'drop_rate': 0.0,
            'drop_path_rate': drop_path_rate,
            'use_deep_supervision': use_deep_supervision,
            'fusion_method': fusion_method,
            'use_bottleneck': use_bottleneck,
            'adapter_mode': adapter_mode,
            'use_multiscale_agg': use_multiscale_agg,
            'use_groupnorm': use_groupnorm,  # Pass GroupNorm flag
            'use_se_msfe': use_se_msfe,  # Pass SE-MSFE flag
            'use_msfa_mct_bottleneck': use_msfa_mct_bottleneck,  # Pass MSFA+MCT bottleneck flag
        }
        
        # Create the CNN-Transformer model
        self.model = create_model(model_config)
        
        print(f"CNN-Transformer U-Net initialized:")
        print(f"  - Image size: {img_size}")
        print(f"  - Number of classes: {num_classes}")
        if encoder_type == 'resnet50':
            print(f"  - Encoder: ResNet-50 (official)")
        else:
            print(f"  - Encoder: EfficientNet-B4")
            print(f"  - EfficientNet variant: {model_config['efficientnet_model']}")
        print(f"  - Embed dimension: {model_config['embed_dim']}")
        print(f"  - Decoder depths: {model_config['depths_decoder']} (from config/parameter)")
        if use_deep_supervision:
            print(f"  - ✅ Deep Supervision: ENABLED (3 auxiliary outputs)")
        else:
            print(f"  - Deep Supervision: DISABLED")
        print(f"  - Fusion Method: {fusion_method.upper()}")
        print(f"  - Bottleneck (2 Swin blocks): {'ENABLED' if use_bottleneck else 'DISABLED'}")
        if use_multiscale_agg:
            print(f"  - ✅ Multi-Scale Aggregation: ENABLED (bottleneck)")
        print(f"  - Adapter Mode: {adapter_mode.upper()}")
        if use_groupnorm:
            print(f"  - ✅ GroupNorm: ENABLED (in adapters)")
        else:
            print(f"  - GroupNorm: DISABLED (using LayerNorm)")
        
    def forward(self, x):
        """
        Forward pass through the CNN-Transformer model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, num_classes, H, W)
        """
        return self.model(x)
    
    def _resize_relative_position_bias_table(self, table, old_window, new_window):
        """
        Interpolate relative position bias table from old_window to new_window.
        
        Args:
            table: Relative position bias table tensor with shape (old_size * old_size, num_heads)
            old_window: Original window size (e.g., 6 for SimMIM)
            new_window: Target window size (e.g., 7 for decoder)
            
        Returns:
            Interpolated table with shape (new_size * new_size, num_heads)
        """
        import torch.nn.functional as F
        import math
        
        # Derive the real window size from the checkpoint if possible.
        # Some checkpoints (e.g., fine-tuned on DivaHisDB) already store bias tables for window_size=7
        # even when the config name suggests SimMIM (window_size=6). Using the inferred size avoids
        # invalid reshapes like 2704 -> [11,11,16].
        derived_window = old_window
        if table.numel() > 0:
            # table shape: (num_entries, num_heads)
            num_entries = table.shape[0]
            inferred = (math.isqrt(num_entries) + 1) // 2
            if (2 * inferred - 1) ** 2 == num_entries:
                derived_window = inferred

        old_size = 2 * derived_window - 1
        new_size = 2 * new_window - 1
        
        # Reshape to (1, num_heads, old_size, old_size)
        num_heads = table.shape[-1]
        table = table.reshape(old_size, old_size, num_heads).permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, old_size, old_size)
        
        # Interpolate using bicubic interpolation
        table = F.interpolate(table, size=(new_size, new_size), mode='bicubic', align_corners=False)
        
        # Reshape back to (new_size * new_size, num_heads)
        table = table.squeeze(0).permute(1, 2, 0).reshape(new_size * new_size, num_heads)
        
        return table
    
    def _is_simmim_config(self, config):
        """
        Check if the SimMIM config is being used.
        
        Args:
            config: Configuration object
            
        Returns:
            bool: True if SimMIM config is detected
        """
        # Check config name
        if hasattr(config.MODEL, 'NAME') and 'simmim' in config.MODEL.NAME.lower():
            return True
        
        # Check pretrained checkpoint path
        if hasattr(config.MODEL, 'PRETRAIN_CKPT') and config.MODEL.PRETRAIN_CKPT:
            pretrained_path = config.MODEL.PRETRAIN_CKPT
            if 'simmim' in pretrained_path.lower():
                return True
        
        return False
    
    def load_from(self, config):
        """
        Load pretrained decoder weights from Swin Transformer checkpoint (e.g., SimMIM).
        
        This method properly maps SimMIM encoder layers to decoder layers and handles
        window size differences by interpolating relative position bias tables.
        
        The network model uses EfficientNet/ResNet encoder (pretrained automatically),
        but can load Swin Transformer decoder weights from a checkpoint if provided.
        
        Args:
            config: Configuration object with MODEL.PRETRAIN_CKPT path
            
        Note:
            - Encoder weights come from EfficientNet/ResNet ImageNet pretraining (automatic)
            - Decoder weights are loaded from SimMIM checkpoint by mapping encoder layers to decoder
            - Only Swin block weights are loaded (attn, mlp, norms) - NOT patch expanding layers
            - Relative position bias tables are interpolated for different window sizes
        """
        if config is None or not hasattr(config.MODEL, 'PRETRAIN_CKPT'):
            print("No pretrained checkpoint path in config - skipping decoder weight loading")
            print("CNN-Transformer model uses EfficientNet/ResNet pretrained encoder weights automatically.")
            return True
        
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if not pretrained_path or pretrained_path == "":
            print("No pretrained checkpoint path specified - skipping decoder weight loading")
            print("CNN-Transformer model uses EfficientNet/ResNet pretrained encoder weights automatically.")
            return True
        
        import os
        import copy
        
        # Resolve relative paths
        if not os.path.isabs(pretrained_path):
            # Try relative to config file location, then relative to current working directory
            config_dir = os.path.dirname(os.path.abspath(pretrained_path)) if os.path.dirname(pretrained_path) else os.getcwd()
            if not os.path.exists(pretrained_path):
                # Try relative to common/configs directory
                base_dir = os.path.join(os.path.dirname(__file__), '../../common/configs')
                alt_path = os.path.join(os.path.dirname(base_dir), pretrained_path.lstrip('./'))
                if os.path.exists(alt_path):
                    pretrained_path = alt_path
        
        if not os.path.exists(pretrained_path):
            print(f"⚠️  Pretrained checkpoint not found: {pretrained_path}")
            print("   Continuing without loading decoder weights (decoder will be randomly initialized)")
            return False
        
        print(f"Loading decoder weights from SimMIM checkpoint: {pretrained_path}")
        print("   (Encoder uses EfficientNet/ResNet ImageNet pretrained weights automatically)")
        
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(pretrained_dict, dict) and 'model' in pretrained_dict:
                pretrained_dict = pretrained_dict['model']
            elif isinstance(pretrained_dict, dict) and 'model_state' in pretrained_dict:
                pretrained_dict = pretrained_dict['model_state']
            
            # Remove prefix if present (e.g., "swin_unet." or "model.")
            if any(k.startswith('swin_unet.') for k in pretrained_dict.keys()):
                pretrained_dict = {k[10:]: v for k, v in pretrained_dict.items() if k.startswith('swin_unet.')}
            elif any(k.startswith('model.') for k in pretrained_dict.keys()):
                pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k.startswith('model.')}
            
            # Get model's state dict
            model_dict = self.model.state_dict()
            
            # Detect SimMIM config and window sizes
            is_simmim = self._is_simmim_config(config)
            # Match SwinUnet and Hybrid: SimMIM uses window_size=6, model uses window_size=7 (interpolate from 6 to 7)
            old_window_size = 6 if is_simmim else 7  # SimMIM uses window_size=6 (matching SwinUnet/Hybrid)
            new_window_size = config.MODEL.SWIN.WINDOW_SIZE if hasattr(config.MODEL, 'SWIN') and hasattr(config.MODEL.SWIN, 'WINDOW_SIZE') else 7
            
            if is_simmim and old_window_size != new_window_size:
                print(f"   Detected SimMIM checkpoint (window_size={old_window_size})")
                print(f"   Model decoder uses window_size={new_window_size}")
                print(f"   Will interpolate relative position bias tables from window_size={old_window_size} to window_size={new_window_size}")
            
            # Build mapping: SimMIM encoder layers -> decoder layers (reverse order)
            # layers.3 -> layers_up.0, layers.2 -> layers_up.1, layers.1 -> layers_up.2, layers.0 -> layers_up.3
            full_dict = {}
            loaded_count = 0
            skipped_count = 0
            interpolated_count = 0
            
            for k, v in pretrained_dict.items():
                # Skip non-Swin keys (patch_embed, norm, output head, etc.)
                if any(skip_key in k for skip_key in ['patch_embed', 'output', 'head', 'mask_token']):
                    skipped_count += 1
                    continue
                
                # Skip norm (encoder norm, not decoder norm_up)
                if k == 'norm' or (k.startswith('norm') and 'norm_up' not in k and 'layers' not in k):
                    skipped_count += 1
                    continue
                
                # Map encoder layers to decoder layers (reverse order)
                if "layers." in k and "layers_up" not in k:
                    try:
                        # Extract layer number from encoder (e.g., "layers.3.blocks.0.attn.qkv" -> layer 3)
                        layers_pos = k.find("layers.")
                        if layers_pos != -1:
                            start_pos = layers_pos + len("layers.")
                            end_pos = k.find(".", start_pos)
                            if end_pos == -1:
                                end_pos = len(k)
                            layer_str = k[start_pos:end_pos]
                            layer_num = int(layer_str)
                            
                            # Map to decoder layer (reverse order: layer 3 -> layers_up.0, layer 2 -> layers_up.1, etc.)
                            decoder_layer_num = 3 - layer_num
                            if decoder_layer_num >= 0:
                                # Reconstruct key for decoder: "layers.3.blocks.0.attn.qkv" -> "layers_up.0.blocks.0.attn.qkv"
                                decoder_k = "layers_up." + str(decoder_layer_num) + k[end_pos:]
                                
                                # Only load Swin block weights (attn, mlp, norms) - NOT patch expanding
                                if any(block_key in decoder_k for block_key in ['blocks.', 'attn.', 'mlp.', 'norm']):
                                    # Skip patch expanding/upsampling layers
                                    if any(skip in decoder_k for skip in ['upsample', 'expand', 'patch_expand']):
                                        skipped_count += 1
                                        continue
                                    
                                    if decoder_k in model_dict:
                                        # Handle relative position bias table interpolation
                                        if "relative_position_bias_table" in decoder_k:
                                            if old_window_size != new_window_size:
                                                v = self._resize_relative_position_bias_table(v, old_window_size, new_window_size)
                                                interpolated_count += 1
                                        
                                        if v.shape == model_dict[decoder_k].shape:
                                            full_dict[decoder_k] = v
                                            loaded_count += 1
                                        else:
                                            # Skip printing individual warnings for relative_position_index and attn_mask
                                            # These are computed based on window_size and will be recomputed during model initialization
                                            if "relative_position_index" not in decoder_k and "attn_mask" not in decoder_k:
                                                print(f"   ⚠️  Shape mismatch for {decoder_k}: pretrained {v.shape} vs model {model_dict[decoder_k].shape}")
                                            skipped_count += 1
                                    else:
                                        skipped_count += 1
                                else:
                                    skipped_count += 1
                            else:
                                skipped_count += 1
                    except (ValueError, IndexError) as e:
                        skipped_count += 1
                        continue
                
                # Also check for direct decoder keys (if checkpoint already has layers_up)
                elif "layers_up" in k:
                    # Only load Swin block weights, not patch expanding
                    if any(block_key in k for block_key in ['blocks.', 'attn.', 'mlp.', 'norm']):
                        if any(skip in k for skip in ['upsample', 'expand', 'patch_expand']):
                            skipped_count += 1
                            continue
                        
                        if k in model_dict:
                            # Handle relative position bias table interpolation
                            if "relative_position_bias_table" in k:
                                if old_window_size != new_window_size:
                                    v = self._resize_relative_position_bias_table(v, old_window_size, new_window_size)
                                    interpolated_count += 1
                            
                            if v.shape == model_dict[k].shape:
                                full_dict[k] = v
                                loaded_count += 1
                            else:
                                skipped_count += 1
                        else:
                            skipped_count += 1
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
            
            if loaded_count > 0:
                # Load decoder weights
                msg = self.model.load_state_dict(full_dict, strict=False)
                print(f"   ✓ Loaded {loaded_count} decoder weight(s) from SimMIM checkpoint")
                if interpolated_count > 0:
                    print(f"   ✓ Interpolated {interpolated_count} relative position bias table(s) from window_size={old_window_size} to window_size={new_window_size}")
                # Note: relative_position_index and attn_mask shape mismatches are expected (computed from window_size, will be recomputed during init)
                if msg.missing_keys:
                    missing_important = [k for k in msg.missing_keys if 'patch_expand' not in k and 'upsample' not in k and 'expand' not in k]
                    if missing_important:
                        print(f"   ⚠️  {len(missing_important)} decoder weights not found in checkpoint (will use random initialization)")
                if msg.unexpected_keys:
                    print(f"   ℹ️  {len(msg.unexpected_keys)} unexpected keys in checkpoint (ignored)")
                return True
            else:
                print(f"   ⚠️  No matching decoder weights found in checkpoint")
                print(f"   Decoder will be randomly initialized")
                return False
                
        except Exception as e:
            print(f"   ⚠️  Failed to load decoder weights: {e}")
            import traceback
            traceback.print_exc()
            print(f"   Continuing without pretrained decoder weights (decoder will be randomly initialized)")
            return False


# Alias for compatibility
SwinUnet = CNNTransformerUnet
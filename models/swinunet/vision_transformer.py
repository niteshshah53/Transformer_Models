# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def _resize_relative_position_bias_table(self, table, old_window, new_window):
        """
        Interpolate relative position bias table from old_window to new_window.
        
        Args:
            table: Relative position bias table tensor with shape (old_size * old_size, num_heads)
            old_window: Original window size (e.g., 6 for SimMIM)
            new_window: Target window size (e.g., 7 for Swin-UNet)
            
        Returns:
            Interpolated table with shape (new_size * new_size, num_heads)
        """
        old_size = 2 * old_window - 1
        new_size = 2 * new_window - 1
        
        # Reshape to (1, old_size, old_size, num_heads)
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
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
                # print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            
            # Check if SimMIM config is being used
            is_simmim = self._is_simmim_config(config)
            old_window_size = 6  # SimMIM uses window_size=6
            new_window_size = config.MODEL.SWIN.WINDOW_SIZE  # Current config window size (typically 7)
            
            if "model"  not in pretrained_dict:
                # print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        # print("delete key:{}".format(k))
                        del pretrained_dict[k]
                
                # Interpolate relative position bias tables if SimMIM config is used
                if is_simmim and old_window_size != new_window_size:
                    interpolated_count = 0
                    for k in list(pretrained_dict.keys()):
                        if "relative_position_bias_table" in k:
                            if pretrained_dict[k].shape[0] == (2 * old_window_size - 1) ** 2:
                                pretrained_dict[k] = self._resize_relative_position_bias_table(
                                    pretrained_dict[k], old_window_size, new_window_size
                                )
                                interpolated_count += 1
                    if interpolated_count > 0:
                        print(f"SimMIM config detected: Interpolated {interpolated_count} relative position bias tables from window_size={old_window_size} to window_size={new_window_size}")
                
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            # print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    # Extract layer number more robustly (e.g., "layers.0" -> 0, "layers.1" -> 1)
                    try:
                        # Find the position after "layers."
                        layers_pos = k.find("layers.")
                        if layers_pos != -1:
                            # Extract the number after "layers."
                            start_pos = layers_pos + len("layers.")
                            # Find the next dot or end of string
                            end_pos = k.find(".", start_pos)
                            if end_pos == -1:
                                end_pos = len(k)
                            layer_str = k[start_pos:end_pos]
                            layer_num = int(layer_str)
                            current_layer_num = 3 - layer_num
                            # Reconstruct the key for layers_up
                            current_k = "layers_up." + str(current_layer_num) + k[end_pos:]
                            full_dict.update({current_k: v})
                    except (ValueError, IndexError) as e:
                        # Skip keys that don't match expected format
                        # print(f"Skipping key with unexpected format: {k} (error: {e})")
                        continue
            
            # Interpolate relative position bias tables if SimMIM config is used
            if is_simmim and old_window_size != new_window_size:
                interpolated_count = 0
                for k in list(full_dict.keys()):
                    if "relative_position_bias_table" in k:
                        # Check if this key needs interpolation
                        # The pretrained dict might have shape for old_window, but model_dict expects new_window
                        if k in model_dict:
                            expected_size = (2 * new_window_size - 1) ** 2
                            actual_size = full_dict[k].shape[0]
                            if actual_size == (2 * old_window_size - 1) ** 2 and model_dict[k].shape[0] == expected_size:
                                full_dict[k] = self._resize_relative_position_bias_table(
                                    full_dict[k], old_window_size, new_window_size
                                )
                                interpolated_count += 1
                if interpolated_count > 0:
                    print(f"SimMIM config detected: Interpolated {interpolated_count} relative position bias tables from window_size={old_window_size} to window_size={new_window_size}")
            
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                            # print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            # print("none pretrain")
            pass
 
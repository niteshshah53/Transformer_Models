import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import random
import torchvision.transforms.functional as TF

"""
Simple DIVAHISDB dataset loader.

This loader supports two modes:
 - use_patched_data=True: expects a folder layout like
     <root>/Image/<split>/*.png
     <root>/mask/<split>/*.png
   where mask images encode bitmask values in RGB channels as described in the dataset.
 - use_patched_data=False: will search for original images under nested img-* and pixel-level-gt-* folders

It decodes the DIVAHISDB bitmask PNGs into single-channel class indices (0..3) using a priority rule
when multiple bits are set. This keeps training simple (single-label per pixel).
"""

# Bit flags (from dataset paper)
BIT_BACKGROUND = 0x000001
BIT_COMMENT = 0x000002
BIT_DECORATION = 0x000004
BIT_MAIN_TEXT = 0x000008
BOUNDARY_FLAG = 0x800000

# Map bit to class index
BIT_TO_CLASS = {
    BIT_BACKGROUND: 0,
    BIT_COMMENT: 1,
    BIT_DECORATION: 2,
    BIT_MAIN_TEXT: 3,
}


def decode_bitmask_mask(mask_rgb):
    """Decode HxWx3 uint8 RGB mask to HxW class indices 0..3.

    If multiple bits are set, priority order is: main-text > comment > decoration.
    Boundary flag is ignored (not used for training labels).
    """
    # combine channels into integer bitmask
    r = mask_rgb[:, :, 0].astype(np.uint32)
    g = mask_rgb[:, :, 1].astype(np.uint32)
    b = mask_rgb[:, :, 2].astype(np.uint32)
    bitmask = (r << 16) | (g << 8) | b

    H, W = bitmask.shape
    labels = np.zeros((H, W), dtype=np.int64)

    # Apply priority: main text, then comment, then decoration
    for bit in (BIT_MAIN_TEXT, BIT_COMMENT, BIT_DECORATION):
        mask = (bitmask & bit) != 0
        labels[mask] = BIT_TO_CLASS[bit]

    # Background stays 0 where no other bits set
    return labels


def rgb_to_class(mask):
    """Compatibility wrapper used by training code (accepts numpy HxWx3 array)."""
    return decode_bitmask_mask(mask)


class DivaHisDBDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, patch_size=224, stride=224, use_patched_data=False):
        self.root_dir = root_dir
        self.split = split
        self.use_patched_data = use_patched_data
        self.patch_size = patch_size
        self.stride = stride

        if transform is None:
            # simple identity transforms; caller may pass a stronger transform
            self.transform = lambda img, mask: (img, mask)
        else:
            self.transform = transform

        if use_patched_data:
            self.img_paths, self.mask_paths = self._get_patched_file_paths()
        else:
            self.img_paths, self.mask_paths = self._get_original_file_paths()

    def _get_patched_file_paths(self):
        img_dir = os.path.join(self.root_dir, 'Image', self.split)
        mask_dir = os.path.join(self.root_dir, 'mask', self.split)
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            return [], []
        imgs = sorted(glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg')))
        masks = []
        img_paths = []
        mask_paths = []
        for p in imgs:
            base = os.path.splitext(os.path.basename(p))[0]
            # expect mask with same base name
            mp = os.path.join(mask_dir, base + '.png')
            if os.path.exists(mp):
                img_paths.append(p)
                mask_paths.append(mp)
        return img_paths, mask_paths

    def _get_original_file_paths(self):
        # Fallback: search for img-* / pixel-level-gt-* structure similar to other loaders
        img_paths = []
        mask_paths = []
        for img_dir in glob.glob(os.path.join(self.root_dir, '**', 'img-*', self.split), recursive=True):
            mask_dir = img_dir.replace('img-', 'pixel-level-gt-')
            if not os.path.isdir(mask_dir):
                continue
            for img_name in sorted(os.listdir(img_dir)):
                if not img_name.lower().endswith(('.jpg', '.png')):
                    continue
                img_path = os.path.join(img_dir, img_name)
                mask_name = os.path.splitext(img_name)[0] + '.png'
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    img_paths.append(img_path)
                    mask_paths.append(mask_path)
        return img_paths, mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        mask_np = np.array(mask)
        mask_class = decode_bitmask_mask(mask_np)

        # simple resize for whole-image mode to a standard size if not patched
        if not self.use_patched_data:
            image = image.resize((2016, 1344), Image.BILINEAR)
            mask_class = Image.fromarray(mask_class.astype(np.uint8)).resize((2016, 1344), Image.NEAREST)
            mask_class = np.array(mask_class)

        image, mask_class = self.transform(image, mask_class)
        image = TF.to_tensor(image)
        return {"image": image, "label": torch.from_numpy(mask_class).long(), "case_name": os.path.splitext(os.path.basename(img_path))[0]}

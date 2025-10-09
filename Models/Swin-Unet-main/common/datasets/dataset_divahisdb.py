import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import random
import torchvision.transforms.functional as TF
from functools import partial


def default_transform(image, mask_class, img_size=(224, 224)):
    """Simple transform like SwinUnet - just resize and basic flips/rotations."""
    # Resize
    image = image.resize(img_size, Image.BILINEAR)
    mask_class = Image.fromarray(mask_class.astype(np.uint8)).resize(img_size, Image.NEAREST)
    mask_class = np.array(mask_class)

    # Random horizontal flip (p=0.5)
    if random.random() >= 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask_class = np.fliplr(mask_class)

    # Random vertical flip (p=0.5)
    if random.random() >= 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        mask_class = np.flipud(mask_class)

    # Random rotation by 90, 180, or 270 degrees (p=0.5)
    if random.random() >= 0.5:
        angle = random.choice([90, 180, 270])
        image = image.rotate(angle, resample=Image.BILINEAR)
        mask_class = Image.fromarray(mask_class.astype(np.uint8)).rotate(angle, resample=Image.NEAREST)
        mask_class = np.array(mask_class)

    mask_class = mask_class.copy()
    return image, mask_class


def training_transform(img, mask, patch_size):
    """Training transform wrapper."""
    return default_transform(img, mask, img_size=(patch_size, patch_size))


def identity_transform(img, mask):
    """Identity transform for validation."""
    return img, mask

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

    Uses vectorized RGB color matching for DivaHisDB masks.
    """
    H, W = mask_rgb.shape[:2]
    labels = np.zeros((H, W), dtype=np.int64)
    
    # Convert to float for easier comparison
    mask_float = mask_rgb.astype(np.float32)
    
    # Define color ranges with tolerance for slight variations
    # Comment (green) - high green, low red and blue
    comment_mask = (mask_float[:,:,1] > 200) & (mask_float[:,:,0] < 100) & (mask_float[:,:,2] < 100)
    labels[comment_mask] = 1
    
    # Decoration (red) - high red, low green and blue  
    decoration_mask = (mask_float[:,:,0] > 200) & (mask_float[:,:,1] < 100) & (mask_float[:,:,2] < 100)
    labels[decoration_mask] = 2
    
    # Main Text (blue) - high blue, low red and green
    maintext_mask = (mask_float[:,:,2] > 200) & (mask_float[:,:,0] < 100) & (mask_float[:,:,1] < 100)
    labels[maintext_mask] = 3
    
    # Background (black) - everything else defaults to background (0)
    # No need to explicitly set background_mask since labels starts as zeros
    
    # Ensure all labels are in valid range [0, 3]
    labels = np.clip(labels, 0, 3)
    
    return labels


def rgb_to_class(mask):
    """Compatibility wrapper used by training code (accepts numpy HxWx3 array)."""
    return decode_bitmask_mask(mask)


class DivaHisDBDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, patch_size=224, stride=224, use_patched_data=False, manuscript=None, model_type=None):
        self.root_dir = root_dir
        self.split = split
        self.use_patched_data = use_patched_data
        self.patch_size = patch_size
        self.stride = stride
        self.manuscript = manuscript
        self.model_type = model_type

        if transform is None:
            # Use SSTRANS-specific transforms if model is SSTRANS
            if model_type and model_type.lower() == 'sstrans':
                if split == 'training':
                    from datasets.sstrans_transforms import sstrans_training_transform
                    self.transform = sstrans_training_transform(patch_size=patch_size)
                else:
                    from datasets.sstrans_transforms import sstrans_validation_transform
                    self.transform = sstrans_validation_transform(patch_size=patch_size)
            elif model_type and model_type.lower() in ['hybrid1', 'hybrid2']:
                # Use simple transforms like SwinUnet (no complex augmentation)
                if split == 'training':
                    from functools import partial
                    self.transform = partial(training_transform, patch_size=patch_size)
                else:
                    self.transform = identity_transform
            else:
                # Default identity transforms for other models (same as SwinUnet)
                self.transform = lambda img, mask: (img, mask)
        else:
            self.transform = transform

        if use_patched_data:
            self.img_paths, self.mask_paths = self._get_patched_file_paths()
        else:
            self.img_paths, self.mask_paths = self._get_original_file_paths()

    def _get_patched_file_paths(self):
        if self.manuscript:
            # Manuscript-specific path structure: root/manuscript/Image/split and root/manuscript/mask/split_labels
            img_dir = os.path.join(self.root_dir, self.manuscript, 'Image', self.split)
            mask_dir = os.path.join(self.root_dir, self.manuscript, 'mask', f'{self.split}_labels')
        else:
            # Original path structure: root/Image/split and root/mask/split
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
            # Look for mask with "_zones_NA" suffix (DIVA-HisDB patched format)
            mp = os.path.join(mask_dir, base + '_zones_NA.png')
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

        # Apply transforms
        if self.model_type and self.model_type.lower() == 'sstrans':
            # SSTRANS transforms handle tensor conversion and normalization
            image_tensor, mask_tensor = self.transform(image, mask_class)
        else:
            # Simple transforms for all other models (including hybrid)
            if not self.use_patched_data:
                image = image.resize((2016, 1344), Image.BILINEAR)
                mask_class = Image.fromarray(mask_class.astype(np.uint8)).resize((2016, 1344), Image.NEAREST)
                mask_class = np.array(mask_class)
                # Apply transforms only for non-patched data
                image, mask_class = self.transform(image, mask_class)
            
            image_tensor = TF.to_tensor(image)
            mask_tensor = torch.from_numpy(mask_class).long()
        
        return {"image": image_tensor, "label": mask_tensor, "case_name": os.path.splitext(os.path.basename(img_path))[0]}
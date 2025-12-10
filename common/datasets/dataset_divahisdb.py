import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
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


def class_aware_training_transform_divahisdb(image, mask_class, patch_size, use_aggressive_aug=False):
    """
    Class-aware training transform for DivaHisDB that applies stronger augmentation for rare classes.
    
    Class distribution for DivaHisDB (4 classes):
    - Background (0): 80.7% - common
    - Comment (1): 8.95% - rare (~9%)
    - Decoration (2): 0.355% - very rare (~0.36%)
    - Main Text (3): 9.99% - rare (~10%)
    
    Rare classes: Comment (1), Main Text (3) - ~9-10% frequency
    Very rare classes: Decoration (2) - ~0.36% frequency (extremely rare)
    
    Args:
        image: PIL Image
        mask_class: numpy array of class indices (H, W) with values 0-3
        patch_size: Target patch size
        use_aggressive_aug: If True, always use aggressive augmentation (for rare class samples)
    
    Returns:
        (augmented_image, augmented_mask_class)
    """
    # DivaHisDB class mapping:
    # 0 = Background (common, 80.7%)
    # 1 = Comment (rare, 8.95%)
    # 2 = Decoration (very rare, 0.36%)
    # 3 = Main Text (rare, 9.99%)
    
    rare_classes = {1, 3}  # Comment, Main Text (~9-10% frequency)
    very_rare_classes = {2}  # Decoration (~0.36% frequency - extremely rare)
    
    unique_classes = set(np.unique(mask_class))
    has_rare_class = use_aggressive_aug or bool(rare_classes.intersection(unique_classes))
    has_very_rare_class = use_aggressive_aug or bool(very_rare_classes.intersection(unique_classes))
    
    # Resize first
    image = image.resize((patch_size, patch_size), Image.BILINEAR)
    mask_class = Image.fromarray(mask_class.astype(np.uint8)).resize((patch_size, patch_size), Image.NEAREST)
    mask_class = np.array(mask_class)
    
    # Determine augmentation probabilities and ranges based on rarity
    # Note: Reduced aggressiveness to prevent training instability and gradient explosions
    if has_very_rare_class:
        # Moderate augmentation for Decoration (0.36% frequency - extremely rare)
        # Reduced from very aggressive to prevent NaN/Inf gradients
        rotation_prob = 0.6  # Reduced from 0.8
        flip_prob = 0.6  # Reduced from 0.8
        color_prob = 0.4  # Reduced from 0.7
        affine_prob = 0.15  # Reduced from 0.3 to minimize border artifacts and training instability
        rotation_range = (-25, 25)  # Reduced from (-40, 40)
        affine_rotation_range = (-12, 12)  # Reduced from (-18, 18)
        translate_range = (-0.08, 0.08)  # Reduced from (-0.12, 0.12) to minimize synthetic pixels
        scale_range = (0.94, 1.06)  # Reduced from (0.88, 1.12) to minimize synthetic pixels
    elif has_rare_class:
        # Moderate augmentation for Comment & Main Text (9-10% frequency)
        rotation_prob = 0.5  # Reduced from 0.7
        flip_prob = 0.5  # Reduced from 0.7
        color_prob = 0.3  # Reduced from 0.6
        affine_prob = 0.15  # Reduced from 0.25 to minimize border artifacts and training instability
        rotation_range = (-20, 20)  # Reduced from (-30, 30)
        affine_rotation_range = (-10, 10)  # Reduced from (-15, 15)
        translate_range = (-0.06, 0.06)  # Reduced from (-0.08, 0.08)
        scale_range = (0.96, 1.04)  # Reduced from (0.92, 1.08)
    else:
        # Standard augmentation for common classes (Background - 80.7%)
        rotation_prob = 0.5
        flip_prob = 0.5
        color_prob = 0.0
        affine_prob = 0.0
        rotation_range = (-15, 15)
        affine_rotation_range = (0, 0)
        translate_range = (0, 0)
        scale_range = (1.0, 1.0)
    
    # Apply random rotation
    if random.random() < rotation_prob and rotation_range[1] > 0:
        angle = random.uniform(*rotation_range)
        image = image.rotate(angle, resample=Image.BILINEAR)
        mask_class = Image.fromarray(mask_class.astype(np.uint8)).rotate(angle, resample=Image.NEAREST)
        mask_class = np.array(mask_class)
    
    # Apply random flips
    if random.random() < flip_prob:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask_class = np.fliplr(mask_class)
    if random.random() < flip_prob:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        mask_class = np.flipud(mask_class)
    
    # Apply color jitter (only for rare classes)
    if random.random() < color_prob:
        jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        image = jitter(image)
    
    # Apply random affine (rotation, translation, scale)
    # Only for rare classes (affine_prob > 0)
    if random.random() < affine_prob and affine_rotation_range[1] > 0:
        angle = random.uniform(*affine_rotation_range)
        translate = (random.uniform(*translate_range) * patch_size, random.uniform(*translate_range) * patch_size)
        scale = random.uniform(*scale_range)
        # For images: use fill=None to replicate edge pixels (avoids black borders that create spurious edges)
        # Black pixels (fill=0) create artificial high-contrast edges that the model may learn spuriously
        image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=0, fill=None)
        # For masks: use fill=0 (Background class) - synthetic pixels at borders are treated as Background
        # This is acceptable since Background is the dominant class and these pixels are at borders
        mask_pil = Image.fromarray(mask_class.astype(np.uint8))
        mask_pil = TF.affine(mask_pil, angle=angle, translate=translate, scale=scale, shear=0, fill=0)
        mask_class = np.array(mask_pil)
    
    mask_class = mask_class.copy()
    return image, mask_class

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
    
    Optimized vectorized decoding for DivaHisDB masks.
    Uses single-pass vectorized operations for maximum performance.
    """
    # Extract channels directly (no float conversion needed for uint8 comparison)
    r, g, b = mask_rgb[:,:,0], mask_rgb[:,:,1], mask_rgb[:,:,2]
    
    # Initialize with Background (0)
    labels = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
    
    # Use single vectorized operations with priority order
    # Priority: Comment (green) > Decoration (red) > Main Text (blue)
    # Comment (green): high G (>200), low R&B (<100)
    labels[(g > 200) & (r < 100) & (b < 100)] = 1
    
    # Decoration (red): high R (>200), low G&B (<100) - overwrites Comment if both conditions met
    labels[(r > 200) & (g < 100) & (b < 100)] = 2
    
    # Main Text (blue): high B (>200), low R&G (<100)
    labels[(b > 200) & (r < 100) & (g < 100)] = 3
    
    # Background (0) is already set as default, no need to explicitly set
    
    return labels


def rgb_to_class(mask):
    """Compatibility wrapper used by training code (accepts numpy HxWx3 array)."""
    return decode_bitmask_mask(mask)


class DivaHisDBDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, patch_size=224, stride=224, use_patched_data=False, manuscript=None, model_type=None, use_class_aware_aug=False):
        self.root_dir = root_dir
        self.split = split
        self.use_patched_data = use_patched_data
        self.patch_size = patch_size
        self.stride = stride
        self.manuscript = manuscript
        self.model_type = model_type
        self.use_class_aware_aug = use_class_aware_aug

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
                # Use class-aware augmentation if enabled, otherwise standard transforms
                if split == 'training':
                    if use_class_aware_aug:
                        self.transform = partial(class_aware_training_transform_divahisdb, patch_size=patch_size)
                    else:
                        self.transform = partial(training_transform, patch_size=patch_size)
                else:
                    self.transform = identity_transform
            elif model_type and model_type.lower() == 'network':
                # Network model: Use identity transform for DivaHisDB (like SwinUnet) to prevent gradient explosions
                # SwinUnet uses no augmentation and achieves <0.1% skipped batches
                # Simple transforms cause 50-65% skipped batches for network model
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
        """Optimized file path construction - reduces file system calls."""
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
        
        # Get all masks first (single glob call, faster than checking existence per image)
        all_masks = {os.path.basename(p): p for p in glob.glob(os.path.join(mask_dir, '*_zones_NA.png'))}
        
        # Get all images
        imgs = sorted(glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg')))
        
        # Match images with masks using dictionary lookup (O(1) instead of O(n) file existence check)
        img_paths = []
        mask_paths = []
        for p in imgs:
            base = os.path.splitext(os.path.basename(p))[0]
            mask_name = base + '_zones_NA.png'
            if mask_name in all_masks:
                img_paths.append(p)
                mask_paths.append(all_masks[mask_name])
        
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
            # Simple transforms for all other models (including hybrid/network)
            # Note: Removed redundant resize for non-patched data - transform will handle resizing to patch_size
            # This eliminates one unnecessary resize operation
            
            # Apply transforms (for both patched and non-patched data)
            # Transform will resize to patch_size if needed
            image, mask_class = self.transform(image, mask_class)
            
            # Convert to tensor (single conversion, no redundant PIL->numpy->PIL conversions)
            image_tensor = TF.to_tensor(image)
            
            # Apply ImageNet normalization for Hybrid1/Network (EfficientNet encoder)
            if self.model_type and self.model_type.lower() in ['hybrid1', 'network']:
                image_tensor = TF.normalize(
                    image_tensor,
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean
                    std=[0.229, 0.224, 0.225]     # ImageNet std
                )
            
            # Convert mask directly from numpy to tensor (avoid intermediate conversions)
            mask_tensor = torch.from_numpy(mask_class).long()
        
        return {"image": image_tensor, "label": mask_tensor, "case_name": os.path.splitext(os.path.basename(img_path))[0]}
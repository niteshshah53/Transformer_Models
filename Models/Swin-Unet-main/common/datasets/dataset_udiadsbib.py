import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import random


import torchvision.transforms.functional as TF
from torchvision import transforms as tvtf
from functools import partial
import multiprocessing
from scipy import ndimage
from PIL import ImageFilter
import math
import numpy as np

def default_transform(image, mask_class, img_size=(448, 448)):
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

# Top-level transform functions for Windows compatibility
def training_transform(img, mask, patch_size):
    return default_transform(img, mask, img_size=(patch_size, patch_size))


def strong_training_transform(img, mask, patch_size,
                              prob_flip=0.5,
                              prob_rotate=0.3,
                              prob_brightness=0.5,
                              prob_contrast=0.5,
                              prob_gamma=0.3,
                              prob_blur=0.25,
                              prob_elastic=0.2,
                              max_elastic_alpha=34,
                              elastic_sigma=4):
    """
    Strong augmentations for document segmentation. All inputs are PIL image and
    numpy mask (H,W). Returns augmented (PIL image, numpy mask).
    """
    # Ensure PIL image, numpy mask
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))

    # Resize to patch size first (if needed)
    if img.size != (patch_size, patch_size):
        img = img.resize((patch_size, patch_size), Image.BILINEAR)
        mask = Image.fromarray(mask.astype(np.uint8)).resize((patch_size, patch_size), Image.NEAREST)
        mask = np.array(mask)

    # Random flips
    if random.random() < prob_flip:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = np.flipud(mask)

    # Small random rotation (90-degree rotations handled elsewhere)
    if random.random() < prob_rotate:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BILINEAR)
        mask = Image.fromarray(mask.astype(np.uint8)).rotate(angle, resample=Image.NEAREST)
        mask = np.array(mask)

    # Photometric transforms (cautious)
    # Brightness
    if random.random() < prob_brightness:
        factor = random.uniform(0.8, 1.2)
        img = TF.adjust_brightness(img, factor)

    # Contrast
    if random.random() < prob_contrast:
        factor = random.uniform(0.85, 1.15)
        img = TF.adjust_contrast(img, factor)

    # Gamma
    if random.random() < prob_gamma:
        gamma = random.uniform(0.9, 1.2)
        img = TF.adjust_gamma(img, gamma)

    # Gaussian blur
    if random.random() < prob_blur:
        radius = random.uniform(0.0, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Elastic deformation (applied to both image and mask)
    if random.random() < prob_elastic:
        alpha = random.uniform(0, max_elastic_alpha)
        sigma = elastic_sigma
        # convert to numpy arrays
        img_np = np.array(img)
        mask_np = np.array(mask)
        shape = img_np.shape[:2]

        # generate displacement fields
        dx = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha
        dy = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # remap each channel
        warped = np.zeros_like(img_np)
        for c in range(img_np.shape[2]):
            warped[:, :, c] = ndimage.map_coordinates(img_np[:, :, c], indices, order=1, mode='reflect').reshape(shape)

        # remap mask with nearest neighbor
        warped_mask = ndimage.map_coordinates(mask_np, indices, order=0, mode='reflect').reshape(shape)

        img = Image.fromarray(warped.astype(np.uint8))
        mask = warped_mask

    mask = mask.copy()
    return img, mask

def identity_transform(img, mask):
    return img, mask


def class_aware_training_transform(image, mask_class, patch_size, num_classes=6, use_aggressive_aug=False):
    """
    Class-aware training transform that applies stronger augmentation for rare classes.
    
    Rare classes (for U-DIADS-Bib) - from actual training statistics:
    - Paratext (1): ~5-8% (rare)
    - Decoration (2): ~2-3% (rare)
    - Title (4): ~1-2% (very rare)
    - Chapter Headings (5): ~<1% (extremely rare) - only for 6-class datasets
    
    Common classes:
    - Background (0): ~60-70%
    - Main Text (3): ~20-25%
    
    Note: Actual percentages may vary by manuscript. Check compute_class_weights() output
    for manuscript-specific statistics.
    
    Args:
        image: PIL Image
        mask_class: numpy array of class indices (H, W)
        patch_size: Target patch size
        num_classes: Number of classes (5 for Syr341, 6 for other manuscripts)
        use_aggressive_aug: If True, always use aggressive augmentation (for rare class samples)
    
    Returns:
        (augmented_image, augmented_mask_class)
    """
    # Check if mask contains rare classes and determine rarity level
    # Rare classes: Paratext (1), Decoration (2), Title (4), Chapter Headings (5) if num_classes==6
    # Very rare classes: Title (4), Chapter Headings (5) if num_classes==6 - <2% frequency
    if num_classes == 5:
        # For Syr341 (5 classes): no Chapter Headings (class 5)
        rare_classes = {1, 2, 4}  # Paratext, Decoration, Title
        very_rare_classes = {4}  # Title only (<2% frequency)
    else:  # num_classes == 6 (default)
        rare_classes = {1, 2, 4, 5}  # Paratext, Decoration, Title, Chapter Headings
        very_rare_classes = {4, 5}  # Title, Chapter Heading (<2% frequency)
    
    unique_classes = set(np.unique(mask_class))
    has_rare_class = use_aggressive_aug or bool(rare_classes.intersection(unique_classes))
    has_very_rare_class = use_aggressive_aug or bool(very_rare_classes.intersection(unique_classes))
    
    # Resize first
    image = image.resize((patch_size, patch_size), Image.BILINEAR)
    mask_class = Image.fromarray(mask_class.astype(np.uint8)).resize((patch_size, patch_size), Image.NEAREST)
    mask_class = np.array(mask_class)
    
    # Determine augmentation probabilities and ranges based on rarity
    # Note: Reduced aggressiveness to prevent training instability
    # Conservative fix: Using fill=0 (Background) instead of fill=255 (ignore_index) for affine transforms
    # Synthetic pixels are treated as Background, which is acceptable since Background is the dominant class
    if has_very_rare_class:
        # VERY aggressive augmentation for Title & Chapter Heading (<2% frequency)
        # Reduced probabilities to prevent training instability
        rotation_prob = 0.7  # Reduced from 0.8
        flip_prob = 0.7  # Reduced from 0.8
        color_prob = 0.6  # Reduced from 0.7
        affine_prob = 0.2  # Reduced from 0.4 to minimize border artifacts and training instability
        rotation_range = (-35, 35)  # Reduced from (-40, 40)
        affine_rotation_range = (-15, 15)  # Reduced from (-18, 18)
        translate_range = (-0.10, 0.10)  # Further reduced from (-0.12, 0.12) to minimize synthetic pixels
        scale_range = (0.90, 1.10)  # Further reduced from (0.88, 1.12) to minimize synthetic pixels
    elif has_rare_class:
        # Aggressive augmentation for Paratext & Decoration (2-8% frequency)
        rotation_prob = 0.6  # Reduced from 0.7
        flip_prob = 0.6  # Reduced from 0.7
        color_prob = 0.5  # Reduced from 0.6
        affine_prob = 0.2  # Reduced from 0.3 to minimize border artifacts and training instability
        rotation_range = (-25, 25)  # Reduced from (-30, 30)
        affine_rotation_range = (-12, 12)  # Reduced from (-15, 15)
        translate_range = (-0.06, 0.06)  # Further reduced from (-0.08, 0.08)
        scale_range = (0.94, 1.06)  # Further reduced from (0.92, 1.08)
    else:
        # Standard augmentation for common classes (Background, Main Text)
        rotation_prob = 0.5
        flip_prob = 0.5
        color_prob = 0.0
        affine_prob = 0.0
        rotation_range = (0, 0)  # No rotation for common classes
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
        jitter = tvtf.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        image = jitter(image)
    
    # Apply random affine (rotation, translation, scale)
    # Only for rare classes (affine_prob > 0)
    # Conservative fix: Use fill=0 (Background) instead of fill=255 (ignore_index) to prevent training instability
    if random.random() < affine_prob and affine_rotation_range[1] > 0:
        angle = random.uniform(*affine_rotation_range)
        translate = (random.uniform(*translate_range), random.uniform(*translate_range))
        scale = random.uniform(*scale_range)
        image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=0, fill=0)
        # Apply same affine to mask
        # Use fill=0 (Background class) for mask - synthetic pixels will be treated as Background
        # This is acceptable since Background is the dominant class and these pixels are at borders
        # Conservative fix: No ignore_index to prevent training instability from too many ignored pixels
        mask_pil = Image.fromarray(mask_class.astype(np.uint8))
        mask_pil = TF.affine(mask_pil, angle=angle, translate=translate, scale=scale, shear=0, fill=0)
        mask_class = np.array(mask_pil)
    
    # For common classes, apply standard 90-degree rotations if no rotation was applied
    # (This maintains backward compatibility with standard augmentation)
    if not has_rare_class and random.random() >= 0.5:
        angle = random.choice([90, 180, 270])
        image = image.rotate(angle, resample=Image.BILINEAR)
        mask_class = Image.fromarray(mask_class.astype(np.uint8)).rotate(angle, resample=Image.NEAREST)
        mask_class = np.array(mask_class)
    
    mask_class = mask_class.copy()
    return image, mask_class

# U-DIADS-Bib color to class index mapping (standard 6-class version)
COLOR_MAP_6_CLASSES = {
    (0, 0, 0): 0,         # Background
    (255, 255, 0): 1,     # Paratext (Yellow)
    (0, 255, 255): 2,     # Decoration (Cyan)
    (255, 0, 255): 3,     # Main Text (Magenta)
    (255, 0, 0): 4,       # Title (Red)
    (0, 255, 0): 5,       # Chapter Headings (Lime)
}

# U-DIADS-Bib color to class index mapping for Syriaque341 (5-class version, no Chapter Headings)
COLOR_MAP_5_CLASSES = {
    (0, 0, 0): 0,         # Background
    (255, 255, 0): 1,     # Paratext (Yellow)
    (0, 255, 255): 2,     # Decoration (Cyan)
    (255, 0, 255): 3,     # Main Text (Magenta)
    (255, 0, 0): 4,       # Title (Red)
    # Note: Chapter Headings (0, 255, 0) not present in Syriaque341
}

def rgb_to_class(mask, num_classes=6):
    """
    Convert RGB mask to class indices.
    
    Args:
        mask: RGB mask image
        num_classes: Number of classes (5 for Syriaque341, 6 for others)
    
    Returns:
        Class index mask
    """
    mask_class = np.zeros(mask.shape[:2], dtype=np.int64)
    
    # Choose appropriate color map based on number of classes
    if num_classes == 5:
        color_map = COLOR_MAP_5_CLASSES
        # For 5-class mode, explicitly handle Chapter Headings pixels
        # Map them to Background (0) since they don't exist in Syr341FS
        chapter_headings_color = (0, 255, 0)
    else:  # Default to 6 classes
        color_map = COLOR_MAP_6_CLASSES
        chapter_headings_color = None
    
    # Track which pixels have been mapped
    mapped_mask = np.zeros(mask.shape[:2], dtype=bool)
    
    for rgb, cls in color_map.items():
        matches = np.all(mask == rgb, axis=-1)
        mask_class[matches] = cls
        mapped_mask[matches] = True
    
    # For 5-class mode: explicitly map Chapter Headings to Background
    if num_classes == 5 and chapter_headings_color is not None:
        chapter_matches = np.all(mask == chapter_headings_color, axis=-1)
        mask_class[chapter_matches] = 0  # Map to Background
        mapped_mask[chapter_matches] = True
    
    # Check for unmapped pixels and warn if found (only warn once per call)
    unmapped_pixels = ~mapped_mask
    if np.any(unmapped_pixels):
        unique_colors = np.unique(mask[unmapped_pixels].reshape(-1, 3), axis=0)
        # Only print warning if significant number of unmapped pixels (>0.1% of image)
        unmapped_ratio = np.sum(unmapped_pixels) / mask.size
        if unmapped_ratio > 0.001:  # More than 0.1% unmapped
            print(f"⚠️  WARNING: Found {np.sum(unmapped_pixels)} unmapped pixels ({unmapped_ratio*100:.2f}%)!")
            print(f"   Unique unmapped colors: {unique_colors.tolist()}")
            print(f"   Mapping them to Background (class 0)")
        mask_class[unmapped_pixels] = 0
    
    return mask_class


import glob

class UDiadsBibDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, patch_size=448, stride=224, use_patched_data=False, manuscript=None, model_type=None, num_classes=6, use_class_aware_aug=False):
        self.use_patched_data = use_patched_data
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size if not use_patched_data and split == 'training' else None
        self.stride = stride if not use_patched_data and split == 'training' else None
        self.manuscript = manuscript
        self.model_type = model_type
        self.num_classes = num_classes
        self.use_class_aware_aug = use_class_aware_aug
        
        # Set up the transform (no lambdas, only top-level functions)
        if transform is None:
            # Use SSTRANS-specific transforms if model is SSTRANS
            if model_type and model_type.lower() == 'sstrans':
                if split == 'training':
                    from datasets.sstrans_transforms import sstrans_training_transform
                    self.transform = sstrans_training_transform(patch_size=patch_size)
                else:
                    from datasets.sstrans_transforms import sstrans_validation_transform
                    self.transform = sstrans_validation_transform(patch_size=patch_size)
            elif model_type and model_type.lower() in ['network', 'hybrid2']:
                # Use simple transforms like SwinUnet (no complex augmentation)
                # Note: 'network' is the new name for the model previously called 'hybrid1'
                if split == 'training':
                    # Use class-aware augmentation if enabled, otherwise standard
                    if use_class_aware_aug:
                        self.transform = partial(class_aware_training_transform, patch_size=patch_size, num_classes=num_classes)
                    else:
                        # bind patch_size so the transform is a picklable top-level callable
                        self.transform = partial(training_transform, patch_size=patch_size)
                else:
                    self.transform = identity_transform
            else:
                # Default transforms for other models. For the 'network' model
                # (previously called 'hybrid1') we use stronger augmentations to improve generalization.
                if split == 'training':
                    # Use class-aware augmentation if enabled
                    if use_class_aware_aug:
                        self.transform = partial(class_aware_training_transform, patch_size=patch_size, num_classes=num_classes)
                    else:
                        # bind patch_size so the transform is a picklable top-level callable
                        # Network model uses EfficientNet-B4 encoder, so use strong augmentation
                        if model_type and model_type.lower() == 'network':
                            self.transform = partial(strong_training_transform, patch_size=patch_size)
                        else:
                            self.transform = partial(training_transform, patch_size=patch_size)
                else:
                    self.transform = identity_transform
        else:
            self.transform = transform
            
        # Get file paths based on whether we're using patched data or not
        if use_patched_data:
            self.img_paths, self.mask_paths = self._get_patched_file_paths()
        else:
            self.img_paths, self.mask_paths = self._get_original_file_paths()
            
            # For original data with patch extraction, prepare patches for training
            if not use_patched_data and self.patch_size is not None and split == 'training':
                self.patches = []
                self._prepare_patches()

    def _get_original_file_paths(self):
        """Get file paths for original (non-patched) data"""
        img_paths = []
        mask_paths = []
        img_dirs = glob.glob(os.path.join(self.root_dir, '**', f'img-*', self.split), recursive=True)
        for img_dir in img_dirs:
            manuscript = os.path.basename(os.path.dirname(img_dir))
            mask_dir = os.path.join(os.path.dirname(img_dir).replace('img-', 'pixel-level-gt-'), self.split)
            if not os.path.isdir(mask_dir):
                continue
            for img_name in sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]):
                mask_name = img_name.replace('.jpg', '.png').replace('.JPG', '.png')
                img_path = os.path.join(img_dir, img_name)
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    img_paths.append(img_path)
                    mask_paths.append(mask_path)
        return img_paths, mask_paths
        
    def _get_patched_file_paths(self):
        """Get file paths for pre-generated patches"""
        img_paths = []
        mask_paths = []
        
        if self.manuscript:
            # Manuscript-specific path structure: root/manuscript/Image/split and root/manuscript/mask/split_labels
            img_dir = os.path.join(self.root_dir, self.manuscript, 'Image', self.split)
            mask_dir = os.path.join(self.root_dir, self.manuscript, 'mask', f'{self.split}_labels')
        else:
            # Original path structure: root/Image/split and root/mask/split
            img_dir = os.path.join(self.root_dir, 'Image', self.split)
            mask_dir = os.path.join(self.root_dir, 'mask', self.split)
        
        if multiprocessing.current_process().name == 'MainProcess':
            print(f"Looking for images in: {img_dir}")
            print(f"Looking for masks in: {mask_dir}")
        
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            if multiprocessing.current_process().name == 'MainProcess':
                print(f"Warning: One of the directories does not exist!")
                # Check what directories actually exist to help debug
                if self.manuscript:
                    manuscript_dir = os.path.join(self.root_dir, self.manuscript)
                    if os.path.exists(manuscript_dir):
                        print(f"Available directories in {manuscript_dir}:")
                        for item in sorted(os.listdir(manuscript_dir)):
                            item_path = os.path.join(manuscript_dir, item)
                            if os.path.isdir(item_path):
                                subdirs = [d for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d))]
                                print(f"  {item}/: {', '.join(subdirs) if subdirs else '(empty)'}")
                    else:
                        print(f"Manuscript directory does not exist: {manuscript_dir}")
                        print(f"Available manuscripts in {self.root_dir}:")
                        if os.path.exists(self.root_dir):
                            manuscripts = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
                            print(f"  {', '.join(sorted(manuscripts)) if manuscripts else '(none found)'}")
                else:
                    if os.path.exists(self.root_dir):
                        print(f"Available directories in {self.root_dir}:")
                        for item in sorted(os.listdir(self.root_dir)):
                            item_path = os.path.join(self.root_dir, item)
                            if os.path.isdir(item_path):
                                print(f"  {item}/")
            return [], []
            
        # Get all patch images
        patch_imgs = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        if multiprocessing.current_process().name == 'MainProcess':
            print(f"Found {len(patch_imgs)} patch images")
        
        for img_path in patch_imgs:
            # Extract the base name (without extension)
            img_basename = os.path.basename(img_path)
            img_basename_no_ext = os.path.splitext(img_basename)[0]
            
            # Construct the mask filename directly
            mask_basename = f"{img_basename_no_ext}_zones_NA.png"
            mask_path = os.path.join(mask_dir, mask_basename)
            
            if os.path.exists(mask_path):
                img_paths.append(img_path)
                mask_paths.append(mask_path)
        
        if multiprocessing.current_process().name == 'MainProcess':
            print(f"Successfully matched {len(img_paths)} image-mask pairs")
            if len(img_paths) == 0:
                print(f"ERROR: No image-mask pairs found!")
                print(f"  Image directory exists: {os.path.exists(img_dir)}")
                print(f"  Mask directory exists: {os.path.exists(mask_dir)}")
                if os.path.exists(img_dir):
                    img_files = glob.glob(os.path.join(img_dir, '*.png'))
                    print(f"  Image files found: {len(img_files)}")
                    if len(img_files) > 0:
                        print(f"  Example image file: {os.path.basename(img_files[0])}")
                if os.path.exists(mask_dir):
                    mask_files = glob.glob(os.path.join(mask_dir, '*.png'))
                    print(f"  Mask files found: {len(mask_files)}")
                    if len(mask_files) > 0:
                        print(f"  Example mask file: {os.path.basename(mask_files[0])}")
        return img_paths, mask_paths

    def _prepare_patches(self):
        # For each image, store (img_idx, x, y) for each patch
        for idx in range(len(self.img_paths)):
            image = Image.open(self.img_paths[idx]).convert("RGB")
            w, h = image.size
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    self.patches.append((idx, x, y))

    def __len__(self):
        if not self.use_patched_data and self.patch_size is not None and self.split == 'training':
            return len(self.patches)
        else:
            return len(self.img_paths)

    def __getitem__(self, idx):
        if not self.use_patched_data and self.patch_size is not None and self.split == 'training':
            # Patch-based mode for original data
            img_idx, x, y = self.patches[idx]
            img_path = self.img_paths[img_idx]
            mask_path = self.mask_paths[img_idx]
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")
            # Crop patch
            image_patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))
            mask_patch = mask.crop((x, y, x + self.patch_size, y + self.patch_size))
            mask_class = rgb_to_class(np.array(mask_patch), self.num_classes)
            
            if self.model_type and self.model_type.lower() == 'sstrans':
                # SSTRANS transforms handle tensor conversion and normalization
                image_tensor, mask_tensor = self.transform(image_patch, mask_class)
            else:
                # Simple transforms for all other models (including hybrid)
                image_patch, mask_class = self.transform(image_patch, mask_class)
                image_tensor = TF.to_tensor(image_patch)
                
                # Apply ImageNet normalization for models with EfficientNet encoder
                # Network model uses EfficientNet-B4 encoder (pretrained on ImageNet)
                # Note: 'network' is the new name for the model previously called 'hybrid1'
                # Hybrid2 uses Swin encoder, so it doesn't need ImageNet normalization
                if self.model_type and self.model_type.lower() in ['network', 'swinunet']:
                    image_tensor = TF.normalize(
                        image_tensor,
                        mean=[0.485, 0.456, 0.406],  # ImageNet mean
                        std=[0.229, 0.224, 0.225]     # ImageNet std
                    )
                
                mask_tensor = torch.from_numpy(mask_class).long()
            
            case_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_x{x}_y{y}"
            return {"image": image_tensor, "label": mask_tensor, "case_name": case_name}
        elif self.use_patched_data:
            # Pre-generated patches mode
            img_path = self.img_paths[idx]
            mask_path = self.mask_paths[idx]
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")
            mask_class = rgb_to_class(np.array(mask), self.num_classes)
            
            if self.model_type and self.model_type.lower() == 'sstrans':
                # SSTRANS transforms handle tensor conversion and normalization
                image_tensor, mask_tensor = self.transform(image, mask_class)
            else:
                # Simple transforms for all other models (including hybrid)
                # For pre-generated patches, apply transforms but don't resize (they're already patch-sized)
                image, mask_class = self.transform(image, mask_class)
                image_tensor = TF.to_tensor(image)
                
                # Apply ImageNet normalization for models with EfficientNet encoder
                # Network model uses EfficientNet-B4 encoder (pretrained on ImageNet)
                # Note: 'network' is the new name for the model previously called 'hybrid1'
                # Hybrid2 uses Swin encoder, so it doesn't need ImageNet normalization
                if self.model_type and self.model_type.lower() in ['network', 'swinunet']:
                    image_tensor = TF.normalize(
                        image_tensor,
                        mean=[0.485, 0.456, 0.406],  # ImageNet mean
                        std=[0.229, 0.224, 0.225]     # ImageNet std
                    )
                
                mask_tensor = torch.from_numpy(mask_class).long()
            
            # Just use the filename (without extension) as the case name
            filename = os.path.basename(img_path)
            case_name = os.path.splitext(filename)[0]
            return {"image": image_tensor, "label": mask_tensor, "case_name": case_name}
        else:
            # Whole-image mode (for inference)
            img_path = self.img_paths[idx]
            mask_path = self.mask_paths[idx]
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")
            mask_class = rgb_to_class(np.array(mask), self.num_classes)
            # Resize to original image size (2016x1344) for model input (W, H)
            target_size = (2016, 1344)
            image = image.resize(target_size, Image.BILINEAR)
            mask_class = Image.fromarray(mask_class.astype(np.uint8)).resize(target_size, Image.NEAREST)
            mask_class = np.array(mask_class)
            
            if self.model_type and self.model_type.lower() == 'sstrans':
                # SSTRANS transforms handle tensor conversion and normalization
                image_tensor, mask_tensor = self.transform(image, mask_class)
            else:
                # Simple transforms for all other models (including hybrid)
                image, mask_class = self.transform(image, mask_class)
                image_tensor = TF.to_tensor(image)
                
                # Apply ImageNet normalization for models with EfficientNet encoder
                # Network model uses EfficientNet-B4 encoder (pretrained on ImageNet)
                # Note: 'network' is the new name for the model previously called 'hybrid1'
                # Hybrid2 uses Swin encoder, so it doesn't need ImageNet normalization
                if self.model_type and self.model_type.lower() in ['network', 'swinunet']:
                    image_tensor = TF.normalize(
                        image_tensor,
                        mean=[0.485, 0.456, 0.406],  # ImageNet mean
                        std=[0.229, 0.224, 0.225]     # ImageNet std
                    )
                
                mask_tensor = torch.from_numpy(mask_class).long()
            
            case_name = os.path.splitext(os.path.basename(img_path))[0]
            return {"image": image_tensor, "label": mask_tensor, "case_name": case_name}
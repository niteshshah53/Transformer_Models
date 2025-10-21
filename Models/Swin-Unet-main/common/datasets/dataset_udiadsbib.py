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

def identity_transform(img, mask):
    return img, mask

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
    else:  # Default to 6 classes
        color_map = COLOR_MAP_6_CLASSES
    
    for rgb, cls in color_map.items():
        matches = np.all(mask == rgb, axis=-1)
        mask_class[matches] = cls
    return mask_class


import glob

class UDiadsBibDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, patch_size=448, stride=224, use_patched_data=False, manuscript=None, model_type=None, num_classes=6):
        self.use_patched_data = use_patched_data
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size if not use_patched_data and split == 'training' else None
        self.stride = stride if not use_patched_data and split == 'training' else None
        self.manuscript = manuscript
        self.model_type = model_type
        self.num_classes = num_classes
        
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
            elif model_type and model_type.lower() in ['hybrid1', 'hybrid2']:
                # Use simple transforms like SwinUnet (no complex augmentation)
                if split == 'training':
                    # bind patch_size so the transform is a picklable top-level callable
                    self.transform = partial(training_transform, patch_size=patch_size)
                else:
                    self.transform = identity_transform
            else:
                # Default transforms for other models (same as SwinUnet)
                if split == 'training':
                    # bind patch_size so the transform is a picklable top-level callable
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
        return img_paths, mask_paths
            
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
                
                # Apply ImageNet normalization for Hybrid1 (EfficientNet encoder)
                if self.model_type and self.model_type.lower() == 'hybrid1':
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
                
                # Apply ImageNet normalization for Hybrid1 (EfficientNet encoder)
                if self.model_type and self.model_type.lower() == 'hybrid1':
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
                
                # Apply ImageNet normalization for Hybrid1 (EfficientNet encoder)
                if self.model_type and self.model_type.lower() == 'hybrid1':
                    image_tensor = TF.normalize(
                        image_tensor,
                        mean=[0.485, 0.456, 0.406],  # ImageNet mean
                        std=[0.229, 0.224, 0.225]     # ImageNet std
                    )
                
                mask_tensor = torch.from_numpy(mask_class).long()
            
            case_name = os.path.splitext(os.path.basename(img_path))[0]
            return {"image": image_tensor, "label": mask_tensor, "case_name": case_name}
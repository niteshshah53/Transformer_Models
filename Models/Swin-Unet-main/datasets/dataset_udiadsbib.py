import torchvision.transforms.functional as TF
from torchvision import transforms as tvtf

# Default transform for resizing images and masks to 224x224
import random
def default_transform(image, mask_class, img_size=(448, 448)):
    # Resize
    image = image.resize(img_size, Image.BILINEAR)
    mask_class = Image.fromarray(mask_class.astype(np.uint8)).resize(img_size, Image.NEAREST)
    mask_class = np.array(mask_class)

    # Color jitter (only image)
    if random.random() > 0.5:
        color_jitter = tvtf.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        image = color_jitter(image)

    # Random crop (with padding if needed)
    if random.random() > 0.5:
        crop_size = int(img_size[0] * 0.85)  # slightly smaller than patch size
        i = random.randint(0, img_size[1] - crop_size)
        j = random.randint(0, img_size[0] - crop_size)
        image = image.crop((j, i, j + crop_size, i + crop_size))
        mask_class = mask_class[i:i+crop_size, j:j+crop_size]
        # Resize back to patch size
        image = image.resize(img_size, Image.BILINEAR)
        mask_class = Image.fromarray(mask_class.astype(np.uint8)).resize(img_size, Image.NEAREST)
        mask_class = np.array(mask_class)

    # Random horizontal flip
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask_class = np.fliplr(mask_class)
    # Random vertical flip
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        mask_class = np.flipud(mask_class)
    # Random rotation (0, 90, 180, 270 or small angle)
    angle = random.choice([0, 90, 180, 270, random.uniform(-20, 20)])
    if angle != 0:
        image = image.rotate(angle, resample=Image.BILINEAR)
        mask_class = Image.fromarray(mask_class.astype(np.uint8)).rotate(angle, resample=Image.NEAREST)
        mask_class = np.array(mask_class)

    mask_class = mask_class.copy()
    return image, mask_class
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

# U-DIADS-Bib color to class index mapping
# U-DIADS-Bib color to class index mapping
COLOR_MAP = {
    (0, 0, 0): 0,         # Background
    (255, 255, 0): 1,     # Paratext (Yellow)
    (0, 255, 255): 2,     # Decoration (Cyan)
    (255, 0, 255): 3,     # Main Text (Magenta)
    (255, 0, 0): 4,       # Title (Red)
    (0, 255, 0): 5,       # Chapter Headings (Lime)
}

def rgb_to_class(mask):
    mask_class = np.zeros(mask.shape[:2], dtype=np.int64)
    for rgb, cls in COLOR_MAP.items():
        matches = np.all(mask == rgb, axis=-1)
        mask_class[matches] = cls
    return mask_class


import glob

class UDiadsBibDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, patch_size=448, stride=224, use_patched_data=False):
        self.use_patched_data = use_patched_data
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size if not use_patched_data and split == 'training' else None
        self.stride = stride if not use_patched_data and split == 'training' else None
        
        # Set up the transform
        if transform is None:
            if split == 'training':
                # Use high-res patch size for training
                self.transform = lambda img, mask: default_transform(img, mask, img_size=(patch_size, patch_size))
            else:
                # For val/test, do not resize or augment, just return as is (for sliding window inference)
                self.transform = lambda img, mask: (img, mask)
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
        
        # Pattern for patched dataset structure
        # U-DIADS-Bib-MS_patched/{manuscript}/Image/{split}/image_name_{patch_id}.png
        # U-DIADS-Bib-MS_patched/{manuscript}/mask/{split}_labels/image_name_{patch_id}_zones_NA.png
        
        # We'll just use the manuscript specified in the root_dir
        manuscript = os.path.basename(self.root_dir)
        base_dir = os.path.dirname(self.root_dir)
        
        img_dir = f'{base_dir}/{manuscript}/Image/{self.split}'
        mask_dir = f'{base_dir}/{manuscript}/mask/{self.split}_labels'
        
        print(f"Looking for images in: {img_dir}")
        print(f"Looking for masks in: {mask_dir}")
        
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"Warning: One of the directories does not exist!")
            return [], []
            
        # Get all patch images
        patch_imgs = sorted(glob.glob(os.path.join(img_dir, '*.png')))
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
            mask_class = rgb_to_class(np.array(mask_patch))
            image_patch, mask_class = self.transform(image_patch, mask_class)
            image_patch = TF.to_tensor(image_patch)
            case_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_x{x}_y{y}"
            return {"image": image_patch, "label": torch.from_numpy(mask_class).long(), "case_name": case_name}
        elif self.use_patched_data:
            # Pre-generated patches mode
            img_path = self.img_paths[idx]
            mask_path = self.mask_paths[idx]
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")
            mask_class = rgb_to_class(np.array(mask))
            # For pre-generated patches, apply transforms but don't resize (they're already patch-sized)
            image, mask_class = self.transform(image, mask_class)
            image = TF.to_tensor(image)
            # Just use the filename (without extension) as the case name
            filename = os.path.basename(img_path)
            case_name = os.path.splitext(filename)[0]
            return {"image": image, "label": torch.from_numpy(mask_class).long(), "case_name": case_name}
        else:
            # Whole-image mode (for inference)
            img_path = self.img_paths[idx]
            mask_path = self.mask_paths[idx]
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("RGB")
            mask_class = rgb_to_class(np.array(mask))
            # Resize to original image size (2016x1344) for model input (W, H)
            target_size = (2016, 1344)
            image = image.resize(target_size, Image.BILINEAR)
            mask_class = Image.fromarray(mask_class.astype(np.uint8)).resize(target_size, Image.NEAREST)
            mask_class = np.array(mask_class)
            image, mask_class = self.transform(image, mask_class)
            image = TF.to_tensor(image)
            case_name = os.path.splitext(os.path.basename(img_path))[0]
            return {"image": image, "label": torch.from_numpy(mask_class).long(), "case_name": case_name}

"""
SSTRANS-specific data augmentation and preprocessing transforms.

This module provides heavy data augmentation specifically for the SSTRANS model
when used with DIVAHISDB and UDIADS_BIB datasets, matching the original 
Smart-Swin-Transformer pipeline.
"""

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as tvtf
from scipy.ndimage import map_coordinates


class SSTRANSTransform:
    """
    Heavy data augmentation pipeline for SSTRANS model training.
    
    This transform applies comprehensive augmentation including:
    - Random horizontal and vertical flips
    - Random affine transformations (scaling, rotation, shear, translation)
    - Random additive Gaussian noise
    - Random Gaussian blur
    - Random linear contrast adjustment
    - Proper normalization with Normalize([0.5], [0.5])
    """
    
    def __init__(self, patch_size=224, is_training=True):
        """
        Initialize SSTRANS transform.
        
        Args:
            patch_size (int): Target patch size for resizing
            is_training (bool): Whether to apply augmentations (True for training, False for validation/test)
        """
        self.patch_size = patch_size
        self.is_training = is_training
        
    def __call__(self, image, mask):
        """
        Apply transforms to image and mask.
        
        Args:
            image (PIL.Image): Input image
            mask (numpy.ndarray): Input mask as numpy array
            
        Returns:
            tuple: (transformed_image_tensor, transformed_mask_tensor)
        """
        # Convert mask to PIL Image for easier manipulation
        if isinstance(mask, np.ndarray):
            mask_pil = Image.fromarray(mask.astype(np.uint8))
        else:
            mask_pil = mask
            
        # Resize to target patch size
        image = image.resize((self.patch_size, self.patch_size), Image.BILINEAR)
        mask_pil = mask_pil.resize((self.patch_size, self.patch_size), Image.NEAREST)
        
        if self.is_training:
            # Apply heavy augmentation during training
            image, mask_pil = self._apply_augmentations(image, mask_pil)
        
        # Convert to tensor
        image_tensor = TF.to_tensor(image)
        
        # Apply normalization: Normalize([0.5], [0.5]) for each channel
        # This normalizes to [-1, 1] range
        image_tensor = TF.normalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        # Convert mask back to numpy and then to tensor
        mask_np = np.array(mask_pil)
        mask_tensor = torch.from_numpy(mask_np).long()
        
        return image_tensor, mask_tensor
    
    def _apply_augmentations(self, image, mask):
        """
        Apply heavy data augmentations to image and mask.
        
        Args:
            image (PIL.Image): Input image
            mask (PIL.Image): Input mask
            
        Returns:
            tuple: (augmented_image, augmented_mask)
        """
        # Random horizontal flip (p=0.6) - increased for more diversity
        if random.random() < 0.6:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flip (p=0.4) - increased
        if random.random() < 0.4:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            
        # Random brightness and contrast adjustment (NEW)
        if random.random() < 0.5:
            # Brightness adjustment
            brightness_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
            
            # Contrast adjustment  
            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast_factor)
        
        # Random 90-degree rotations (multiples of 90°) - MODERATE
        if random.random() < 0.2:  # 20% chance (was 15%)
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        
        # Random rotation by multiples of 90° (alternative implementation) - REDUCED
        if random.random() < 0.1:  # 10% chance (was 20%)
            k = random.randint(0, 3)  # 0, 1, 2, 3 rotations
            image = TF.rotate(image, k * 90, fill=0)
            mask = TF.rotate(mask, k * 90, fill=0)
        
        # Random flip along random axis - REDUCED
        if random.random() < 0.1:  # 10% chance (was 20%)
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            else:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
        
        # Random rotation (-15° to 15°) - increased range and probability
        if random.random() < 0.5:  # 50% chance, increased range
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        
        # Random affine transformations (CONSERVATIVE parameters) - REDUCED
        if random.random() < 0.4:  # 40% chance (was 70%)
            # Random rotation (-15° to 15°) - REDUCED
            angle = random.uniform(-15, 15)
            
            # Random scaling (0.8 to 1.2) - REDUCED
            scale = random.uniform(0.8, 1.2)
            
            # Random shear (-8° to 8°) - REDUCED
            shear = random.uniform(-8, 8)
            
            # Random translation (-10% to 10% of image size) - REDUCED
            translate_x = random.uniform(-0.1, 0.1) * self.patch_size
            translate_y = random.uniform(-0.1, 0.1) * self.patch_size
            translate = (translate_x, translate_y)
            
            # Apply affine transformation
            image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=shear, fill=0)
            mask = TF.affine(mask, angle=angle, translate=translate, scale=scale, shear=shear, fill=0)
        
        # PiecewiseAffine transformation - DISABLED (too aggressive)
        # if random.random() < 0.3:  # 30% chance
        #     image, mask = self._piecewise_affine(image, mask)
        
        # Random additive Gaussian noise - REDUCED
        if random.random() < 0.15:  # 15% chance (was 30%)
            image = self._add_gaussian_noise(image)
        
        # Random Gaussian blur (corrected sigma range) - REDUCED
        if random.random() < 0.1:  # 10% chance (was 20%)
            sigma = random.uniform(0, 1.5)  # Reduced sigma range
            radius = sigma * 2.5  # Convert sigma to radius
            image = image.filter(ImageFilter.GaussianBlur(radius=max(0.1, radius)))
        
        # Random linear contrast adjustment (corrected range) - REDUCED
        if random.random() < 0.2:  # 20% chance (was 40%)
            contrast_factor = random.uniform(0.8, 1.2)  # More conservative range
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
        
        return image, mask
    
    def _add_gaussian_noise(self, image):
        """
        Add random Gaussian noise to image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Image with added noise
        """
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Generate noise with correct scale (0.005 × 255 = 1.275)
        noise_scale = 0.005 * 255  # Recommended scale
        noise = np.random.normal(0, noise_scale, img_array.shape)
        
        # Add noise
        noisy_img = img_array + noise
        
        # Clip values to valid range [0, 255]
        noisy_img = np.clip(noisy_img, 0, 255)
        
        # Convert back to PIL Image
        return Image.fromarray(noisy_img.astype(np.uint8))
    
    def _piecewise_affine(self, image, mask):
        """
        Apply PiecewiseAffine transformation to image and mask.
        
        Args:
            image (PIL.Image): Input image
            mask (PIL.Image): Input mask
            
        Returns:
            tuple: (transformed_image, transformed_mask)
        """
        # Convert to numpy arrays
        img_array = np.array(image)
        mask_array = np.array(mask)
        
        # Get image dimensions
        h, w = img_array.shape[:2]
        
        # Create displacement field
        scale = random.uniform(0.008, 0.03)  # Recommended scale range
        displacement = np.random.uniform(-scale, scale, (h, w, 2))
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.array([y_coords, x_coords])
        
        # Add displacement
        new_coords = coords + displacement.transpose(2, 0, 1)
        
        # Apply transformation to image
        if len(img_array.shape) == 3:  # Color image
            transformed_img = np.zeros_like(img_array)
            for c in range(img_array.shape[2]):
                transformed_img[:, :, c] = map_coordinates(img_array[:, :, c], new_coords, order=1, mode='nearest')
        else:  # Grayscale image
            transformed_img = map_coordinates(img_array, new_coords, order=1, mode='nearest')
        
        # Apply transformation to mask (use nearest neighbor for masks)
        transformed_mask = map_coordinates(mask_array, new_coords, order=0, mode='nearest')
        
        # Convert back to PIL Images
        if len(img_array.shape) == 3:
            transformed_image = Image.fromarray(transformed_img.astype(np.uint8))
        else:
            transformed_image = Image.fromarray(transformed_img.astype(np.uint8))
        
        transformed_mask_pil = Image.fromarray(transformed_mask.astype(np.uint8))
        
        return transformed_image, transformed_mask_pil


def sstrans_training_transform(patch_size=224):
    """
    Create SSTRANS training transform with heavy augmentation.
    
    Args:
        patch_size (int): Target patch size
        
    Returns:
        SSTRANSTransform: Training transform with augmentation
    """
    return SSTRANSTransform(patch_size=patch_size, is_training=True)


def sstrans_validation_transform(patch_size=224):
    """
    Create SSTRANS validation/test transform without augmentation.
    
    Args:
        patch_size (int): Target patch size
        
    Returns:
        SSTRANSTransform: Validation transform without augmentation
    """
    return SSTRANSTransform(patch_size=patch_size, is_training=False)


def sstrans_identity_transform(image, mask):
    """
    Identity transform for compatibility with existing code.
    
    Args:
        image (PIL.Image): Input image
        mask (numpy.ndarray): Input mask
        
    Returns:
        tuple: (image_tensor, mask_tensor)
    """
    # Convert mask to PIL Image
    if isinstance(mask, np.ndarray):
        mask_pil = Image.fromarray(mask.astype(np.uint8))
    else:
        mask_pil = mask
    
    # Convert to tensor
    image_tensor = TF.to_tensor(image)
    
    # Apply normalization: Normalize([0.5], [0.5]) for each channel
    image_tensor = TF.normalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    # Convert mask to tensor
    mask_np = np.array(mask_pil)
    mask_tensor = torch.from_numpy(mask_np).long()
    
    return image_tensor, mask_tensor

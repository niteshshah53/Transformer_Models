#!/usr/bin/env python3
"""
Generate 192x192 patches from U-DIADS-Bib-MS dataset.
Creates patches in the same structure as U-DIADS-Bib-MS_patched but saves to U-DIADS-Bib-MS_patched_192.
"""

import glob
import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

# Configuration
target_size = 96  # Stride for overlapping patches (50% overlap)
interval = 192  # Patch size (192x192) - interval = target_size * 2 for 50% overlap
source_root = '/home/hpc/iwi5/iwi5250h/Transformer_Models/U-DIADS-Bib-MS'
output_root = '/home/hpc/iwi5/iwi5250h/Transformer_Models/U-DIADS-Bib-MS_patched_192'

# Manuscripts to process
manuscripts = ['Latin2', 'Latin14396', 'Latin16746', 'Syr341']
splits = ['training', 'validation', 'test']

# Transform to tensor
totensor = transforms.Compose([
    transforms.ToTensor(),
])

def create_patches_for_split(manuscript, split):
    """Create patches for a specific manuscript and split."""
    print(f"\nProcessing {manuscript}/{split}...")
    
    # Input paths (U-DIADS-Bib-MS structure: manuscript/img-manuscript/split)
    img_dir = os.path.join(source_root, manuscript, f'img-{manuscript}', split)
    mask_dir = os.path.join(source_root, manuscript, f'pixel-level-gt-{manuscript}', split)
    
    # Output paths (U-DIADS-Bib-MS_patched structure: manuscript/Image/split and manuscript/mask/split_labels)
    output_img_dir = os.path.join(output_root, manuscript, 'Image', split)
    output_mask_dir = os.path.join(output_root, manuscript, 'mask', f'{split}_labels')
    
    # Create output directories
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # Get all images and masks
    image_files = sorted(glob.glob(os.path.join(img_dir, '*')))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*')))
    
    if len(image_files) != len(mask_files):
        print(f"  WARNING: Mismatch in number of images ({len(image_files)}) and masks ({len(mask_files)})")
        return
    
    print(f"  Found {len(image_files)} image-mask pairs")
    
    total_patches = 0
    
    # Process each image-mask pair
    for img_idx, img_path in enumerate(image_files):
        # Get corresponding mask by matching filename
        # U-DIADS-Bib-MS: mask filename is image filename with .jpg/.JPG replaced by .png
        img_basename = os.path.basename(img_path)
        img_basename_no_ext = os.path.splitext(img_basename)[0]
        
        # Try to find matching mask (mask might be .png even if image is .jpg)
        mask_path = None
        for mask_file in mask_files:
            mask_basename = os.path.basename(mask_file)
            mask_basename_no_ext = os.path.splitext(mask_basename)[0]
            # Match by base filename (handles .jpg/.png differences)
            if img_basename_no_ext == mask_basename_no_ext:
                mask_path = mask_file
                break
        
        if mask_path is None:
            print(f"  WARNING: No mask found for {img_basename}")
            continue
        
        # Load image and mask
        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('RGB')
        except Exception as e:
            print(f"  ERROR loading {os.path.basename(img_path)}: {e}")
            continue
        
        # Convert to tensor
        img_tensor = totensor(image)
        mask_tensor = totensor(mask)
        
        # Get dimensions
        _, h, w = img_tensor.shape
        
        # Generate patches (img_basename_no_ext already extracted above)
        patch_id = 0
        for i in range(0, h + 1, target_size):
            for j in range(0, w + 1, target_size):
                # Check if we can extract a full patch (using interval as patch size)
                if i + interval <= h and j + interval <= w:
                    # Crop image patch
                    img_patch = img_tensor[:, i:i + interval, j:j + interval]
                    
                    # Crop mask patch
                    mask_patch = mask_tensor[:, i:i + interval, j:j + interval]
                    
                    # Save image patch
                    img_patch_name = f'{img_basename_no_ext}_{patch_id:06d}.png'
                    save_image(img_patch, os.path.join(output_img_dir, img_patch_name))
                    
                    # Save mask patch (convert back to PIL for proper saving)
                    mask_patch_pil = transforms.ToPILImage()(mask_patch)
                    # U-DIADS-Bib-MS uses same base name as image with _zones_NA suffix
                    mask_patch_name = f'{img_basename_no_ext}_{patch_id:06d}_zones_NA.png'
                    mask_patch_pil.save(os.path.join(output_mask_dir, mask_patch_name))
                    
                    patch_id += 1
                    total_patches += 1
        
        if (img_idx + 1) % 10 == 0:
            print(f"  Processed {img_idx + 1}/{len(image_files)} images...")
    
    print(f"  ✓ Completed {manuscript}/{split}: {total_patches} patches created")


def main():
    """Main function to generate all patches."""
    print("="*80)
    print("Generating 192x192 patches from U-DIADS-Bib-MS")
    print("="*80)
    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"Patch size: {interval}x{interval}")
    print(f"Stride: {target_size} (50% overlap)")
    print(f"Manuscripts: {', '.join(manuscripts)}")
    print("="*80)
    
    # Create root output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Process each manuscript and split
    for manuscript in manuscripts:
        for split in splits:
            create_patches_for_split(manuscript, split)
    
    print("\n" + "="*80)
    print("All patches generated successfully!")
    print("="*80)


if __name__ == '__main__':
    main()


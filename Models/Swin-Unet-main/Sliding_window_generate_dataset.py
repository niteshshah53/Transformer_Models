import glob
import os
import PIL
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Change to 224 for Swin-Unet as required by Mr. Wu's instructions
target_size = 224
# Use no overlap for extraction as specified
interval = target_size

# List of all manuscripts in the dataset
manuscripts = ['Latin2', 'Latin14396', 'Latin16746', 'Syr341']
# List of data splits to process - consider removing 'test' if you want to use full images for testing
data_splits = ['training', 'validation', 'test']

# Function to process a single manuscript and data split
def process_manuscript_split(manuscript_name, split_name):
    print(f"Processing manuscript: {manuscript_name}, split: {split_name}")
    
    # Set paths for current manuscript and split based on actual directory structure
    ip = f'U-DIADS-Bib-MS/{manuscript_name}/img-{manuscript_name}/{split_name}'
    mp = f'U-DIADS-Bib-MS/{manuscript_name}/pixel-level-gt-{manuscript_name}/{split_name}'
    
    # Skip if input directory doesn't exist
    if not os.path.exists(ip) or not os.path.exists(mp):
        print(f"Skipping {manuscript_name}/{split_name} - directories not found")
        return
    
    # Create output directories with new structure
    save_path_context = f'U-DIADS-Bib-MS_patched/{manuscript_name}/Image/{split_name}'
    save_path_context_mask = f'U-DIADS-Bib-MS_patched/{manuscript_name}/mask/{split_name}_labels'
    
    if not os.path.exists(save_path_context):
        os.makedirs(save_path_context)
    if not os.path.exists(save_path_context_mask):
        os.makedirs(save_path_context_mask)
    
    # Get all images and masks
    images = sorted(glob.glob(os.path.join(ip, '*')))
    masks = sorted(glob.glob(os.path.join(mp, '*')))
    
    if len(images) == 0 or len(masks) == 0:
        print(f"No images or masks found for {manuscript_name}/{split_name}")
        return
    
    print(f"Found {len(images)} images and {len(masks)} masks")
    
    # Process images
    totensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Process all images
    for i in range(len(images)):
        image = Image.open(images[i])
        suffix = images[i].split('/')[-1].split('.')[0]
        toimage = totensor(image)
        _, h, w = toimage.shape
        id = 0
        for y in range(0, h + 1, target_size):
            for x in range(0, w + 1, target_size):
                if y + interval <= h and x + interval <= w:
                    crop = toimage[:, y:y + interval, x:x + interval]
                    save_image(crop, os.path.join(save_path_context, suffix + '_{:06d}.png'.format(id)))
                    id += 1
        print(f"Processed image {i+1}/{len(images)}: {suffix} - created {id} patches")
    
    # Process all masks
    for i in range(len(masks)):
        mask = Image.open(masks[i])
        suffix = masks[i].split('/')[-1].split('.')[0]
        toimage = totensor(mask)
        _, h, w = toimage.shape
        id = 0
        for y in range(0, h + 1, target_size):
            for x in range(0, w + 1, target_size):
                if y + interval <= h and x + interval <= w:
                    crop = toimage[:, y:y + interval, x:x + interval]
                    save_image(crop, os.path.join(save_path_context_mask, suffix + '_{:06d}'.format(id) + '_zones_NA.png'))
                    id += 1
        print(f"Processed mask {i+1}/{len(masks)}: {suffix} - created {id} patches")

# Main execution - run for all manuscripts and splits
if __name__ == "__main__":
    # Create the root output directory
    if not os.path.exists('U-DIADS-Bib-MS_patched'):
        os.makedirs('U-DIADS-Bib-MS_patched')
    
    # First, let's detect what directory structure we have
    print("Analyzing available manuscripts and data...")
    
    # Check manuscript directories
    available_manuscripts = []
    for manuscript_name in manuscripts:
        if os.path.exists(f'U-DIADS-Bib-MS/{manuscript_name}'):
            available_manuscripts.append(manuscript_name)
            print(f"Found manuscript directory: {manuscript_name}")
    
    if not available_manuscripts:
        print("ERROR: No manuscript directories found!")
        exit(1)
    else:
        manuscripts = available_manuscripts
    
    # Process each manuscript and each data split
    for manuscript_name in manuscripts:
        # Create manuscript directory structure
        if not os.path.exists(f'U-DIADS-Bib-MS_patched/{manuscript_name}'):
            os.makedirs(f'U-DIADS-Bib-MS_patched/{manuscript_name}')
        
        # Create Image and mask folders for each manuscript
        if not os.path.exists(f'U-DIADS-Bib-MS_patched/{manuscript_name}/Image'):
            os.makedirs(f'U-DIADS-Bib-MS_patched/{manuscript_name}/Image')
        if not os.path.exists(f'U-DIADS-Bib-MS_patched/{manuscript_name}/mask'):
            os.makedirs(f'U-DIADS-Bib-MS_patched/{manuscript_name}/mask')
        
        # Process each data split
        for split_name in data_splits:
            process_manuscript_split(manuscript_name, split_name)
    
    print("All manuscripts and splits processed successfully!")


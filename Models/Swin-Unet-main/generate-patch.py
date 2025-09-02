import os
import glob
from PIL import Image
import argparse


# Defaults aligned with training config (224x224 non-overlapping patches)
DEFAULT_PATCH_SIZE = 224
DEFAULT_STRIDE = 224
DEFAULT_SPLITS = ["training", "validation", "test"]
DEFAULT_UDIADS_MANUSCRIPTS = ["Latin2", "Latin14396", "Latin16746", "Syr341"]


def _save_image(pil_img, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pil_img.save(out_path)


def _extract_image_patches(img_path, out_dir, basename, patch_size, stride):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    idx = 0
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            crop = img.crop((x, y, x + patch_size, y + patch_size))
            out_path = os.path.join(out_dir, f"{basename}_{idx:06d}.png")
            _save_image(crop, out_path)
            idx += 1
    return idx


def _extract_mask_patches(mask_path, out_dir, basename, patch_size, stride):
    mask = Image.open(mask_path).convert('RGB')
    w, h = mask.size
    idx = 0
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            crop = mask.crop((x, y, x + patch_size, y + patch_size))
            out_path = os.path.join(out_dir, f"{basename}_{idx:06d}_zones_NA.png")
            _save_image(crop, out_path)
            idx += 1
    return idx


def process_divahisdb(diva_root, manuscripts=None, splits=None, patch_size=DEFAULT_PATCH_SIZE, stride=DEFAULT_STRIDE):
    if splits is None:
        splits = DEFAULT_SPLITS
    img_root = os.path.join(diva_root, 'img')
    gt_root = os.path.join(diva_root, 'pixel-level-gt')
    if not os.path.exists(img_root) or not os.path.exists(gt_root):
        raise FileNotFoundError(f"DIVAHISDB layout not found under {diva_root}")

    if manuscripts is None:
        manuscripts = sorted([d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))])

    patched_root = diva_root + '_patched'
    os.makedirs(patched_root, exist_ok=True)

    for m in manuscripts:
        print(f"DIVAHISDB: processing manuscript {m}")
        for s in splits:
            img_dir = os.path.join(img_root, m, s)
            mask_dir = os.path.join(gt_root, m, s)
            out_img = os.path.join(patched_root, m, 'Image', s)
            out_mask = os.path.join(patched_root, m, 'mask', f"{s}_labels")
            if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
                print(f"  Skipping {m}/{s}: input dirs not found")
                continue
            imgs = sorted(glob.glob(os.path.join(img_dir, '*')))
            masks = sorted(glob.glob(os.path.join(mask_dir, '*')))
            if len(imgs) == 0 or len(masks) == 0:
                print(f"  No files found for {m}/{s}")
                continue
            total_img = 0
            for p in imgs:
                b = os.path.splitext(os.path.basename(p))[0]
                total_img += _extract_image_patches(p, out_img, b, patch_size, stride)
            print(f"  {m}/{s}: created {total_img} image patches")

            total_mask = 0
            for p in masks:
                b = os.path.splitext(os.path.basename(p))[0]
                total_mask += _extract_mask_patches(p, out_mask, b, patch_size, stride)
            print(f"  {m}/{s}: created {total_mask} mask patches")

    print("DIVAHISDB processing finished")


def process_udiadsbib(udiads_root, manuscripts=None, splits=None, patch_size=DEFAULT_PATCH_SIZE, stride=DEFAULT_STRIDE):
    if splits is None:
        splits = DEFAULT_SPLITS
    if manuscripts is None:
        manuscripts = DEFAULT_UDIADS_MANUSCRIPTS

    patched_root = udiads_root + '_patched'
    os.makedirs(patched_root, exist_ok=True)

    for m in manuscripts:
        print(f"U-DIADS-Bib: processing manuscript {m}")
        img_root = os.path.join(udiads_root, m, f'img-{m}')
        gt_root = os.path.join(udiads_root, m, f'pixel-level-gt-{m}')
        for s in splits:
            img_dir = os.path.join(img_root, s)
            mask_dir = os.path.join(gt_root, s)
            out_img = os.path.join(patched_root, m, 'Image', s)
            out_mask = os.path.join(patched_root, m, 'mask', f"{s}_labels")
            if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
                print(f"  Skipping {m}/{s}: input dirs not found")
                continue
            imgs = sorted(glob.glob(os.path.join(img_dir, '*')))
            masks = sorted(glob.glob(os.path.join(mask_dir, '*')))
            if len(imgs) == 0 or len(masks) == 0:
                print(f"  No files found for {m}/{s}")
                continue
            total_img = 0
            for p in imgs:
                b = os.path.splitext(os.path.basename(p))[0]
                total_img += _extract_image_patches(p, out_img, b, patch_size, stride)
            print(f"  {m}/{s}: created {total_img} image patches")

            total_mask = 0
            for p in masks:
                b = os.path.splitext(os.path.basename(p))[0]
                total_mask += _extract_mask_patches(p, out_mask, b, patch_size, stride)
            print(f"  {m}/{s}: created {total_mask} mask patches")

    print("U-DIADS-Bib processing finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 224x224 patches for DIVAHISDB or U-DIADS-Bib')
    parser.add_argument('--dataset', choices=['DIVAHISDB', 'UDIADS_BIB', 'auto'], default='auto', help="Which dataset to process. 'auto' detects layout.")
    parser.add_argument('--diva_root', default='DIVAHISDB', help='Root folder for DIVAHISDB')
    parser.add_argument('--udiads_root', default='U-DIADS-Bib-MS', help='Root folder for U-DIADS-Bib')
    parser.add_argument('--manuscripts', default=None, help='Comma-separated manuscript names to process (optional)')
    parser.add_argument('--splits', default=None, help='Comma-separated splits to process (optional)')
    parser.add_argument('--patch_size', type=int, default=DEFAULT_PATCH_SIZE, help='Patch size (default 224)')
    parser.add_argument('--stride', type=int, default=DEFAULT_STRIDE, help='Stride (default 224)')
    args = parser.parse_args()

    manuscripts = args.manuscripts.split(',') if args.manuscripts else None
    splits = args.splits.split(',') if args.splits else None

    dataset = args.dataset
    if dataset == 'auto':
        if os.path.exists(os.path.join(args.diva_root, 'img')):
            dataset = 'DIVAHISDB'
        elif os.path.exists(args.udiads_root):
            dataset = 'UDIADS_BIB'
        else:
            raise FileNotFoundError('Neither DIVAHISDB nor U-DIADS-Bib detected; specify --dataset')

    if dataset == 'DIVAHISDB':
        process_divahisdb(args.diva_root, manuscripts=manuscripts, splits=splits, patch_size=args.patch_size, stride=args.stride)
    else:
        process_udiadsbib(args.udiads_root, manuscripts=manuscripts, splits=splits, patch_size=args.patch_size, stride=args.stride)
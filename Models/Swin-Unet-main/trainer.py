
import numpy as np
import logging
import os
import sys
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from utils import DiceLoss
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

# Only one top-level worker_init_fn
def worker_init_fn(worker_id):
    import random
    base_seed = getattr(worker_init_fn, 'base_seed', 1234)
    random.seed(base_seed + worker_id)

def trainer_synapse(args, model, snapshot_path, train_dataset=None, val_dataset=None):
    import random
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    # Set up data loaders
    worker_init_fn.base_seed = args.seed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    # Print training information header
    print("\n" + "="*80)
    print(f"TRAINING CONFIGURATION")
    print("-"*80)
    print(f"• Dataset: {args.dataset}")
    print(f"• Model: {args.model}")
    print(f"• Batch Size: {batch_size}")
    print(f"• Max Epochs: {args.max_epochs}")
    print(f"• Learning Rate: {base_lr}")
    print(f"• Number of Classes: {num_classes}")
    print(f"• Output Directory: {snapshot_path}")
    print("="*80 + "\n")

    # Compute class weights for UDiadsBibDataset
    if hasattr(train_dataset, 'img_paths') and hasattr(train_dataset, 'mask_paths') and args.dataset.lower() == 'udiads_bib':
        class_counts = np.zeros(args.num_classes, dtype=np.float64)
        print('Computing class weights...')
        for mask_path in train_dataset.mask_paths:
            from PIL import Image
            mask = Image.open(mask_path).convert('RGB')
            mask = np.array(mask)
            # Use the same mapping as in dataset_udiadsbib.py
            mask_class = np.zeros(mask.shape[:2], dtype=np.int64)
            COLOR_MAP = {
                (0, 0, 0): 0,
                (255, 255, 0): 1,
                (0, 255, 255): 2,
                (255, 0, 255): 3,
                (255, 0, 0): 4,
                (0, 255, 0): 5,
            }
            for rgb, cls in COLOR_MAP.items():
                matches = np.all(mask == rgb, axis=-1)
                class_counts[cls] += matches.sum()
        class_freq = class_counts / class_counts.sum()
        weights = 1.0 / (class_freq + 1e-6)
        weights = weights / weights.sum()  # Normalize
        
        # Print class frequencies and weights in a table format
        print("\n" + "-"*80)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("-"*80)
        print(f"{'Class':<6} {'Frequency':<15} {'Weight':<15}")
        print("-"*80)
        for cls in range(args.num_classes):
            print(f"{cls:<6} {class_freq[cls]:<15.6f} {weights[cls]:<15.6f}")
        print("-"*80 + "\n")
            
        ce_loss = CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).cuda())
    else:
        ce_loss = CrossEntropyLoss()
    from utils import FocalLoss
    dice_loss = DiceLoss(num_classes)
    focal_loss = FocalLoss(gamma=2, weight=torch.tensor(weights, dtype=torch.float32).cuda())
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    writer = SummaryWriter(snapshot_path + '/log')

    best_val_loss = float('inf')
    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            # Support both dict and tuple batch
            if isinstance(batch, dict):
                images = batch['image'].cuda()
                labels = batch['label'].cuda()
            else:
                images = batch[0].cuda()
                labels = batch[1].cuda()
            outputs = model(images)
            loss_ce = ce_loss(outputs, labels)
            loss_focal = focal_loss(outputs, labels)
            # Compute Dice on probabilities for stable behaviour
            loss_dice = dice_loss(outputs, labels, softmax=True)
            # Main loss: Weighted sum (CE, Focal, Dice) - more weight to Focal/Dice
            loss = 0.05 * loss_ce + 0.475 * loss_focal + 0.475 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        writer.add_scalar('train/loss', train_loss, epoch)
        
        # Print epoch summary with consistent formatting
        print("\n" + "="*80)
        print(f"EPOCH {epoch+1}/{args.max_epochs}")
        print("-"*80)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"  • CE Loss: {loss_ce.item():.4f}")
        print(f"  • Focal Loss: {loss_focal.item():.4f}")
        print(f"  • Dice Loss: {loss_dice.item():.4f}")
        
        # Validation with sliding window inference
        model.eval()
        val_loss = 0.0
        patch_size = getattr(model, 'img_size', 224) if hasattr(model, 'img_size') else 224
        stride = patch_size  # or patch_size // 2 for overlap
        from torchvision.transforms import functional as TF
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Load original high-res image and mask from disk
                img_path = val_dataset.img_paths[batch_idx]
                mask_path = val_dataset.mask_paths[batch_idx]
                from PIL import Image
                orig_img_pil = Image.open(img_path).convert("RGB")
                orig_img_np = np.array(orig_img_pil)
                h, w = orig_img_np.shape[:2]
                # Prepare empty prediction and count maps
                pred_full = np.zeros((h, w, args.num_classes), dtype=np.float32)
                count_map = np.zeros((h, w), dtype=np.float32)
                # Sliding window over the image
                for y in range(0, h - patch_size + 1, stride):
                    for x in range(0, w - patch_size + 1, stride):
                        patch = orig_img_np[y:y+patch_size, x:x+patch_size, :]
                        patch_tensor = TF.to_tensor(Image.fromarray(patch)).unsqueeze(0).cuda()
                        output = model(patch_tensor)
                        output_np = output.squeeze(0).cpu().numpy()  # (C, H, W)
                        output_np = np.transpose(output_np, (1, 2, 0))  # (H, W, C)
                        pred_full[y:y+patch_size, x:x+patch_size, :] += output_np
                        count_map[y:y+patch_size, x:x+patch_size] += 1
                # Normalize by count_map
                count_map = np.maximum(count_map, 1)[:, :, None]
                pred_full = pred_full / count_map
                pred_label = np.argmax(pred_full, axis=-1)
                # Load ground truth mask
                gt_pil = Image.open(mask_path).convert("RGB")
                from datasets.dataset_udiadsbib import rgb_to_class
                gt_np = np.array(gt_pil)
                gt_class = rgb_to_class(gt_np)
                # Compute loss on the full image
                pred_tensor = torch.from_numpy(pred_full.transpose(2, 0, 1)).unsqueeze(0).float().cuda()  # (1, C, H, W)
                gt_tensor = torch.from_numpy(gt_class).unsqueeze(0).long().cuda()  # (1, H, W)
                loss_ce = ce_loss(pred_tensor, gt_tensor)
                # pred_tensor contains logits averaged over patches; compute Dice on probabilities
                loss_dice = dice_loss(pred_tensor, gt_tensor, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                val_loss += loss.item()
        val_loss /= len(val_loader)
        writer.add_scalar('val/loss', val_loss, epoch)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            # Always save the latest best checkpoint to a fixed filename
            latest_best_path = os.path.join(snapshot_path, "best_model_latest.pth")
            torch.save(model.state_dict(), latest_best_path)
            best_val_loss = val_loss
            save_path = os.path.join(snapshot_path, f"best_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✓ Best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")
            print("-"*80)

            # Keep only the 3 latest/best checkpoints
            import glob
            import re
            checkpoint_files = glob.glob(os.path.join(snapshot_path, "best_model_epoch*.pth"))
            # Extract epoch number from filename
            def extract_epoch(fname):
                match = re.search(r"best_model_epoch(\\d+)\\.pth", fname)
                return int(match.group(1)) if match else -1
            # Sort by epoch descending (latest first)
            checkpoint_files.sort(key=extract_epoch, reverse=True)
            # Remove files after the 3 most recent
            for ckpt in checkpoint_files[3:]:
                try:
                    os.remove(ckpt)
                except Exception:
                    pass
        else:
            print("-"*80)

        # Step the learning rate scheduler
        scheduler.step()
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

    # Print training summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("-"*80)
    print(f"• Best Validation Loss: {best_val_loss:.4f}")
    print(f"• Model Saved To: {snapshot_path}")
    print("="*80 + "\n")
    
    writer.close()



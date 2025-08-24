from os.path import split
import argparse
import logging
import os
import random
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

from config import get_config
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_udiadsbib import UDiadsBibDataset
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import test_single_volume

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='swinunet', choices=['swinunet', 'missformer'], help='Model to use: swinunet or missformer')

parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/test_vol_h5',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--udiadsbib_root', type=str, default='U-DIADS-Bib-MS', help='Root dir for U-DIADS-Bib dataset')
parser.add_argument('--udiadsbib_split', type=str, default='test', help='Split for U-DIADS-Bib (training/validation/test)')
parser.add_argument('--manuscript', type=str, choices=['Latin2', 'Latin14396', 'Latin16746', 'Syr341'], required=True,
                    help='Manuscript to test (Latin2, Latin14396, Latin16746, or Syr341)')
parser.add_argument('--use_patched_data', action='store_true', help='Use pre-generated patches instead of full images')
parser.add_argument('--dataset', type=str,
                    default='datasets', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=2016, help='input image width for full-res inference (2016 for U-DIADS-Bib)')
# Patch-based arguments
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

parser.add_argument("--n_class", default=4, type=int)
parser.add_argument("--split_name", default="test", help="Directory of the input list")

args = parser.parse_args()

if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
config = get_config(args)


def inference(args, model, test_save_path=None):
    from datasets.dataset_udiadsbib import rgb_to_class
    import torchvision.transforms.functional as TF
    if args.dataset.lower() == "udiads_bib":
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from PIL import Image
        import numpy as np
        # Custom colormap for 6 classes
        class_colors = [
            (0, 0, 0),         # 0: Background (black)
            (255, 255, 0),     # 1: Paratext (yellow)
            (0, 255, 255),     # 2: Decoration (cyan)
            (255, 0, 255),     # 3: Main Text (magenta)
            (255, 0, 0),       # 4: Title (red)
            (0, 255, 0),       # 5: Chapter Headings (lime)
        ]
        cmap = ListedColormap(class_colors)
        
        # For Option 1: Use pre-generated patches
        logging.info("Using pre-generated patches for testing")
        
        # Instead of loading the original dataset, we'll work with the patched data
        patch_size = 224  # Size of patches from Sliding_window_generate_dataset.py
        
        # Initialize metric accumulators
        n_classes = 6
        TP = np.zeros(n_classes, dtype=np.float64)
        FP = np.zeros(n_classes, dtype=np.float64)
        FN = np.zeros(n_classes, dtype=np.float64)
        IoU = np.zeros(n_classes, dtype=np.float64)
        
        # Prepare result directory for PNGs
        result_dir = os.path.join(test_save_path, "result") if test_save_path is not None else None
        if result_dir is not None:
            os.makedirs(result_dir, exist_ok=True)
            
        # Dictionary to group patches by original image
        patch_groups = {}  # key: original_image_name, value: list of patch paths
        patch_positions = {}  # key: patch_path, value: (x, y) position in original image
        # Find all patch images for test set
        manuscript_name = args.manuscript
        patch_dir = f'{args.udiadsbib_root}/{manuscript_name}/Image/test'
        mask_dir = f'{args.udiadsbib_root}/{manuscript_name}/mask/test_labels'
        
        if not os.path.exists(patch_dir) or not os.path.exists(mask_dir):
            logging.info(f"Skipping {manuscript_name} - patch directories not found")
            return
            
        patch_files = sorted(glob.glob(os.path.join(patch_dir, '*.png')))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        
        if len(patch_files) == 0:
            logging.info(f"No patches found for {manuscript_name}")
            return
            
        logging.info(f"Found {len(patch_files)} patches for {manuscript_name}")
        
        # Group patches by original image
        for patch_path in patch_files:
            # Extract original image name and patch position
            filename = os.path.basename(patch_path)
            # Expected format: original_name_XXXXXX.png
            parts = filename.split('_')
            if len(parts) >= 2:
                original_name = '_'.join(parts[:-1])  # Everything before the last underscore
                patch_id = int(parts[-1].split('.')[0])  # The number after the last underscore
                
                # Group by original image
                if original_name not in patch_groups:
                    patch_groups[original_name] = []
                patch_groups[original_name].append(patch_path)
                
                # Store patch ID for position calculation later
                patch_positions[patch_path] = patch_id
        
        # Now process each original image by stitching its patches
        for original_name, patches in patch_groups.items():
            logging.info(f"Processing original image: {original_name} with {len(patches)} patches")
            
            # First, find the original image dimensions to accurately calculate patch positions
            orig_width = 0
            orig_height = 0
            
            # Try to find the original image to get its dimensions
            manuscript_name = args.manuscript
            # Look for .jpg in the original directory (not patched)
            orig_path_jpg = f'U-DIADS-Bib-MS/{manuscript_name}/img-{manuscript_name}/test/{original_name}.jpg'
            if os.path.exists(orig_path_jpg):
                with Image.open(orig_path_jpg) as img:
                    orig_width, orig_height = img.size
            
            if orig_width == 0 or orig_height == 0:
                logging.warning(f"Could not find original image for {original_name}, estimating dimensions from patches")
                # Estimate dimensions from patch positions
                for patch_path in patches:
                    patch_id = patch_positions[patch_path]
                    # We need to determine the number of patches per row in the original extraction
                    # This requires knowing the original image width
                    # As a fallback, estimate based on max patch_id
                    patches_per_row = 10  # Default fallback
                    
                max_patch_id = max([patch_positions[p] for p in patches])
                max_x = ((max_patch_id % patches_per_row) + 1) * patch_size
                max_y = ((max_patch_id // patches_per_row) + 1) * patch_size
            else:
                # Calculate patches per row based on original width
                patches_per_row = orig_width // patch_size
                if patches_per_row == 0:
                    patches_per_row = 1
                # Use original dimensions but ensure they're multiples of patch_size
                max_x = ((orig_width // patch_size) + (1 if orig_width % patch_size else 0)) * patch_size
                max_y = ((orig_height // patch_size) + (1 if orig_height % patch_size else 0)) * patch_size
            
            logging.info(f"Original image dimensions (estimated): {max_x}x{max_y}")
            
            # Create empty prediction map - using int32 to allow for accumulation
            pred_full = np.zeros((max_y, max_x), dtype=np.int32)
            count_map = np.zeros((max_y, max_x), dtype=np.int32)
            
            # Process each patch
            for patch_path in patches:
                patch_id = patch_positions[patch_path]
                
                # Calculate position using the same logic as in Sliding_window_generate_dataset.py
                # In the extraction script, patches are created in row-major order (across then down)
                x = (patch_id % patches_per_row) * patch_size
                y = (patch_id // patches_per_row) * patch_size
                
                # Load patch
                patch = Image.open(patch_path).convert("RGB")
                patch_tensor = TF.to_tensor(patch).unsqueeze(0).cuda()
                
                # Get prediction
                with torch.no_grad():
                    output = model(patch_tensor)
                    pred_patch = torch.argmax(output, dim=1).cpu().numpy()[0]
                
                # Add to prediction map (check boundaries to avoid index errors)
                if y+patch_size <= pred_full.shape[0] and x+patch_size <= pred_full.shape[1]:
                    pred_full[y:y+patch_size, x:x+patch_size] += pred_patch
                    count_map[y:y+patch_size, x:x+patch_size] += 1
                else:
                    # Handle edge cases (partial patches at image boundaries)
                    valid_h = min(patch_size, pred_full.shape[0]-y)
                    valid_w = min(patch_size, pred_full.shape[1]-x)
                    if valid_h > 0 and valid_w > 0:
                        pred_full[y:y+valid_h, x:x+valid_w] += pred_patch[:valid_h, :valid_w]
                        count_map[y:y+valid_h, x:x+valid_w] += 1
            # Normalize by count map
            pred_full = np.round(pred_full / np.maximum(count_map, 1)).astype(np.uint8)
            
            # For evaluation, we need to find the corresponding ground truth
            # We'll look for the original full mask in the original dataset
            original_gt_found = False
            manuscript_name = args.manuscript
            gt_path = f'U-DIADS-Bib-MS/{manuscript_name}/pixel-level-gt-{manuscript_name}/test/{original_name}.png'
            if os.path.exists(gt_path):
                gt_pil = Image.open(gt_path).convert("RGB")
                gt_np = np.array(gt_pil)
                gt_class = rgb_to_class(gt_np)
                original_gt_found = True
            
            if not original_gt_found:
                logging.warning(f"Could not find original ground truth for {original_name}")
                # Create a dummy ground truth (all zeros) for saving comparison images
                gt_class = np.zeros_like(pred_full)
            
            # Save predicted mask as PNG in result directory
            if result_dir is not None:
                # Convert class indices to RGB
                rgb_mask = np.zeros((pred_full.shape[0], pred_full.shape[1], 3), dtype=np.uint8)
                for idx, color in enumerate(class_colors):
                    rgb_mask[pred_full == idx] = color
                pred_png_path = os.path.join(result_dir, f"{original_name}.png")
                Image.fromarray(rgb_mask).save(pred_png_path)
            
            # Save side-by-side comparison image for visualization
            if test_save_path is not None and original_gt_found:
                compare_dir = os.path.join(test_save_path, 'compare')
                os.makedirs(compare_dir, exist_ok=True)
                
                # Resize gt_class if dimensions don't match (could happen due to patch positions estimation)
                if gt_class.shape != pred_full.shape:
                    logging.warning(f"Resizing ground truth for {original_name} from {gt_class.shape} to {pred_full.shape}")
                    gt_class_resized = np.zeros_like(pred_full)
                    min_h = min(gt_class.shape[0], pred_full.shape[0])
                    min_w = min(gt_class.shape[1], pred_full.shape[1])
                    gt_class_resized[:min_h, :min_w] = gt_class[:min_h, :min_w]
                    gt_class = gt_class_resized
                
                # Create visualization
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                
                # Load original image for visualization
                orig_img_path = None
                manuscript_name = args.manuscript
                test_img_path_jpg = f'U-DIADS-Bib-MS/{manuscript_name}/img-{manuscript_name}/test/{original_name}.jpg'
                if os.path.exists(test_img_path_jpg):
                    orig_img_path = test_img_path_jpg
                
                if orig_img_path:
                    orig_img = Image.open(orig_img_path).convert("RGB")
                    orig_img_np = np.array(orig_img)
                    # Resize if dimensions don't match
                    if orig_img_np.shape[:2] != pred_full.shape:
                        orig_img_np_resized = np.zeros((pred_full.shape[0], pred_full.shape[1], 3), dtype=np.uint8)
                        min_h = min(orig_img_np.shape[0], pred_full.shape[0])
                        min_w = min(orig_img_np.shape[1], pred_full.shape[1])
                        orig_img_np_resized[:min_h, :min_w] = orig_img_np[:min_h, :min_w]
                        orig_img_np = orig_img_np_resized
                    axs[0].imshow(orig_img_np)
                else:
                    # Create a blank image if original not found
                    axs[0].imshow(np.zeros((pred_full.shape[0], pred_full.shape[1], 3), dtype=np.uint8))
                
                axs[0].set_title('Original')
                axs[0].axis('off')
                axs[1].imshow(pred_full, cmap=cmap, vmin=0, vmax=5)
                axs[1].set_title('Prediction')
                axs[1].axis('off')
                axs[2].imshow(gt_class, cmap=cmap, vmin=0, vmax=5)
                axs[2].set_title('Ground Truth')
                axs[2].axis('off')
                plt.tight_layout()
                save_img_path = os.path.join(compare_dir, f"{original_name}_compare.png")
                plt.savefig(save_img_path, bbox_inches='tight')
                plt.close(fig)
            # Compute metrics for each class if ground truth is available
            if original_gt_found:
                for cls in range(n_classes):
                    pred_c = (pred_full == cls)
                    gt_c = (gt_class == cls)
                    TP[cls] += np.logical_and(pred_c, gt_c).sum()
                    FP[cls] += np.logical_and(pred_c, np.logical_not(gt_c)).sum()
                    FN[cls] += np.logical_and(np.logical_not(pred_c), gt_c).sum()
                    union = np.logical_or(pred_c, gt_c).sum()
                    IoU[cls] += np.logical_and(pred_c, gt_c).sum() / (union + 1e-8)
            
            logging.info(f"Processed {original_name}")
        
        # Calculate metrics
        manuscript_name = args.manuscript
        num_processed_images = sum(1 for img in patch_groups 
                                if os.path.exists(f'U-DIADS-Bib-MS/{manuscript_name}/pixel-level-gt-{manuscript_name}/test/{img}.png'))
        
        if num_processed_images > 0:
            precision = TP / (TP + FP + 1e-8)
            recall = TP / (TP + FN + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            mean_iou = IoU / num_processed_images
        else:
            precision = np.zeros(n_classes)
            recall = np.zeros(n_classes) 
            f1 = np.zeros(n_classes)
            mean_iou = np.zeros(n_classes)
        class_names = ['Background', 'Paratext', 'Decoration', 'Main Text', 'Title', 'Chapter Headings']
        logging.info("\nPer-class metrics:")
        for cls in range(n_classes):
            logging.info(f"{class_names[cls]}: Precision={precision[cls]:.4f}, Recall={recall[cls]:.4f}, F1={f1[cls]:.4f}, IoU={mean_iou[cls]:.4f}")
        logging.info("\nMean metrics:")
        logging.info(f"Mean Precision: {np.mean(precision):.4f}")
        logging.info(f"Mean Recall: {np.mean(recall):.4f}")
        logging.info(f"Mean F1: {np.mean(f1):.4f}")
        logging.info(f"Mean IoU: {np.mean(mean_iou):.4f}")
        logging.info("Inference on U-DIADS-Bib completed.")
        return "Testing Finished!"
    else:
        db_test = Synapse_dataset(base_dir=args.volume_path, split=args.split_name, list_dir=args.list_dir)
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
        logging.info("{} test iterations per epoch".format(len(testloader)))
        model.eval()
        metric_list = 0.0
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            if args.dataset == "datasets":
                case_name = split(case_name.split(",")[0])[-1]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                          patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
                i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        metric_list = metric_list / len(db_test)
        for i in range(1, args.num_classes):
            logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
        return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset

    if args.dataset.lower() == "udiads_bib":
        args.num_classes = 6
        args.is_pretrain = True
    else:
        dataset_config = {
            args.dataset: {
                'root_path': args.root_path,
                'list_dir': f'./lists/{args.dataset}',
                'num_classes': args.n_class,
                "z_spacing": 1
            },
        }
        args.num_classes = dataset_config[dataset_name]['num_classes']
        args.volume_path = dataset_config[dataset_name]['root_path']
        args.list_dir = dataset_config[dataset_name]['list_dir']
        args.z_spacing = dataset_config[dataset_name]['z_spacing']
        args.is_pretrain = True

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    # Model selection logic (same as train.py)
    def get_model(args, config):
        model_name = args.model.lower()
        if model_name == 'swinunet':
            from networks.vision_transformer import SwinUnet as ViT_seg
            net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
            net.load_from(config)
            return net
        elif model_name == 'missformer':
            from networks.MissFormer.MISSFormer import MISSFormer
            net = MISSFormer(num_classes=args.num_classes)
            net = net.cuda()
            return net
        else:
            print(f"Unknown model: {args.model}. Supported: swinunet, missformer")
            sys.exit(1)

    net = get_model(args, config)


    # Find best model checkpoint for UDIADS_BIB
    if args.dataset.lower() == "udiads_bib":
        # Find the best_model_epoch*.pth file with the highest epoch
        import glob
        ckpts = glob.glob(os.path.join(args.output_dir, 'best_model_epoch*.pth'))
        if not ckpts:
            raise FileNotFoundError("No best_model_epoch*.pth found in output_dir")
        # Use the one with the highest epoch number
        ckpt = sorted(ckpts, key=lambda x: int(x.split('epoch')[-1].split('.')[0]))[-1]
        snapshot = ckpt
    else:
        snapshot = os.path.join(args.output_dir, 'best_model.pth')
        if not os.path.exists(snapshot):
            snapshot = snapshot.replace('best_model', 'epoch_' + str(args.max_epochs - 1))
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet", msg)
    snapshot_name = os.path.basename(snapshot)

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)

# python train.py --dataset Synapse --cfg $CFG --root_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE
# python train.py --output_dir './model_out/datasets' --dataset datasets --img_size 224 --batch_size 32 --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_preprocessed/Dataset001_mm/nnUNetPlans_2d_split
# python test.py --output_dir ./model_out/datasets --dataset datasets --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --root_path /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_preprocessed/Dataset001_mm/test --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24

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
        db_test = UDiadsBibDataset(
            root_dir=args.udiadsbib_root,
            split=args.udiadsbib_split,
            patch_size=None,
            stride=None
        )
        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
        logging.info("{} test iterations per epoch".format(len(testloader)))
        model.eval()
        # Initialize metric accumulators
        n_classes = 6
        TP = np.zeros(n_classes, dtype=np.float64)
        FP = np.zeros(n_classes, dtype=np.float64)
        FN = np.zeros(n_classes, dtype=np.float64)
        IoU = np.zeros(n_classes, dtype=np.float64)
        from PIL import Image
        patch_size = getattr(model, 'img_size', 224) if hasattr(model, 'img_size') else 224
        stride = patch_size  # no overlap, or set to patch_size//2 for 50% overlap
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            # Load original high-res image from disk
            img_path = db_test.img_paths[i_batch]
            orig_img_pil = Image.open(img_path).convert("RGB")
            orig_img_np = np.array(orig_img_pil)
            h, w = orig_img_np.shape[:2]
            # Prepare empty prediction and count maps
            pred_full = np.zeros((h, w), dtype=np.float32)
            count_map = np.zeros((h, w), dtype=np.float32)
            # Sliding window over the image
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    patch = orig_img_np[y:y+patch_size, x:x+patch_size, :]
                    patch_tensor = TF.to_tensor(Image.fromarray(patch)).unsqueeze(0).cuda()
                    with torch.no_grad():
                        output = model(patch_tensor)
                        pred_patch = torch.argmax(output, dim=1).cpu().numpy()[0]
                    pred_full[y:y+patch_size, x:x+patch_size] += pred_patch
                    count_map[y:y+patch_size, x:x+patch_size] += 1
            # Normalize by count_map to handle overlaps
            pred_full = np.round(pred_full / np.maximum(count_map, 1)).astype(np.uint8)
            # Load ground truth mask
            mask_path = db_test.mask_paths[i_batch]
            gt_pil = Image.open(mask_path).convert("RGB")
            gt_np = np.array(gt_pil)
            gt_class = rgb_to_class(gt_np)
            # Save side-by-side comparison image (Original, Prediction, Ground Truth)
            if test_save_path is not None:
                compare_dir = os.path.join(test_save_path, 'compare')
                os.makedirs(compare_dir, exist_ok=True)
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(orig_img_np)
                axs[0].set_title('Original')
                axs[0].axis('off')
                axs[1].imshow(pred_full, cmap=cmap, vmin=0, vmax=5)
                axs[1].set_title('Prediction')
                axs[1].axis('off')
                axs[2].imshow(gt_class, cmap=cmap, vmin=0, vmax=5)
                axs[2].set_title('Ground Truth')
                axs[2].axis('off')
                plt.tight_layout()
                save_img_path = os.path.join(compare_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_compare.png")
                plt.savefig(save_img_path, bbox_inches='tight')
                plt.close(fig)
            # Compute metrics for each class
            for cls in range(n_classes):
                pred_c = (pred_full == cls)
                gt_c = (gt_class == cls)
                TP[cls] += np.logical_and(pred_c, gt_c).sum()
                FP[cls] += np.logical_and(pred_c, np.logical_not(gt_c)).sum()
                FN[cls] += np.logical_and(np.logical_not(pred_c), gt_c).sum()
                union = np.logical_or(pred_c, gt_c).sum()
                IoU[cls] += np.logical_and(pred_c, gt_c).sum() / (union + 1e-8)
            logging.info(f"Tested {os.path.splitext(os.path.basename(img_path))[0]}")
        # Calculate metrics
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        mean_iou = IoU / len(testloader)
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

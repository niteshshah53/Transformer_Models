#!/bin/bash -l
#SBATCH --job-name=swinunet_syr341
#SBATCH --output=All_Results_with_No_FocalLoss/swinunet/UDIADS_BIB_MS/Syr341/train_test_syr341_%j.out
#SBATCH --error=All_Results_with_No_FocalLoss/swinunet/UDIADS_BIB_MS/Syr341/train_test_syr341_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# Load modules
module purge
module load python/pytorch2.6py3.12
module load cuda/11.8
module load cudnn

# Create logs directory 
mkdir -p ../../logs

conda activate pytorch2.6-py3.12

echo "=== Training Syr341 ==="

python3 train.py \
    --model swinunet \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Syr341 \
    --cfg "../../common/configs/swin_tiny_patch4_window7_224_lite.yaml" \
    --num_classes 5 \
    --base_lr 0.0002 \
    --batch_size 32 \
    --max_epochs 300 \
    --patience 30 \
    --use_patched_data \
    --output_dir "All_Results_with_No_FocalLoss/swinunet/UDIADS_BIB_MS/Syr341"

echo "=== Training Completed ==="

echo "=== Testing Syr341 ==="

python3 test.py \
    --model swinunet \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Syr341 \
    --cfg "../../common/configs/swin_tiny_patch4_window7_224_lite.yaml" \
    --use_patched_data \
    --is_savenii \
    --output_dir "All_Results_with_No_FocalLoss/swinunet/UDIADS_BIB_MS/Syr341"

echo "=== Testing Completed ==="

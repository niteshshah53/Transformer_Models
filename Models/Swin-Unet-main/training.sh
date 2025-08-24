#!/bin/bash -l
#SBATCH --job-name=das_train_test
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# Load modules
module purge
module load python/pytorch2.6py3.12
module load cuda/11.8
module load cudnn

# Create logs directory
mkdir -p logs

conda activate pytorch2.6-py3.12

# --- Run training with SwinUnet ---
python3 train.py \
    --model swinunet \
    --dataset UDIADS_BIB \
    --udiadsbib_root "U-DIADS-Bib-MS_patched/Latin14396" \
    --udiadsbib_split training \
    --img_size 224 \
    --num_classes 6 \
    --output_dir ./model_out/udiadsbib_patch224_swinunet_Latin14396 \
    --max_epochs 300 \
    --batch_size 32 \
    --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
    --use_patched_data



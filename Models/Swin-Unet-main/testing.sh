#!/bin/bash -l
#SBATCH --job-name=das_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
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

# --- Run testing with SwinUnet for Latin2 ---
python3 test.py \
    --model swinunet \
    --dataset UDIADS_BIB \
    --udiadsbib_root "U-DIADS-Bib-MS_patched" \
    --udiadsbib_split test \
    --manuscript "Latin14396" \
    --img_size 224 \
    --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
    --num_classes 6 \
    --output_dir "./model_out/udiadsbib_patch224_swinunet_Latin14396" \
    --is_savenii \
    --use_postprocessing \
    --use_patched_data 

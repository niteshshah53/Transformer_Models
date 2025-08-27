#!/bin/bash -l
#SBATCH --job-name=das_train_test_all
#SBATCH --output=logs/train_test_all_%j.out
#SBATCH --error=logs/train_test_all_%j.out
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

MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341)

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo "=== Training $MANUSCRIPT ==="
    python3 train.py \
        --model swinunet \
        --dataset UDIADS_BIB \
        --udiadsbib_root "U-DIADS-Bib-MS_patched/${MANUSCRIPT}" \
        --udiadsbib_split training \
        --img_size 224 \
        --num_classes 6 \
        --output_dir ./model_out/udiadsbib_patch224_swinunet_${MANUSCRIPT} \
        --max_epochs 300 \
        --batch_size 32 \
        --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
        --use_patched_data

    echo "=== Testing $MANUSCRIPT ==="
    python3 test.py \
        --model swinunet \
        --dataset UDIADS_BIB \
        --udiadsbib_root "U-DIADS-Bib-MS_patched" \
        --udiadsbib_split test \
        --manuscript "${MANUSCRIPT}" \
        --img_size 224 \
        --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
        --num_classes 6 \
        --output_dir "./model_out/udiadsbib_patch224_swinunet_${MANUSCRIPT}" \
        --is_savenii \
        --use_patched_data

done

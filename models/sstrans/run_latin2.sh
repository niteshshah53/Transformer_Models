#!/bin/bash -l
#SBATCH --job-name=sstrans_latin2
#SBATCH --output=./All_Results_with_No_FocalLoss/sstrans/UDIADS_BIB_MS/latin2_%j.out
#SBATCH --error=./All_Results_with_No_FocalLoss/sstrans/UDIADS_BIB_MS/latin2_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# Load modules
module purge
module load python/pytorch2.6py3.12
module load cuda/11.8
module load cudnn

# Create output directory
mkdir -p ./All_Results_with_No_FocalLoss/sstrans/UDIADS_BIB_MS

conda activate pytorch2.6-py3.12

# Set PyTorch CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Manuscript to train/test
MANUSCRIPT="Latin2"

echo "========================================"
echo "=== Training SSTrans: $MANUSCRIPT ==="
echo "========================================"

# Generate timestamp for logging
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M')

python3 train.py \
    --cfg "../../common/configs/swin_tiny_patch4_window7_224_lite.yaml" \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript ${MANUSCRIPT} \
    --use_patched_data \
    --batch_size 8 \
    --max_epochs 300 \
    --base_lr 0.00003 \
    --output_dir "./All_Results_with_No_FocalLoss/sstrans/UDIADS_BIB_MS/udiadsbib_patch224_sstrans_${MANUSCRIPT}" \
    2>&1 | tee "training_sstrans_${MANUSCRIPT}_${TIMESTAMP}.txt"

echo ""
echo "========================================"
echo "=== Testing SSTrans: $MANUSCRIPT ==="
echo "========================================"

# Generate timestamp for test logging
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M')

python3 test.py \
    --cfg "../../common/configs/swin_tiny_patch4_window7_224_lite.yaml" \
    --output_dir "./All_Results_with_No_FocalLoss/sstrans/UDIADS_BIB_MS/udiadsbib_patch224_sstrans_${MANUSCRIPT}" \
    --dataset UDIADS_BIB \
    --manuscript ${MANUSCRIPT} \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --use_patched_data \
    --use_tta \
    --inference_batch_size 24 \
    --is_savenii \
    2>&1 | tee "test_sstrans_${MANUSCRIPT}_${TIMESTAMP}.txt"

echo ""
echo "========================================"
echo "=== Completed: $MANUSCRIPT ==="
echo "========================================"


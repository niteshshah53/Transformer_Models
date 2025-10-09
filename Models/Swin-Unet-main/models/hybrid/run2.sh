#!/bin/bash -l
#SBATCH --job-name=hybrid_train_test
#SBATCH --output=./All_Results_with_No_FocalLoss/hybrid2/UDIADS_BIB_FS/train_test_all_%j.out
#SBATCH --error=./All_Results_with_No_FocalLoss/hybrid2/UDIADS_BIB_FS/train_test_all_%j.out
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
mkdir -p ../../logs

# Training configuration for Hybrid2:
# - model: hybrid2 (Swin-EfficientNet hybrid)
# - dataset: UDIADS_BIB (5 classes for Syr341FS, 6 classes for others)
# - efficientnet_variant: b4 (balanced performance)
# - base_lr: Initial learning rate
# - patience: Early stopping patience

conda activate pytorch2.6-py3.12

# Set PyTorch CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Train all manuscripts one by one (Latin2 Latin14396 Latin16746 Syr341) (CB55, CSG18, CSG863)
MANUSCRIPTS=(Latin2FS Latin14396FS Latin16746FS Syr341FS) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo "=== Training Hybrid2 $MANUSCRIPT ==="
    python3 train.py \
        --model hybrid2 \
        --efficientnet_variant b4 \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-FS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --batch_size 16 \
        --max_epochs 300 \
        --base_lr 0.0002 \
        --patience 50 \
        --output_dir "./All_Results_with_No_FocalLoss/hybrid2/UDIADS_BIB_FS/udiadsbib_patch224_hybrid2_${MANUSCRIPT}"

    echo "=== Testing Hybrid2 $MANUSCRIPT ==="
    python3 test.py \
        --model hybrid2 \
        --efficientnet_variant b4 \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-FS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --use_tta \
        --output_dir "./All_Results_with_No_FocalLoss/hybrid2/UDIADS_BIB_FS/udiadsbib_patch224_hybrid2_${MANUSCRIPT}"
done

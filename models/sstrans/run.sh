#!/bin/bash -l
#SBATCH --job-name=sstrans_train_test
#SBATCH --output=./All_Results_with_No_FocalLoss/sstrans/UDIADS_BIB_MS/train_test_all_%j.out
#SBATCH --error=./All_Results_with_No_FocalLoss/sstrans/UDIADS_BIB_MS/train_test_all_%j.out
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

# Training configuration for SSTrans:
# - model: sstrans (requires config file)
# - dataset: UDIADS_BIB (5 classes for Syr341, 6 classes for others)
# - batch_size: 32 (increased for faster training)
# - max_epochs: 300
# - base_lr: 0.00003 (reduced for longer, more stable training)
# - patience: 50 epochs (hardcoded default in trainer.py)

conda activate pytorch2.6-py3.12

# Set PyTorch CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Train all manuscripts one by one (Latin2 Latin14396 Latin16746 Syr341)
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo "========================================"
    echo "=== Training SSTrans: $MANUSCRIPT ==="
    echo "========================================"

    python3 train.py \
        --cfg "../../common/configs/swin_tiny_patch4_window7_224_lite.yaml" \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --batch_size 32 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --output_dir "./All_Results_with_No_FocalLoss/sstrans/UDIADS_BIB_MS/udiadsbib_patch224_sstrans_${MANUSCRIPT}"

    echo ""
    echo "========================================"
    echo "=== Testing SSTrans: $MANUSCRIPT ==="
    echo "========================================"
    
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
    
    echo ""
    echo "=== Completed: $MANUSCRIPT ==="
    echo ""
done

echo "========================================"
echo "=== All manuscripts completed! ==="
echo "========================================"

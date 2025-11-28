#!/bin/bash -l
#SBATCH --job-name=2nd
#SBATCH --output=./Results/a2/swintiny_%j.out
#SBATCH --error=./Results/a2/swintiny_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=22:00:00
#SBATCH --gres=gpu:1

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# Load modules
module purge
module load python/pytorch2.6py3.12
module load cuda/11.8
module load cudnn

# Create logs directory 
mkdir -p ./Results/a2

# Training configuration for SwinUnet:
# - model: swinunet (requires config file)
# - dataset: UDIADS_BIB (5 classes for Syr341FS, 6 classes for others)
# - base_lr: Initial learning rate
# - patience: Early stopping patience
# - scheduler_type: Learning rate scheduler (ReduceLROnPlateau for better convergence)
# - warmup_epochs: Warmup epochs for scheduler
# - use_amp: Automatic Mixed Precision for 2-3x faster training
# - use_balanced_sampler: Oversample rare classes for better class balance
# - use_class_aware_aug: Stronger augmentation for rare classes (Title, Paratext, Decoration)
# - use_tta: Test-Time Augmentation for improved test accuracy (8 augmentations averaged)
# - multiscale: Multi-scale testing (0.75x, 1.0x, 1.25x) for improved accuracy on varying text sizes

conda activate pytorch2.6-py3.12

# Train all manuscripts one by one (Latin2 Latin14396 Latin16746 Syr341) (CB55, CSG18, CSG863)
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    # echo "=== Training $MANUSCRIPT ==="
    # python3 train.py \
    #     --model swinunet \
    #     --dataset UDIADS_BIB \
    #     --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    #     --manuscript ${MANUSCRIPT} \
    #     --use_patched_data \
    #     --batch_size 8 \
    #     --max_epochs 300 \
    #     --base_lr 0.0002 \
    #     --warmup_epochs 20 \
    #     --patience 60 \
    #     --yaml swintiny \
    #     --num_workers 8 \
    #     --cfg "../../common/configs/swin_tiny_patch4_window7_224_lite.yaml" \
    #     --scheduler_type CosineAnnealingWarmRestarts \
    #     --use_class_aware_aug \
    #     --output_dir "./Results/a2/${MANUSCRIPT}"

    echo "=== Testing $MANUSCRIPT ==="
    python3 test.py \
        --yaml swintiny \
        --cfg "../../common/configs/swin_tiny_patch4_window7_224_lite.yaml" \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --use_tta \
        --multiscale \
        --output_dir "./Results/a2/${MANUSCRIPT}"
done

#!/bin/bash -l
#SBATCH --job-name=baseline1_aff_ds
#SBATCH --output=./Results/a1/baseline1_aff_ds_%j.out
#SBATCH --error=./Results/a1/baseline1_aff_ds_%j.out
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
mkdir -p ../../logs

# ============================================================================
# HYBRID2 BASELINE CONFIGURATION + SMART SKIP CONNECTIONS
# ============================================================================
# This script trains and tests Hybrid2 Baseline model with Smart Skip Connections
#
# ARCHITECTURE:
#   - Encoder: Swin Transformer (4 stages: 96→192→384→768 channels)
#   - Bottleneck: 2 Swin Transformer blocks (768 dim, 24 heads, aligned to SwinUnet)
#   - Decoder: EfficientNet-B4 MBConv blocks (256→128→64→32 channels)
#
# COMPONENTS IN USE (with --use_baseline + --use_smart_skip):
#   ✓ Smart Skip Connections (enabled via --use_smart_skip)
#     → Attention-based fusion with improved skip connection pattern (AFF)
#   ✓ GroupNorm (always enabled - replaces BatchNorm)
#     → All normalization layers use GroupNorm for better training stability
#   ✓ Positional Embeddings (always enabled)
#     → Learnable 2D positional embeddings matching SwinUnet pattern
#   ✓ Bottleneck: 2 Swin blocks (always enabled for baseline)
#   ✗ Deep Supervision (disabled by default)
#     → Use --deep_supervision flag to enable 3 auxiliary outputs
#   ✗ CBAM Attention (disabled by default)
#     → Use --use_cbam flag to enable channel+spatial attention
#   ✗ Cross-Attention Bottleneck (disabled by default)
#     → Use --use_cross_attn flag to enable cross-attention fusion
#   ✗ Multi-Scale Aggregation (disabled by default)
#     → Use --use_multiscale_agg flag to enable multi-scale feature fusion
#
# DECODER OPTIONS:
#   - 'simple': Simple CNN blocks (Conv2d + GroupNorm + ReLU) - BACKUP/ORIGINAL
#   - 'EfficientNet-B4': Actual EfficientNet-B4 MBConv blocks with SE attention (IN USE)
#   - 'ResNet50': Actual ResNet50 Bottleneck blocks with residual connections
#
# TRAINING SETTINGS:
#   - Batch size: 4
#   - Max epochs: 300
#   - Base learning rate: 0.0001
#   - Scheduler: OneCycleLR (steps per batch)
#   - Early stopping patience: 100 epochs
#   - Differential LR: Encoder (0.1x), Bottleneck (0.5x), Decoder (1.0x)
#
# TESTING SETTINGS:
#   - Test-Time Augmentation (TTA): Enabled (4 augmentations: original, h-flip, v-flip, rotate)
#   - CRF Post-processing: Enabled (DenseCRF with spatial+color pairwise potentials)
# ============================================================================

conda activate pytorch2.6-py3.12

# Train all manuscripts one by one (Latin2 Latin14396 Latin16746 Syr341)
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341)

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo "=== Training $MANUSCRIPT (BASELINE + SMART SKIP) ==="
    python3 train.py \
        --model hybrid2 \
        --use_baseline \
        --decoder EfficientNet-B4 \
        --use_smart_skip \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --batch_size 4 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 100 \
        --deep_supervision \
        --scheduler_type CosineAnnealingWarmRestarts \
        --output_dir "./Results/a1/${MANUSCRIPT}"

    echo "=== Testing $MANUSCRIPT (BASELINE + SMART SKIP) ==="
    python3 test.py \
        --use_baseline \
        --decoder EfficientNet-B4 \
        --use_smart_skip \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --use_tta \
        --use_crf \
        --deep_supervision \
        --output_dir "./Results/a1/${MANUSCRIPT}"
done

echo "=== All Training and Testing Completed ==="


#!/bin/bash -l
#SBATCH --job-name=baseline1_aff_msa_ds
#SBATCH --output=./Results/a2/baseline1_aff_msa_ds_%j.out
#SBATCH --error=./Results/a2/baseline1_aff_msa_ds_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=22:00:00
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080

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
# HYBRID2 BASELINE CONFIGURATION + SMART SKIP CONNECTIONS + MULTI-SCALE AGGREGATION
# ============================================================================
# This script trains and tests Hybrid2 Baseline model with:
#   - Smart Skip Connections (AFF - Attention Feature Fusion)
#   - Multi-Scale Aggregation (MSA)
#
# ARCHITECTURE:
#   - Encoder: Swin Transformer (4 stages: 96→192→384→768 channels)
#   - Bottleneck: 2 Swin Transformer blocks (768 dim, 24 heads, aligned to SwinUnet)
#   - Decoder: EfficientNet-B4 MBConv blocks (256→128→64→32 channels)
#
# COMPONENTS IN USE (with --use_baseline + --use_smart_skip + --use_multiscale_agg):
#   ✓ Smart Skip Connections (enabled via --use_smart_skip)
#     → Attention-based fusion with improved skip connection pattern (AFF)
#   ✓ Multi-Scale Aggregation (enabled via --use_multiscale_agg)
#     → Aggregates features from all encoder stages in the bottleneck
#   ✓ GroupNorm (always enabled - replaces BatchNorm)
#     → All normalization layers use GroupNorm for better training stability
#   ✓ Positional Embeddings (always enabled)
#     → Learnable 2D positional embeddings matching SwinUnet pattern
#   ✓ Bottleneck: 2 Swin blocks (always enabled for baseline)
#   ✓ Deep Supervision (enabled)
#     → Use --deep_supervision flag to enable 3 auxiliary outputs
#   ✗ CBAM Attention (disabled by default)
#     → Use --use_cbam flag to enable channel+spatial attention
#   ✗ Cross-Attention Bottleneck (disabled by default)
#     → Use --use_cross_attn flag to enable cross-attention fusion
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
#   - Scheduler: CosineAnnealingWarmRestarts
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
    echo "=== Training $MANUSCRIPT (BASELINE + SMART SKIP + MSA) ==="
    python3 train.py \
        --model hybrid2 \
        --use_baseline \
        --decoder EfficientNet-B4 \
        --use_smart_skip \
        --use_multiscale_agg \
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
        --output_dir "./Results/a2/${MANUSCRIPT}"

    echo "=== Testing $MANUSCRIPT (BASELINE + SMART SKIP + MSA) ==="
    python3 test.py \
        --use_baseline \
        --decoder EfficientNet-B4 \
        --use_smart_skip \
        --use_multiscale_agg \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --use_tta \
        --use_crf \
        --deep_supervision \
        --output_dir "./Results/a2/${MANUSCRIPT}"
done

echo "=== All Training and Testing Completed ==="


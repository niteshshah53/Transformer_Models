#!/bin/bash -l
#SBATCH --job-name=baseline_groupnorm
#SBATCH --output=./Results/UDIADS_BIB_MS/baseline_groupnorm_%j.out
#SBATCH --error=./Results/UDIADS_BIB_MS/baseline_groupnorm_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=22:00:00
#SBATCH --gres=gpu:1

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# ============================================================================
# HYBRID2 BASELINE TRAINING SCRIPT
# ============================================================================
# Model: Hybrid2 Baseline (Swin Transformer Encoder + EfficientNet Decoder)
# Dataset: U-DIADS-Bib-MS_patched (Full-Size patched dataset)
# Manuscripts: Latin2, Latin14396, Latin16746, Syr341



# Hybrid2 Baseline consists of:
#
# ENCODER:
#   ✓ Swin Transformer Encoder (4 stages)
#     - Stage 1: 96 dim, 3 heads, 2 blocks, resolution: H/4 × W/4
#     - Stage 2: 192 dim, 6 heads, 2 blocks, resolution: H/8 × W/8
#     - Stage 3: 384 dim, 12 heads, 2 blocks, resolution: H/16 × W/16
#     - Stage 4: 768 dim, 24 heads, 2 blocks, resolution: H/32 × W/32
#     - Patch Embedding: 4×4 patches, 3 → 96 channels
#     - Patch Merging: 2×2 downsampling between stages
#     - Window Attention: 7×7 windows with relative position bias
#
# BOTTLENECK:
#   ✓ 2 Swin Transformer Blocks (768 dim, 24 heads)
#     - Resolution: H/32 × W/32
#     - Window size: 7×7
#     - Drop path rate: 0.1
#
# DECODER:
#   ✓ EfficientNet-B4 Style CNN Decoder
#     - Decoder channels: [256, 128, 64, 32]
#     - Upsampling: Bilinear interpolation + Conv layers
#     - Normalization: BatchNorm (baseline)
#     - Activation: ReLU
#
# SKIP CONNECTIONS:
#   ✓ Simple Skip Connections (token → CNN conversion)
#     - Converts encoder tokens to CNN features via projection
#     - Concatenates with decoder features
#     - No attention-based fusion (baseline)
#
# POSITIONAL EMBEDDINGS:
#   ✓ 2D Learnable Positional Embeddings (ENABLED by default)
#     - Matches SwinUnet pattern (relative position bias in Swin blocks)
#     - Added to bottleneck features before decoder
#     - Can be disabled with --no_pos_embed flag
#
# OPTIONAL FEATURES (DISABLED in baseline):
#   ✗ Deep Supervision (--use_deep_supervision)
#   ✗ CBAM Attention (--use_cbam)
#   ✗ Smart Skip Connections (--use_smart_skip)
#   ✗ Cross-Attention Bottleneck (--use_cross_attn)
#   ✗ Multi-Scale Aggregation (--use_multiscale_agg)
#   ✓ GroupNorm (--use_groupnorm) - uses BatchNorm instead
# ============================================================================

# Train all manuscripts one by one (Latin2 Latin14396 Latin16746 Syr341)
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341)

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Training Hybrid2 Baseline Model: $MANUSCRIPT"
    echo "========================================================================"
    echo "Dataset: U-DIADS-Bib-MS_patched"
    echo "Architecture: Swin Transformer Encoder → 2 Swin Blocks Bottleneck → EfficientNet-B4 Decoder"
    echo "Components Enabled:"
    echo "  ✓ Swin Encoder (4 stages)"
    echo "  ✓ Bottleneck: 2 Swin Transformer blocks"
    echo "  ✓ EfficientNet-B4 Decoder"
    echo "  ✓ Simple Skip Connections"
    echo "  ✓ Positional Embeddings (default: True)"
    echo "  ✓ GroupNorm (baseline normalization: uses BatchNorm instead)"
    echo "Components Disabled (baseline):"
    echo "  ✗ Deep Supervision"
    echo "  ✗ CBAM Attention"
    echo "  ✗ Smart Skip Connections"
    echo "  ✗ Cross-Attention Bottleneck"
    echo "  ✗ Multi-Scale Aggregation"
    echo "========================================================================"
    
    python3 train.py \
        --model hybrid2 \
        --use_baseline \
        --efficientnet_variant b4 \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --batch_size 4 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 150 \
        --use_groupnorm \
        --scheduler_type ReduceLROnPlateau \
        --output_dir "./Results/UDIADS_BIB_MS/${MANUSCRIPT}"

    echo ""
    echo "========================================================================"
    echo "Testing Hybrid2 Baseline Model: $MANUSCRIPT"
    echo "========================================================================"
    
    python3 test.py \
        --model hybrid2 \
        --use_baseline \
        --efficientnet_variant b4 \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --use_tta \
        --use_crf \
        --use_groupnorm \
        --output_dir "./Results/UDIADS_BIB_MS/${MANUSCRIPT}"
done

echo ""
echo "========================================================================"
echo "ALL MANUSCRIPTS COMPLETED - HYBRID2 BASELINE MODEL"
echo "========================================================================"
echo "Results saved in: ./Results/UDIADS_BIB_MS/"
echo ""

# Aggregate results across all manuscripts
echo ""
echo "========================================================================"
echo "AGGREGATING RESULTS ACROSS ALL MANUSCRIPTS"
echo "========================================================================"

python3 aggregate_results.py \
    --results_dir "./Results/UDIADS_BIB_MS" \
    --manuscripts Latin2 Latin14396 Latin16746 Syr341 \
    --output "./Results/UDIADS_BIB_MS/aggregated_metrics.txt"

echo ""
echo "========================================================================"
echo "AGGREGATION COMPLETE"
echo "========================================================================"
echo "Aggregated metrics saved to: ./Results/UDIADS_BIB_MS/aggregated_metrics.txt"
echo ""

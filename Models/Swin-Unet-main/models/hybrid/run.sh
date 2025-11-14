#!/bin/bash -l
#SBATCH --job-name=r1
#SBATCH --output=./Result/a1/hybrid3_aff_%j.out
#SBATCH --error=./Result/a1/hybrid3_aff_%j.out
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

conda activate pytorch2.6-py3.12

# Add user site-packages to PYTHONPATH to find user-installed packages like pydensecrf2
export PYTHONPATH="${HOME}/.local/lib/python3.12/site-packages:${PYTHONPATH}"

# ============================================================================
# HYBRID2 BASELINE TRAINING SCRIPT WITH DEEP SUPERVISION
# ============================================================================
# Model: Hybrid2 Baseline (Swin Transformer Encoder + Simple CNN Decoder)
# Dataset: U-DIADS-Bib-MS_patched (Full-Size patched dataset)
# Manuscripts: Latin2, Latin14396, Latin16746, Syr341
#   
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
#     - Resolution: H/32 × W/32 (7×7 for 224×224 input)
#     - Window size: 7×7
#     - Drop path rate: 0.1
#     - Processes Stage 4 tokens directly (no dimension reduction)
#     - Output projected from 768 dim to decoder input dim
#
# DECODER:
#   ✓ Simple CNN Decoder (default when --use_baseline is used)
#     - EfficientNet-inspired channel configuration (b4 variant by default)
#     - Decoder channels: [256, 128, 64, 32] (b4 variant)
#     - Upsampling: Bilinear interpolation + Conv layers
#     - Normalization: GroupNorm (default, better for small batches)
#     - Activation: ReLU
#     - Can be replaced with EfficientNet-B4 or ResNet50 via --decoder flag
#
# SKIP CONNECTIONS:
#   ✓ Smart Skip Connections (--use_smart_skip)
#     - Attention-based fusion of encoder and decoder features
#     - Uses channel attention and spatial attention
#     - Better feature fusion than simple concatenation
#
# POSITIONAL EMBEDDINGS:
#   ✓ 2D Learnable Positional Embeddings (ENABLED by default)
#     - Matches SwinUnet pattern (relative position bias in Swin blocks)
#     - Added to bottleneck features before decoder
#     - Can be disabled with --no_pos_embed flag
#
# OPTIONAL FEATURES (DISABLED in baseline):
#   ✓ Deep Supervision (--use_deep_supervision)
#   ✗ CBAM Attention (--use_cbam)
#   ✗ Smart Skip Connections (--use_smart_skip)
#   ✗ Cross-Attention Bottleneck (--use_cross_attn)
#   ✗ Multi-Scale Aggregation (--use_multiscale_agg)
#   ✗ BatchNorm (--use_batchnorm)
#   ✓ GroupNorm (default: enabled, uses GroupNorm)
# ============================================================================

# Train all manuscripts one by one (Latin2 Latin14396 Latin16746 Syr341)
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341)

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Training Hybrid2 Baseline Model with Smart Skip Connections: $MANUSCRIPT"
    echo "========================================================================"
    echo "Dataset: U-DIADS-Bib-MS_patched"
    echo "Architecture: Swin Transformer Encoder → 2 Swin Blocks Bottleneck → Simple CNN Decoder"
    echo "Decoder: Simple Decoder (EfficientNet-b4 channel configuration)"
    echo ""
    echo "Components Enabled:"
    echo "  ✓ Swin Encoder (4 stages: 96→192→384→768 dim)"
    echo "  ✓ Bottleneck: 2 Swin Transformer blocks (768 dim, 24 heads)"
    echo "  ✓ Simple CNN Decoder (channels: [256, 128, 64, 32])"
    echo "  ✓ Smart Skip Connections"
    echo "  ✓ Positional Embeddings (default: enabled)"
    echo "  ✓ GroupNorm (default normalization)"
    echo ""
    echo "Components Disabled (baseline):"
    echo "  ✗ CBAM Attention"
    echo "  ✗ Cross-Attention Bottleneck"
    echo "  ✗ BatchNorm"
    echo "  ✗ Deep Supervision"
    echo "  ✗ Multi-Scale Aggregation"
    echo "========================================================================"
    
    python3 train.py \
        --use_baseline \
        --use_smart_skip \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --batch_size 4 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 150 \
        --scheduler_type OneCycleLR \
        --output_dir "./Result/a1/${MANUSCRIPT}"

    echo ""
    echo "========================================================================"
    echo "Testing Hybrid2 Baseline Model with Smart Skip Connections: $MANUSCRIPT"
    echo "========================================================================"
    
    python3 test.py \
        --use_baseline \
        --use_smart_skip \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --use_tta \
        --use_crf \
        --output_dir "./Result/a1/${MANUSCRIPT}"
done

echo ""
echo "========================================================================"
echo "ALL MANUSCRIPTS COMPLETED - HYBRID2 BASELINE MODEL WITH SMART SKIP CONNECTIONS"
echo "========================================================================"
echo "Model: Hybrid2 Baseline (Swin Encoder + Simple Decoder with Smart Skip Connections)"
echo "Results saved in: ./Result/a1/"
echo ""

# Aggregate results across all manuscripts
echo ""
echo "========================================================================"
echo "AGGREGATING RESULTS ACROSS ALL MANUSCRIPTS"
echo "========================================================================"

python3 aggregate_results.py \
    --results_dir "./Result/a1/" \
    --manuscripts Latin2 Latin14396 Latin16746 Syr341 \
    --output "./Result/a1/aggregated_metrics.txt"

echo ""
echo "========================================================================"
echo "AGGREGATION COMPLETE"
echo "========================================================================"
echo "Aggregated metrics saved to: ./Result/a1/aggregated_metrics.txt"
echo ""
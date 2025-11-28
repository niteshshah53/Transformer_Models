#!/bin/bash -l
#SBATCH --job-name=r3
#SBATCH --output=./Result/a3/hybrid3_aff_msa_ds_%j.out
#SBATCH --error=./Result/a3/hybrid3_aff_msa_ds_%j.out
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
# Model: Hybrid2 Baseline (Swin Transformer Encoder + EfficientNet-B4 Decoder)
# Dataset: U-DIADS-Bib-MS_patched (Full-Size patched dataset)
# Manuscripts: Latin2, Latin14396, Latin16746, Syr341
#   
# Hybrid2 Baseline consists of:
#
# ENCODER:
#   ✓ Swin Transformer Encoder (SimMIM config: 4 stages)
#     - Stage 1: 128 dim, 4 heads, 2 blocks, resolution: H/4 × W/4
#     - Stage 2: 256 dim, 8 heads, 2 blocks, resolution: H/8 × W/8
#     - Stage 3: 512 dim, 16 heads, 18 blocks, resolution: H/16 × W/16
#     - Stage 4: 1024 dim, 32 heads, 2 blocks, resolution: H/32 × W/32
#     - Patch Embedding: 4×4 patches, 3 → 128 channels
#     - Patch Merging: 2×2 downsampling between stages
#     - Window Attention: 7×7 windows with relative position bias
#     - Pretrained: SimMIM checkpoint (simmim_finetune__swin_base__img224_window7__100ep.pth)
#
# BOTTLENECK:
#   ✓ 2 Swin Transformer Blocks (1024 dim, 32 heads)
#     - Resolution: H/32 × W/32 (7×7 for 224×224 input)
#     - Window size: 7×7
#     - Drop path rate: 0.1
#     - Processes Multi-Scale Aggregation output or Stage 4 tokens (1024 dim)
#     - Output projected from 1024 dim to decoder input dim (32 channels)
#
# DECODER:
#   ✓ EfficientNet-B4 Decoder (Real MBConv blocks, not simple CNN)
#     - Decoder channels: [256, 128, 64, 32] (b4 variant)
#     - Upsampling: Bilinear interpolation + MBConv blocks
#     - Normalization: GroupNorm (default, better for small batches)
#     - Activation: ReLU (within MBConv blocks)
#     - Uses actual Mobile Inverted Bottleneck Convolution (MBConv) blocks
#     - Can be replaced with Simple or ResNet50 via --decoder flag
#
# MULTI-SCALE AGGREGATION:
#   ✓ Multi-Scale Aggregation (--use_multiscale_agg)
#     - Aggregates features from all 4 encoder stages (f1, f2, f3, f4)
#     - Projects all stages to common channel dimension and fuses them
#     - Output feeds the bottleneck (replaces direct Stage 4 input)
#     - Provides richer multi-scale context to the bottleneck
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
#   ✗ Cross-Attention Bottleneck (--use_cross_attn)
#   ✗ BatchNorm (--use_batchnorm)
# ============================================================================

# Train all manuscripts one by one (Latin2 Latin14396 Latin16746 Syr341)
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341)

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Training Hybrid2 Baseline Model with Smart Skip (AFF) + Multi-Scale Aggregation + Deep Supervision: $MANUSCRIPT"
    echo "========================================================================"
    echo "Dataset: U-DIADS-Bib-MS_patched"
    echo "Architecture: Swin Transformer Encoder → 2 Swin Blocks Bottleneck → EfficientNet-B4 Decoder"
    echo "Decoder: EfficientNet-B4 Decoder (Real MBConv blocks)"
    echo ""
    echo "Components Enabled:"
    echo "  ✓ Swin Encoder (SimMIM config: 128 dim, depths [2,2,18,2], heads [4,8,16,32])"
    echo "  ✓ Bottleneck: 2 Swin Transformer blocks (1024 dim, 32 heads)"
    echo "  ✓ EfficientNet-B4 Decoder (Real MBConv blocks, channels: [256, 128, 64, 32])"
    echo "  ✓ Smart Skip Connections (Attention-based Features)"
    echo "  ✓ Multi-Scale Aggregation"
    echo "  ✓ Deep Supervision (Multi-resolution)"
    echo "  ✓ Positional Embeddings (default: enabled)"
    echo "  ✓ GroupNorm (default normalization)"
    echo "  ✓ Balanced Sampler (oversampling rare classes)"
    echo "  ✓ Class-Aware Augmentation"
    echo "  ✓ SimMIM Pretrained Weights (from config)"
    echo ""
    echo "Components Disabled:"
    echo "  ✗ CBAM Attention"
    echo "  ✗ Cross-Attention Bottleneck"
    echo "  ✗ BatchNorm"
    echo "========================================================================"
    
    python3 train.py \
        --use_baseline \
        --decoder EfficientNet-B4 \
        --yaml simmim \
        --use_smart_skip \
        --use_multiscale_agg \
        --use_deep_supervision \
        --use_balanced_sampler \
        --use_class_aware_aug \
        --focal_gamma 2.0 \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --batch_size 12 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 150 \
        --scheduler_type OneCycleLR \
        --amp_opt_level O1 \
        --output_dir "./Result/a3/${MANUSCRIPT}"

    echo ""
    echo "========================================================================"
    echo "Testing Hybrid2 Baseline Model with Smart Skip (AFF) + Multi-Scale Aggregation + Deep Supervision: $MANUSCRIPT"
    echo "========================================================================"
    
    python3 test.py \
        --use_baseline \
        --decoder EfficientNet-B4 \
        --yaml simmim \
        --use_smart_skip \
        --use_multiscale_agg \
        --use_deep_supervision \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --use_tta \
        --amp-opt-level O1 \
        --output_dir "./Result/a3/${MANUSCRIPT}"
done

echo ""
echo "========================================================================"
echo "ALL MANUSCRIPTS COMPLETED - HYBRID2 BASELINE MODEL WITH SMART SKIP (AFF) + MULTI-SCALE AGGREGATION + DEEP SUPERVISION"
echo "========================================================================"
echo "Model: Hybrid2 Baseline (Swin Encoder + EfficientNet-B4 Decoder with Smart Skip (AFF) + Multi-Scale Aggregation + Deep Supervision)"
echo "Results saved in: ./Result/a3/"
echo ""

# Aggregate results across all manuscripts
echo ""
echo "========================================================================"
echo "AGGREGATING RESULTS ACROSS ALL MANUSCRIPTS"
echo "========================================================================"

python3 aggregate_results.py \
    --results_dir "./Result/a3/" \
    --manuscripts Latin2 Latin14396 Latin16746 Syr341 \
    --output "./Result/a3/aggregated_metrics.txt"

echo ""
echo "========================================================================"
echo "AGGREGATION COMPLETE"
echo "========================================================================"
echo "Aggregated metrics saved to: ./Result/a3/aggregated_metrics.txt"
echo ""
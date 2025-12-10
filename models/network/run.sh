#!/bin/bash -l
#SBATCH --job-name=d_d
#SBATCH --output=./Result/simmim/network_divahisdb_%j.out
#SBATCH --error=./Result/simmim/network_divahisdb_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100

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

# Memory optimization: Reduce CUDA memory fragmentation
# This helps prevent OOM errors during TTA (Test-Time Augmentation) which processes 4 augmentations at once
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# BASELINE + DEEP SUPERVISION + SMART FEATURE FUSION CONFIGURATION
# ============================================================================
# Enhanced Model Configuration:
#   ✓ EfficientNet-B4 Encoder
#   ✓ Bottleneck: 2 Swin Transformer blocks (enabled)
#   ✓ Swin Transformer Decoder
#   ✓ Fusion Method: smart (attention-based skip connections)
#   ✓ Deep Supervision: ENABLED (multi-resolution auxiliary outputs)
#   ✓ Adapter mode: streaming (integrated adapters)
#   ✓ GroupNorm: enabled
#   ✓ Loss functions: CB Loss (Class-Balanced) + Focal (γ=2.0) + Dice
#   ✓ Differential LR: Encoder (0.05x), Bottleneck (1.0x), Decoder (1.0x)
#   ✓ Balanced Sampler: DISABLED (oversamples rare classes)
#   ✓ Class-Aware Augmentation: ENABLED (stronger augmentation for rare classes)
#   ✓ Class-Balanced Loss: ENABLED (beta=0.9999, best for extreme imbalance >100:1)
#
# Deep Supervision Components:
#   ✓ 3 auxiliary outputs at decoder stages
#   ✓ Outputs at native resolutions (H/16, H/8, H/4)
#   ✓ Multi-resolution loss computation (MSAGHNet-style)
#   ✓ Ground truth downsampled to match resolutions
#
# Smart Feature Fusion Components:
#   ✓ MultiheadAttention-based fusion for skip connections
#   ✓ Attention mechanism selects relevant features
#   ✓ Applied at 3 skip connection stages
#
# Components Disabled:
#   ✗ SE-MSFE
#   ✗ MSFA+MCT Bottleneck
#   ✗ GCFF Fusion
#   ✗ Multi-Scale Aggregation
#   ✗ Fourier Feature Fusion
# ============================================================================

echo "============================================================================"
echo "BASELINE + DEEP SUPERVISION + SMART FEATURE FUSION"
echo "============================================================================"
echo "Configuration: BASELINE + DEEP SUPERVISION + SMART FEATURE FUSION"
echo ""
echo "Component Details:"
echo "  ✓ EfficientNet-B4 Encoder"
echo "  ✓ Bottleneck: 2 Swin Transformer blocks (enabled)"
echo "  ✓ Swin Transformer Decoder"
echo "  ✓ Fusion Method: smart (attention-based skip connections)"
echo "    - MultiheadAttention-based fusion"
echo "    - Attention mechanism selects relevant features"
echo "    - Applied at 3 skip connection stages"
echo "  ✓ Deep Supervision: ENABLED (multi-resolution auxiliary outputs)"
echo "    - 3 auxiliary outputs at decoder stages"
echo "    - Outputs at native resolutions (H/16, H/8, H/4)"
echo "    - Multi-resolution loss computation (MSAGHNet-style)"
echo "    - Ground truth downsampled to match resolutions"
echo "  ✓ Adapter mode: streaming (integrated)"
echo "  ✓ GroupNorm: enabled"
echo "  ✓ Balanced Sampler: DISABLED (oversamples rare classes)"
echo "  ✓ Class-Aware Augmentation: ENABLED (stronger augmentation for rare classes)"
echo "  ✓ Loss: CB Loss (Class-Balanced, beta=0.9999) + Focal (γ=2.0) + Dice"
echo "  ✗ SE-MSFE: disabled"
echo "  ✗ MSFA+MCT Bottleneck: disabled"
echo "  ✗ GCFF Fusion: disabled"
echo "  ✗ Multi-Scale Aggregation: disabled"
echo "  ✗ Fourier Feature Fusion: disabled"
echo ""
echo "Training Parameters:"
echo "  - Batch Size: 16"
echo "  - Max Epochs: 300"
echo "  - Learning Rate: 0.0001"
echo "  - Scheduler: CosineAnnealingWarmRestarts"
echo "  - Early Stopping: 150 epochs patience"
echo "============================================================================"
echo ""

# Train all manuscripts one by one
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║  TRAINING BASELINE + DEEP SUPERVISION + SMART: $MANUSCRIPT"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Configuration: BASELINE + DEEP SUPERVISION + SMART FEATURE FUSION"
    echo "Output Directory: ./Result/simmim/${MANUSCRIPT}"
    echo ""
    
    python3 train.py \
        --cfg "../../common/configs/network_cnn_transformer.yaml" \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --scheduler_type CosineAnnealingWarmRestarts \
        --batch_size 16 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 70 \
        --encoder_lr_factor 0.05 \
        --use_cb_loss \
        --cb_beta 0.9999 \
        --bottleneck \
        --adapter_mode streaming \
        --fusion_method smart \
        --deep_supervision \
        --use_groupnorm \
        --focal_gamma 2.0 \
        --use_class_aware_aug \
        --use_amp \
        --output_dir "./Result/simmim/${MANUSCRIPT}"
    
    TRAIN_EXIT_CODE=$?
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "╔════════════════════════════════════════════════════════════════════════╗"
        echo "║  ✓ TRAINING COMPLETED: $MANUSCRIPT"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Proceeding to testing..."
        echo ""
        
        echo "╔════════════════════════════════════════════════════════════════════════╗"
        echo "║  TESTING BASELINE + DEEP SUPERVISION + SMART: $MANUSCRIPT"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Test Configuration:"
        echo "  ✓ Test-Time Augmentation (TTA): ENABLED"
        echo "  ✗ CRF Post-processing: DISABLED"
        echo "  - Batch Size: 1 (reduced for TTA memory efficiency)"
        echo ""
        
        # Use batch_size=1 for testing to avoid OOM with TTA (4 augmentations per patch = 4x memory)
        # TTA processes 4 augmentations at once, so batch_size=1 means 4 patches in memory simultaneously
        python3 test.py \
            --cfg "../../common/configs/network_cnn_transformer.yaml" \
            --dataset UDIADS_BIB \
            --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
            --manuscript ${MANUSCRIPT} \
            --use_patched_data \
            --is_savenii \
            --use_tta \
            --batch_size 1 \
            --bottleneck \
            --adapter_mode streaming \
            --fusion_method smart \
            --deep_supervision \
            --use_groupnorm \
            --output_dir "./Result/simmim/${MANUSCRIPT}"
        
        TEST_EXIT_CODE=$?
        
        if [ $TEST_EXIT_CODE -eq 0 ]; then
            echo ""
            echo "╔════════════════════════════════════════════════════════════════════════╗"
            echo "║  ✓ TESTING COMPLETED: $MANUSCRIPT"
            echo "╚════════════════════════════════════════════════════════════════════════╝"
            echo ""
        else
            echo ""
            echo "╔════════════════════════════════════════════════════════════════════════╗"
            echo "║  ✗ TESTING FAILED: $MANUSCRIPT (Exit Code: $TEST_EXIT_CODE)"
            echo "╚════════════════════════════════════════════════════════════════════════╝"
            echo ""
        fi
    else
        echo ""
        echo "╔════════════════════════════════════════════════════════════════════════╗"
        echo "║  ✗ TRAINING FAILED: $MANUSCRIPT (Exit Code: $TRAIN_EXIT_CODE)"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Skipping testing for $MANUSCRIPT due to training failure."
        echo ""
    fi
done

echo ""
echo "============================================================================"
echo "ALL MANUSCRIPTS PROCESSED"
echo "============================================================================"
echo "Configuration Used: BASELINE + DEEP SUPERVISION + SMART FEATURE FUSION"
echo "Results Location: ./Result/simmim/"
echo "============================================================================"
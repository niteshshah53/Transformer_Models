#!/bin/bash -l
#SBATCH --job-name=baseline_gcff_ds
#SBATCH --output=./Result/a2/baseline_gcff_ds_%j.out
#SBATCH --error=./Result/a2/baseline_gcff_ds_%j.out
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

# Memory optimization: Reduce CUDA memory fragmentation
# This helps prevent OOM errors during TTA (Test-Time Augmentation) which processes 4 augmentations at once
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# CNN-TRANSFORMER BASE MODEL + GCFF + DEEP SUPERVISION CONFIGURATION
# ============================================================================
# Base Model Configuration with GCFF and Deep Supervision:
#   ✓ EfficientNet-B4 Encoder
#   ✓ Bottleneck: 2 Swin Transformer blocks (enabled)
#   ✓ Swin Transformer Decoder
#   ✓ Fusion Method: gcff (Global Context Feature Fusion from MSAGHNet)
#   ✓ Deep Supervision: ENABLED (multi-resolution auxiliary outputs)
#   ✓ Adapter mode: streaming (integrated adapters)
#   ✓ GroupNorm: enabled
#   ✓ Loss functions: CB Loss (Class-Balanced) + Focal (γ=2.0) + Dice
#   ✓ Differential LR: Encoder (0.05x), Bottleneck (1.0x), Decoder (1.0x)
#   ✓ Balanced Sampler: ENABLED (oversamples rare classes)
#   ✓ Class-Aware Augmentation: ENABLED (stronger augmentation for rare classes)
#   ✓ Class-Balanced Loss: ENABLED (beta=0.9999, best for extreme imbalance >100:1)
#
# GCFF Components:
#   ✓ Global Context Block (attention-based pooling + MLP)
#   ✓ Channel Attention Module (max/avg pool + MLP)
#   ✓ Applied at 3 skip connection stages
#
# Deep Supervision Components:
#   ✓ 3 auxiliary outputs at decoder stages (native resolutions)
#   ✓ Ground truth downsampled to match auxiliary output resolutions
#   ✓ Simple convolutional heads for auxiliary outputs
#   ✓ Multi-resolution loss computation (MSAGHNet-style)
#
# Components Disabled:
#   ✗ SE-MSFE
#   ✗ Multi-Scale Aggregation
#   ✗ Smart Skip Connections
#   ✗ Fourier Feature Fusion
# ============================================================================

echo "============================================================================"
echo "CNN-TRANSFORMER BASE MODEL + GCFF + DEEP SUPERVISION"
echo "============================================================================"
echo "Configuration: CNN-TRANSFORMER BASE MODEL + GCFF + DEEP SUPERVISION"
echo ""
echo "Component Details:"
echo "  ✓ EfficientNet-B4 Encoder"
echo "  ✓ Bottleneck: 2 Swin Transformer blocks (enabled)"
echo "  ✓ Swin Transformer Decoder"
echo "  ✓ Fusion Method: gcff (Global Context Feature Fusion from MSAGHNet)"
echo "    - Global Context Block (attention pooling + MLP)"
echo "    - Channel Attention Module (max/avg pool + MLP)"
echo "    - Applied at 3 skip connection stages"
echo "  ✓ Deep Supervision: ENABLED (multi-resolution auxiliary outputs)"
echo "    - 3 auxiliary outputs at decoder stages"
echo "    - Ground truth downsampled to match resolutions"
echo "    - Multi-resolution loss computation (MSAGHNet-style)"
echo "  ✓ Adapter mode: streaming (integrated)"
echo "  ✓ GroupNorm: enabled"
echo "  ✓ Balanced Sampler: ENABLED (oversamples rare classes)"
echo "  ✓ Class-Aware Augmentation: ENABLED (stronger augmentation for rare classes)"
echo "  ✓ Loss: CB Loss (Class-Balanced, beta=0.9999) + Focal (γ=2.0) + Dice"
echo "  ✗ SE-MSFE: disabled"
echo "  ✗ Multi-Scale Aggregation: disabled"
echo "  ✗ Smart Skip Connections: disabled (using GCFF)"
echo "  ✗ Fourier Feature Fusion: disabled (using GCFF)"
echo ""
echo "Training Parameters:"
echo "  - Batch Size: 12"
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
    echo "║  TRAINING CNN-TRANSFORMER BASE MODEL + GCFF + DEEP SUPERVISION: $MANUSCRIPT"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Configuration: CNN-TRANSFORMER BASE MODEL + GCFF + DEEP SUPERVISION"
    echo "Output Directory: ./Result/a2/${MANUSCRIPT}"
    echo ""
    
    python3 train.py \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --scheduler_type CosineAnnealingWarmRestarts \
        --batch_size 24 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 150 \
        --encoder_lr_factor 0.05 \
        --use_cb_loss \
        --cb_beta 0.9999 \
        --bottleneck \
        --adapter_mode streaming \
        --fusion_method gcff \
        --deep_supervision \
        --use_groupnorm \
        --focal_gamma 2.0 \
        --use_balanced_sampler \
        --use_class_aware_aug \
        --use_amp \
        --output_dir "./Result/a2/${MANUSCRIPT}"
    
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
        echo "║  TESTING CNN-TRANSFORMER BASE MODEL + GCFF + DEEP SUPERVISION: $MANUSCRIPT"
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
            --dataset UDIADS_BIB \
            --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
            --manuscript ${MANUSCRIPT} \
            --use_patched_data \
            --is_savenii \
            --use_tta \
            --batch_size 1 \
            --bottleneck \
            --adapter_mode streaming \
            --fusion_method gcff \
            --deep_supervision \
            --use_groupnorm \
            --output_dir "./Result/a2/${MANUSCRIPT}"

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
echo "Configuration Used: CNN-TRANSFORMER BASE MODEL + GCFF + DEEP SUPERVISION"
echo "Results Location: ./Result/a2/"
echo "============================================================================"
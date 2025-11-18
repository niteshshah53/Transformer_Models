#!/bin/bash -l
#SBATCH --job-name=1st      
#SBATCH --output=./Result/a1/baseline_resnet50_syr341_%j.out
#SBATCH --error=./Result/a1/baseline_resnet50_syr341_%j.out
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
# BASE NETWORK MODEL WITH RESNET-50 ENCODER
# ============================================================================
# Base Model Configuration with ResNet-50 Encoder:
#   ✓ ResNet-50 Encoder (official)
#   ✓ Bottleneck: 2 Swin Transformer blocks (enabled)
#   ✓ Swin Transformer Decoder
#   ✓ Fusion Method: simple (concatenation)
#   ✓ Adapter mode: streaming (integrated adapters)
#   ✓ GroupNorm: enabled
#   ✓ Loss functions: CB Loss (Class-Balanced) + Focal (γ=2.0) + Dice
#   ✓ Differential LR: Encoder (0.05x), Bottleneck (1.0x), Decoder (1.0x)
#   ✓ Balanced Sampler: ENABLED (oversamples rare classes)
#   ✓ Class-Aware Augmentation: ENABLED (stronger augmentation for rare classes)
#   ✓ Class-Balanced Loss: ENABLED (beta=0.9999, best for extreme imbalance >100:1)
#   ✓ Numerical Stability: Reduced LR (0.00005), AMP disabled, gradient clipping (encoder: 5.0, decoder: 1.0)
#
# Components Disabled:
#   ✗ Deep Supervision
#   ✗ Fourier Feature Fusion
#   ✗ Smart Skip Connections
#   ✗ Multi-Scale Aggregation
# ============================================================================

echo "============================================================================"
echo "CNN-TRANSFORMER BASE MODEL WITH RESNET-50 ENCODER"
echo "============================================================================"
echo "Configuration: CNN-TRANSFORMER BASE MODEL (ResNet-50 Encoder)"
echo ""
echo "Component Details:"
echo "  ✓ ResNet-50 Encoder (official)"
echo "  ✓ Bottleneck: 2 Swin Transformer blocks (enabled)"
echo "  ✓ Swin Transformer Decoder"
echo "  ✓ Fusion Method: simple (concatenation)"
echo "  ✓ Adapter mode: streaming (integrated)"
echo "  ✓ GroupNorm: enabled"
echo "  ✓ Balanced Sampler: ENABLED (oversamples rare classes)"
echo "  ✓ Class-Aware Augmentation: ENABLED (stronger augmentation for rare classes)"
echo "  ✓ Loss: CB Loss (Class-Balanced, beta=0.9999) + Focal (γ=2.0) + Dice"
echo "  ✗ Deep Supervision: disabled"
echo "  ✗ Multi-Scale Aggregation: disabled"
echo "  ✗ Fourier Feature Fusion: disabled (using simple fusion)"
echo "  ✗ Smart Skip Connections: disabled (using simple fusion)"
echo ""
echo "Training Parameters:"
echo "  - Batch Size: 12 (best result configuration)"
echo "  - Max Epochs: 300"
echo "  - Learning Rate: 0.00005 (reduced for numerical stability)"
echo "  - Scheduler: CosineAnnealingWarmRestarts"
echo "  - Early Stopping: 150 epochs patience"
echo "  - AMP: DISABLED (for numerical stability)"
echo "  - Gradient Clipping: Encoder (5.0), Decoder (1.0)"
echo ""
echo "Configuration:"
echo "  ✓ Balanced sampler: ENABLED"
echo "  ✓ Class-aware augmentation: ENABLED"
echo "  ✓ Focal gamma: 2.0"
echo "============================================================================"
echo ""

# Train all manuscripts one by one Latin2 Latin14396 Latin16746 Syr341
MANUSCRIPTS=(Syr341) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║  TRAINING CNN-TRANSFORMER BASE MODEL (ResNet-50): $MANUSCRIPT"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Configuration: CNN-TRANSFORMER BASE MODEL WITH RESNET-50 ENCODER"
    echo "Output Directory: ./Result/a1/${MANUSCRIPT}"
    echo ""
    
    python3 train.py \
        --encoder_type resnet50 \
        --bottleneck \
        --adapter_mode streaming \
        --fusion_method simple \
        --use_groupnorm \
        --focal_gamma 2.0 \
        --use_balanced_sampler \
        --use_class_aware_aug \
        --use_cb_loss \
        --cb_beta 0.9999 \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --scheduler_type CosineAnnealingWarmRestarts \
        --batch_size 12 \
        --max_epochs 300 \
        --base_lr 0.00005 \
        --patience 150 \
        --encoder_lr_factor 0.05 \
        --output_dir "./Result/a1/${MANUSCRIPT}"
    
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
        echo "║  TESTING CNN-TRANSFORMER BASE MODEL (ResNet-50): $MANUSCRIPT"
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
            --encoder_type resnet50 \
            --bottleneck \
            --adapter_mode streaming \
            --fusion_method simple \
            --use_groupnorm \
            --output_dir "./Result/a1/${MANUSCRIPT}"
        
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
echo "Configuration Used: CNN-TRANSFORMER BASE MODEL WITH RESNET-50 ENCODER"
echo "Results Location: ./Result/a1/"
echo "============================================================================"
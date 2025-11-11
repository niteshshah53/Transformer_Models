#!/bin/bash -l
#SBATCH --job-name=h1_baseline2           
#SBATCH --output=./UDIADS_BIB_MS/baseline2_%j.out
#SBATCH --error=./UDIADS_BIB_MS/baseline2_%j.out
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
# BASELINE NETWORK MODEL CONFIGURATION
# ============================================================================
# Baseline Configuration:
#   ✓ EfficientNet-B4 Encoder
#   ✓ Bottleneck: 2 Swin Transformer blocks (automatically enabled)
#   ✓ Swin Transformer Decoder
#   ✓ Simple concatenation skip connections
#   ✓ Adapter mode: streaming (default)
#   ✓ GroupNorm: enabled (default)
#   ✓ All three losses: CE + Dice + Focal
#   ✓ Differential LR: Encoder (0.1x), Bottleneck (0.5x), Decoder (1.0x)
#
# Components Disabled (baseline):
#   ✗ Deep Supervision
#   ✗ Smart Skip Connections
#   ✗ Fourier Feature Fusion
#   ✗ Multi-Scale Aggregation
# ============================================================================

echo "============================================================================"
echo "CNN-TRANSFORMER BASELINE NETWORK MODEL"
echo "============================================================================"
echo "Configuration: BASELINE ONLY"
echo ""
echo "Component Details:"
echo "  ✓ EfficientNet-B4 Encoder"
echo "  ✓ Bottleneck: 2 Swin Transformer blocks"
echo "  ✓ Swin Transformer Decoder"
echo "  ✓ Simple concatenation skip connections"
echo "  ✓ Adapter mode: streaming"
echo "  ✓ GroupNorm: enabled"
echo "  ✓ Loss: CE + Dice + Focal (0.3*CE + 0.2*Focal + 0.5*Dice)"
echo "  ✓ Differential LR: Encoder (0.1x), Bottleneck (0.5x), Decoder (1.0x)"
echo ""
echo "Training Parameters:"
echo "  - Batch Size: 4"
echo "  - Max Epochs: 300"
echo "  - Learning Rate: 0.0001"
echo "  - Scheduler: CosineAnnealingWarmRestarts"
echo "  - Early Stopping: 100 epochs patience"
echo "============================================================================"
echo ""

# Train all manuscripts one by one
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║  TRAINING BASELINE: $MANUSCRIPT"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Configuration: BASELINE ONLY"
    echo "Output Directory: ./UDIADS_BIB_MS/${MANUSCRIPT}"
    echo ""
    
    python3 train.py \
        --use_baseline \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --scheduler_type CosineAnnealingWarmRestarts \
        --batch_size 4 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 100 \
        --output_dir "./UDIADS_BIB_MS/${MANUSCRIPT}"
    
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
        echo "║  TESTING BASELINE: $MANUSCRIPT"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Test Configuration:"
        echo "  ✓ Test-Time Augmentation (TTA): ENABLED"
        echo "  ✓ CRF Post-processing: ENABLED"
        echo ""
        
        python3 test.py \
            --use_baseline \
            --dataset UDIADS_BIB \
            --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
            --manuscript ${MANUSCRIPT} \
            --use_patched_data \
            --is_savenii \
            --use_tta \
            --use_crf \
            --output_dir "./UDIADS_BIB_MS/${MANUSCRIPT}"
        
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
echo "Configuration Used: BASELINE ONLY"
echo "Results Location: ./UDIADS_BIB_MS/"
echo "============================================================================"
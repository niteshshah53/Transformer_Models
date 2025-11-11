#!/bin/bash -l
#SBATCH --job-name=h11_baseline2       
#SBATCH --output=./a1/baseline2_AFF_%j.out
#SBATCH --error=./a1/baseline2_AFF_%j.out
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

conda activate pytorch2.6-py3.12

# Add user site-packages to PYTHONPATH to find user-installed packages like pydensecrf2
export PYTHONPATH="${HOME}/.local/lib/python3.12/site-packages:${PYTHONPATH}"

# ============================================================================
# BASELINE NETWORK MODEL CONFIGURATION + SMART SKIP CONNECTIONS
# ============================================================================
# Configuration: BASELINE + SMART SKIP
#   ✓ EfficientNet-B4 Encoder
#   ✓ Bottleneck: 2 Swin Transformer blocks (automatically enabled)
#   ✓ Swin Transformer Decoder
#   ✓ Smart skip connections (attention-based fusion)
#   ✓ Adapter mode: streaming (default)
#   ✓ GroupNorm: enabled (default)
#   ✓ All three losses: CE + Dice + Focal
#   ✓ Differential LR: Encoder (0.1x), Bottleneck (0.5x), Decoder (1.0x)
#
# Components Enabled:
#   ✓ Smart Skip Connections (fusion_method='smart')
#
# Components Disabled:
#   ✗ Deep Supervision
#   ✗ Fourier Feature Fusion
#   ✗ Multi-Scale Aggregation
# ============================================================================

echo "============================================================================"
echo "CNN-TRANSFORMER BASELINE NETWORK MODEL + SMART SKIP CONNECTIONS"
echo "============================================================================"
echo "Configuration: BASELINE + SMART SKIP"
echo ""
echo "Component Details:"
echo "  ✓ EfficientNet-B4 Encoder"
echo "  ✓ Bottleneck: 2 Swin Transformer blocks"
echo "  ✓ Swin Transformer Decoder"
echo "  ✓ Smart skip connections (attention-based fusion: fusion_method='smart')"
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
    echo "║  TRAINING BASELINE + SMART SKIP: $MANUSCRIPT"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Configuration: BASELINE + SMART SKIP"
    echo "Output Directory: ./a1/${MANUSCRIPT}"
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
        --fusion_method smart \
        --output_dir "./a1/${MANUSCRIPT}"
    
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
        echo "║  TESTING BASELINE + SMART SKIP: $MANUSCRIPT"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""
        
        python3 test.py \
            --use_baseline \
            --fusion_method smart \
            --dataset UDIADS_BIB \
            --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
            --manuscript ${MANUSCRIPT} \
            --use_patched_data \
            --is_savenii \
            --use_tta \
            --use_crf \
            --output_dir "./a1/${MANUSCRIPT}"
        
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
echo "Configuration Used: BASELINE + SMART SKIP"
echo "Results Location: ./a1/"
echo "============================================================================"
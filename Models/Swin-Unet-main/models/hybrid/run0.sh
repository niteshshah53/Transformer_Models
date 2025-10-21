#!/bin/bash -l
#SBATCH --job-name=Hybrid1_optuna_train_test
#SBATCH --output=./Results_Optimized_Hyperparameters/Hybrid1/UDIADS_BIB_FS/train_test_optuna_%j.out
#SBATCH --error=./Results_Optimized_Hyperparameters/Hybrid1/UDIADS_BIB_FS/train_test_optuna_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
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
mkdir -p ./Results_Optimized_Hyperparameters/Hybrid1/UDIADS_BIB_FS

# Training configuration for Hybrid1 with OPTUNA-OPTIMIZED HYPERPARAMETERS:
# - model: Hybrid1 (Swin-EfficientNet Encoder + TransUNet Decoder)
# - dataset: UDIADS_BIB (5 classes for Syr341, 6 classes for others)
# - base_lr: 0.0002 (Optuna Trial 31, Val Loss: 0.228)
# - batch_size: 8 (Optuna-optimized)
# - patience: 50 (Optuna-optimized)

conda activate base

# Set PyTorch CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Train all manuscripts one by one (Latin2 Latin14396 Latin16746 Syr341) (CB55, CSG18, CSG863)
MANUSCRIPTS=(Latin2FS Latin14396FS Latin16746FS Syr341FS)

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo "=== Training Hybrid1-cnn_swin$MANUSCRIPT ==="
    python3 train.py \
        --model hybrid1 \
        --use_efficientnet \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-FS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --batch_size 8 \
        --max_epochs 300 \
        --base_lr 0.0002 \
        --patience 50 \
        --output_dir "./Results_Optimized_Hyperparameters/Hybrid1/UDIADS_BIB_FS/udiadsbib_Hybrid1_${MANUSCRIPT}"

    echo "=== Testing Hybrid1-Swin-EfficientNet $MANUSCRIPT ==="
    python3 test.py \
        --model hybrid1 \
        --use_efficientnet \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-FS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --output_dir "./Results_Optimized_Hyperparameters/Hybrid1/UDIADS_BIB_FS/udiadsbib_Hybrid1_${MANUSCRIPT}"
done

echo ""
echo "========================================================================"
echo "ALL MANUSCRIPTS COMPLETED - SWIN-EFFICIENTNET DECODER!"
echo "========================================================================"
echo "Results saved in: ./Results_Optimized_Hyperparameters/Hybrid1/UDIADS_BIB_FS/"
echo ""
echo "Model: Hybrid1-Swin-EfficientNet (Swin Encoder + TransUNet Decoder)"
echo "Architecture: Swin Encoder → EfficientNet Decoder"
echo "Features:"
echo "  ✓ Deep Supervision (auxiliary outputs)"
echo ""
echo "Performance Expectations:"
echo "  • Baseline Hybrid1: IoU 0.36"
echo "  • Expected IoU: 0.50-0.55 (+50-69% improvement!)"
echo "  • Swin Encoder + EfficientNet Decoder"
echo ""
echo "Date: $(date)"
echo "========================================================================"

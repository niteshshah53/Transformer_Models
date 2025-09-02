#!/bin/bash -l
#SBATCH --job-name=das_train_test
#SBATCH --output=All_Results/missformer/Full_UDIADS_BIB/train_test_all_%j.out
#SBATCH --error=All_Results/missformer/Full_UDIADS_BIB/train_test_all_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# Load modules
module purge
module load python/pytorch2.6py3.12
module load cuda/11.8
module load cudnn

# Create logs directory 
mkdir -p logs

# Training configuration for MissFormer:
# - model: missformer (no config file needed, designed for 224x224 input)
# - base_lr: Initial learning rate
# - patience: Early stopping patience (stop if no improvement for N epochs)
# - lr_factor: Factor to reduce learning rate by when plateauing
# - lr_patience: Patience for learning rate reduction
# - lr_min: Minimum learning rate
# - lr_threshold: Threshold for considering improvement

conda activate pytorch2.6-py3.12

MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341)

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo "=== Training $MANUSCRIPT ==="
    python3 train.py \
        --model missformer \
        --dataset UDIADS_BIB \
        --udiadsbib_root "U-DIADS-Bib-MS_patched/${MANUSCRIPT}" \
        --use_patched_data \
        --num_classes 6 \
        --batch_size 16 \
        --max_epochs 300 \
        --base_lr 0.001 \
        --patience 50 \
        --lr_factor 0.5 \
        --lr_patience 10 \
        --lr_min 1e-7 \
        --lr_threshold 1e-4 \
        --output_dir "./All_Results/missformer/Full_UDIADS_BIB/udiadsbib_patch224_missformer_${MANUSCRIPT}"

    echo "=== Testing $MANUSCRIPT ==="
    python3 test.py \
        --model missformer \
        --dataset UDIADS_BIB \
        --udiadsbib_root "U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --num_classes 6 \
        --output_dir "./All_Results/missformer/Full_UDIADS_BIB/udiadsbib_patch224_missformer_${MANUSCRIPT}"

done

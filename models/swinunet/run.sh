#!/bin/bash -l
#SBATCH --job-name=swinunet_train_test
#SBATCH --output=All_Results_with_No_FocalLoss/swinunet/UDIADS_BIB_MS/train_test_all_%j.out
#SBATCH --error=All_Results_with_No_FocalLoss/swinunet/UDIADS_BIB_MS/train_test_all_%j.out
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

# Create logs directory 
mkdir -p ../../logs

# Training configuration for SwinUnet:
# - model: swinunet (requires config file)
# - dataset: UDIADS_BIB (5 classes for Syr341FS, 6 classes for others)
# - base_lr: Initial learning rate
# - patience: Early stopping patience

conda activate pytorch2.6-py3.12

# Train all manuscripts one by one (Latin2 Latin14396 Latin16746 Syr341) (CB55, CSG18, CSG863)
MANUSCRIPTS=(Latin14396 Latin16746 Syr341) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo "=== Training $MANUSCRIPT ==="
    python3 train.py \
        --model swinunet \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --cfg "../../common/configs/swin_tiny_patch4_window7_224_lite.yaml" \
        --use_patched_data \
        --batch_size 32 \
        --max_epochs 300 \
        --base_lr 0.0002 \
        --patience 30 \
        --output_dir "All_Results_with_No_FocalLoss/swinunet/UDIADS_BIB_MS/${MANUSCRIPT}"

    echo "=== Testing $MANUSCRIPT ==="
    python3 test.py \
        --model swinunet \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --cfg "../../common/configs/swin_tiny_patch4_window7_224_lite.yaml" \
        --use_patched_data \
        --is_savenii \
        --output_dir "All_Results_with_No_FocalLoss/swinunet/UDIADS_BIB_MS/${MANUSCRIPT}"
done

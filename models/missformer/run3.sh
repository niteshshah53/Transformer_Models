#!/bin/bash -l
#SBATCH --job-name=missformer_train_test
#SBATCH --output=./All_Results_with_No_FocalLoss/missformer/DIVAHISDB/train_test_all_%j.out
#SBATCH --error=./All_Results_with_No_FocalLoss/missformer/DIVAHISDB/train_test_all_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# Load modules
module purge
module load python/pytorch2.6py3.12
module load cuda/11.8
module load cudnn

# Create logs directory 
mkdir -p ../../logs

# Training configuration for MissFormer on DIVAHISDB:
# - model: hybrid  
# - dataset: DIVAHISDB (4 classes: Background, Comment, Decoration, Main Text)
# - base_lr: Initial learning rate  
# - patience: Early stopping patience

conda activate pytorch2.6-py3.12
# Train all manuscripts one by one for DIVAHISDB (CB55, CSG18, CSG863)
MANUSCRIPTS=(CSG863) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do


    echo "=== Testing $MANUSCRIPT ==="
    python3 test.py \
        --model missformer \
        --dataset DIVAHISDB \
        --divahisdb_root "../../DivaHisDB_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --num_classes 4 \
        --is_savenii \
        --output_dir "./All_Results_with_No_FocalLoss/missformer/DIVAHISDB/divahisdb_patch224_missformer_${MANUSCRIPT}"
done

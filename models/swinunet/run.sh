#!/bin/bash -l
#SBATCH --job-name=S_diva
#SBATCH --output=./Results/diva/simmim_window6_divahisdb_%j.out
#SBATCH --error=./Results/diva/simmim_window6_divahisdb_%j.out
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
mkdir -p ./Results/diva
conda activate pytorch2.6-py3.12

# Train all manuscripts one by one (Latin2, Latin14396, Latin16746, Syr341)
MANUSCRIPTS=(CB55) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    # echo "=== Training $MANUSCRIPT ==="
    # python3 train.py \
    #     --model swinunet \
    #     --dataset DIVAHISDB\
    #     --divahisdb_root "../../DivaHisDB_patched" \
    #     --manuscript ${MANUSCRIPT} \
    #     --use_patched_data \
    #     --batch_size 16 \
    #     --max_epochs 300 \
    #     --base_lr 0.0001 \
    #     --warmup_epochs 20 \
    #     --patience 50 \
    #     --num_workers 8 \
    #     --cfg "../../common/configs/simmim_swin_base_patch4_window7_224.yaml" \
    #     --scheduler_type CosineAnnealingWarmRestarts \
    #     --use_class_aware_aug \
    #     --use_amp \
    #     --output_dir "./Results/diva/${MANUSCRIPT}"

    echo "=== Testing $MANUSCRIPT ==="
    python3 test.py \
        --cfg "../../common/configs/simmim_swin_base_patch4_window7_224.yaml" \
        --dataset DIVAHISDB \
        --divahisdb_root "../../DivaHisDB_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
	    --use_tta \
	    --multiscale \
        --output_dir "./Results/diva/${MANUSCRIPT}"
done

#!/bin/bash -l
#SBATCH --job-name=st_udiadsbib
#SBATCH --output=./Results/udiadsbib_fs/swin_tiny_udiadsbib_fs_%j.out
#SBATCH --error=./Results/udiadsbib_fs/swin_tiny_udiadsbib_fs_%j.out
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
mkdir -p ./Results/udiadsbib_fs
conda activate pytorch2.6-py3.12

# Train all manuscripts one by one (Latin2FS, Latin14396FS, Latin16746FS, Syr341FS)
MANUSCRIPTS= (Latin2FS Latin14396FS Latin16746FS Syr341FS) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo "=== Training $MANUSCRIPT ==="
    python3 train.py \
        --model swinunet \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-FS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --batch_size 32 \
        --max_epochs 100 \
        --base_lr 0.0001 \
        --warmup_epochs 20 \
        --patience 30 \
        --num_workers 8 \
        --cfg "../../common/configs/swin_tiny_patch4_window7_224_lite.yaml" \
        --scheduler_type CosineAnnealingWarmRestarts \
        --use_class_aware_aug \
        --use_amp \
        --output_dir "./Results/udiadsbib_fs/${MANUSCRIPT}"

    echo "=== Testing $MANUSCRIPT ==="
    python3 test.py \
        --cfg "../../common/configs/swin_tiny_patch4_window7_224_lite.yaml" \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-FS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
	    --use_tta \
	    --multiscale \
        --output_dir "./Results/udiadsbib_fs/${MANUSCRIPT}"
done

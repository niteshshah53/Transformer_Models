#!/bin/bash -l
#SBATCH --job-name=a1
#SBATCH --output=./Results/a1/test_%j.out
#SBATCH --error=./Results/a1/test_%j.out
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
mkdir -p ./Results/a1
conda activate pytorch2.6-py3.12

# Train all manuscripts one by one (Latin2, Latin14396, Latin16746, Syr341)
MANUSCRIPTS=(Latin2) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo "=== Testing $MANUSCRIPT ==="
    python3 test.py \
        --cfg "../../common/configs/simmim_swin_base_patch4_window7_224.yaml" \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --output_dir "./Results/a1/${MANUSCRIPT}"
done

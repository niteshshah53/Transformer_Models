#!/bin/bash -l
#SBATCH --job-name=H_diva
#SBATCH --output=./Result/diva/hybrid_simmim_divahisdb_%j.out
#SBATCH --error=./Result/diva/hybrid_simmim_divahisdb_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

conda activate pytorch2.6-py3.12

# Add user site-packages to PYTHONPATH to find user-installed packages like pydensecrf2
export PYTHONPATH="${HOME}/.local/lib/python3.12/site-packages:${PYTHONPATH}"

# Manuscripts to run on DivaHisDB
MANUSCRIPTS=(CB55)

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo ""
    echo "Training Hybrid2 baseline (Swin encoder + EfficientNet-B4 decoder) on DivaHisDB: ${MANUSCRIPT}"
    
    python3 train.py \
        --use_baseline \
        --decoder EfficientNet-B4 \
        --cfg "../../common/configs/simmim_swin_base_patch4_window7_224.yaml" \
        --use_smart_skip \
        --use_multiscale_agg \
        --use_deep_supervision \
        --use_balanced_sampler \
        --use_class_aware_aug \
        --focal_gamma 2.0 \
        --dataset DIVAHISDB \
        --divahisdb_root "../../DivaHisDB_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --batch_size 16 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 70 \
        --scheduler_type CosineAnnealingWarmRestarts \
        --amp_opt_level O1 \
        --output_dir "./Result/diva/${MANUSCRIPT}"

    echo ""
    echo "Testing Hybrid2 baseline on DivaHisDB: ${MANUSCRIPT}"
    
    python3 test.py \
        --use_baseline \
        --decoder EfficientNet-B4 \
        --cfg "../../common/configs/simmim_swin_base_patch4_window7_224.yaml" \
        --use_smart_skip \
        --use_multiscale_agg \
        --use_deep_supervision \
        --dataset DIVAHISDB \
        --divahisdb_root "../../DivaHisDB_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --use_tta \
        --amp-opt-level O1 \
        --output_dir "./Result/diva/${MANUSCRIPT}"
done

echo ""
echo "All DivaHisDB manuscripts completed. Results saved in: ./Result/diva/"
echo ""


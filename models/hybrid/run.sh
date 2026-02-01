#!/bin/bash -l
#SBATCH --job-name=H_udiads
#SBATCH --output=./Result/udiadsbib/hybrid_simmim_udiadsbib_%j.out
#SBATCH --error=./Result/udiadsbib/hybrid_simmim_udiadsbib_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
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


#to use DivaHisDB, change the dataset to DIVAHISDB and the root to ../../DivaHisDB_patched and the manuscript to CB55, CSG18, CSG863
#to use UDIADS-Bib-FS, change the dataset to UDIADS_BIB and the root to ../../U-DIADS-Bib-FS_patched and the manuscript to Latin2FS, Latin14396FS, Latin16746FS, Syr341FS
#to use UDIADS-Bib-MS, change the dataset to UDIADS_BIB and the root to ../../U-DIADS-Bib-MS_patched and the manuscript to Latin2, Latin14396, Latin16746, Syr341

# Manuscripts to run on U-DIADS-Bib-MS
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341)


for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo ""
    echo "Training Hybrid2 baseline (Swin encoder + EfficientNet-B4 decoder) on U-DIADS-Bib-MS: ${MANUSCRIPT}"
    
    python3 train.py \
        --use_baseline \
        --decoder EfficientNet-B4 \
        --cfg "../../common/configs/swin_tiny_patch4_window7_224_lite.yaml" \
        --use_smart_skip \
        --use_multiscale_agg \
        --use_deep_supervision \
        --use_balanced_sampler \
        --use_class_aware_aug \
        --focal_gamma 2.0 \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --batch_size 8 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 70 \
        --scheduler_type CosineAnnealingWarmRestarts \
        --amp_opt_level O1 \
        --output_dir "./Result/udiadsbib/${MANUSCRIPT}"

    echo ""
    echo "Testing Hybrid2 baseline on U-DIADS-Bib-MS: ${MANUSCRIPT}"
    
    python3 test.py \
        --use_baseline \
        --decoder EfficientNet-B4 \
        --cfg "../../common/configs/swin_tiny_patch4_window7_224_lite.yaml" \
        --use_smart_skip \
        --use_multiscale_agg \
        --use_deep_supervision \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --use_tta \
        --amp-opt-level O1 \
        --output_dir "./Result/udiadsbib/${MANUSCRIPT}"
done

echo ""
echo "All U-DIADS-Bib-MS manuscripts completed. Results saved in: ./Result/udiadsbib/"
echo ""


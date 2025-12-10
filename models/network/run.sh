#!/bin/bash -l
#SBATCH --job-name=S_udiadsbib
#SBATCH --output=./Result/simmim_udiadsbib/simmim_network_udiadsbib_%j.out
#SBATCH --error=./Result/simmim_udiadsbib/simmim_network_udiadsbib_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
module purge
module load python/pytorch2.6py3.12
module load cuda/11.8
module load cudnn

conda activate pytorch2.6-py3.12

export PYTHONPATH="${HOME}/.local/lib/python3.12/site-packages:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#to use DivaHisDB, change the dataset to DIVAHISDB and the root to ../../DivaHisDB_patched and the manuscript to CB55, CSG18, CSG863
#to use UDIADS-Bib-FS, change the dataset to UDIADS_BIB and the root to ../../U-DIADS-Bib-FS_patched and the manuscript to Latin2FS, Latin14396FS, Latin16746FS, Syr341FS
#to use UDIADS-Bib-MS, change the dataset to UDIADS_BIB and the root to ../../U-DIADS-Bib-MS_patched and the manuscript to Latin2, Latin14396, Latin16746, Syr341
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    python3 train.py \
        --cfg "../../common/configs/network_cnn_transformer.yaml" \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --scheduler_type CosineAnnealingWarmRestarts \
        --batch_size 16 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 70 \
        --encoder_lr_factor 0.05 \
        --use_cb_loss \
        --cb_beta 0.9999 \
        --bottleneck \
        --adapter_mode streaming \
        --fusion_method smart \
        --deep_supervision \
        --use_groupnorm \
        --focal_gamma 2.0 \
        --use_class_aware_aug \
        --use_amp \
        --output_dir "./Result/simmim_udiadsbib/${MANUSCRIPT}"
    
    TRAIN_EXIT_CODE=$?
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        python3 test.py \
            --cfg "../../common/configs/network_cnn_transformer.yaml" \
            --dataset UDIADS_BIB \
            --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
            --manuscript ${MANUSCRIPT} \
            --use_patched_data \
            --is_savenii \
            --use_tta \
            --batch_size 1 \
            --bottleneck \
            --adapter_mode streaming \
            --fusion_method smart \
            --deep_supervision \
            --use_groupnorm \
            --output_dir "./Result/simmim_udiadsbib/${MANUSCRIPT}"
        
        TEST_EXIT_CODE=$?
        
        if [ $TEST_EXIT_CODE -eq 0 ]; then
            echo "Testing completed: $MANUSCRIPT"
        else
            echo "Testing failed: $MANUSCRIPT (Exit Code: $TEST_EXIT_CODE)"
        fi
    else
        echo "Training failed: $MANUSCRIPT (Exit Code: $TRAIN_EXIT_CODE)"
        echo "Skipping testing for $MANUSCRIPT."
    fi
done

echo "All manuscripts processed. Results: ./Result/simmim_udiadsbib/"
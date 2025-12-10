#!/bin/bash -l
#SBATCH --job-name=diva
#SBATCH --output=./Result/network_divahisdb/baseline_ds_smart_%j.out
#SBATCH --error=./Result/network_divahisdb/baseline_ds_smart_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100

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

# Memory optimization: Reduce CUDA memory fragmentation
# This helps prevent OOM errors during TTA (Test-Time Augmentation) which processes 4 augmentations at once
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Train DivaHisDB manuscripts
MANUSCRIPTS=(CB55)

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo "Training: $MANUSCRIPT"
    
    python3 train.py \
        --cfg "../../common/configs/network_cnn_transformer.yaml" \
        --dataset DIVAHISDB \
        --divahisdb_root "../../DivaHisDB_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --scheduler_type CosineAnnealingWarmRestarts \
        --batch_size 16 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 150 \
        --encoder_lr_factor 0.05 \
        --use_cb_loss \
        --cb_beta 0.9999 \
        --bottleneck \
        --adapter_mode streaming \
        --fusion_method smart \
        --deep_supervision \
        --use_groupnorm \
        --focal_gamma 2.0 \
        --use_balanced_sampler \
        --use_amp \
        --output_dir "./Result/network_divahisdb/${MANUSCRIPT}"
    
    if [ $? -eq 0 ]; then
        echo "Testing: $MANUSCRIPT"
        
        python3 test.py \
            --cfg "../../common/configs/network_cnn_transformer.yaml" \
            --dataset DIVAHISDB \
            --divahisdb_root "../../DivaHisDB_patched" \
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
            --output_dir "./Result/network_divahisdb/${MANUSCRIPT}"
    fi
done

echo "All manuscripts processed. Results: ./Result/network_divahisdb/"

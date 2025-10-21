#!/bin/bash -l
#SBATCH --job-name=optuna_hybrid_tune
#SBATCH --output=./optuna_results/hybrid2/optuna_tune_%j.out
#SBATCH --error=./optuna_results/hybrid2/optuna_tune_%j.out
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

# Create output directories
mkdir -p ./optuna_results
mkdir -p ../../logs

# Optuna Hyperparameter Tuning Configuration:
# - model: hybrid1 or hybrid2
# - n_trials: Number of hyperparameter combinations to test
# - optuna_max_epochs: Reduced epochs per trial for faster tuning (50 instead of 300)
# - optuna_patience: Early stopping patience per trial (15 instead of 50)
# - Hyperparameters to tune:
#   * Learning rate: 1e-5 to 1e-3 (log scale)
#   * Weight decay: 1e-3 to 0.1 (log scale)
#   * Batch size: [8, 16, 24, 32]
#   * Optimizer: [AdamW, Adam, SGD]
#   * Scheduler: [CosineAnnealing, ReduceLROnPlateau, CosineWarmRestarts]

conda activate base

# Set PyTorch CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# ============================================================================
# CONFIGURATION - Modify these variables as needed
# ============================================================================
MODEL="hybrid1"                    # hybrid1 or hybrid2
DATASET="UDIADS_BIB"              # UDIADS_BIB or DIVAHISDB
N_TRIALS=50                        # Number of Optuna trials (start with 20-30 for testing)
MAX_EPOCHS_PER_TRIAL=50           # Max epochs per trial (reduced for speed)
PATIENCE=15                        # Early stopping patience per trial

# Dataset paths
UDIADSBIB_ROOT="../../U-DIADS-Bib-MS_patched"
DIVAHISDB_ROOT="../../DivaHisDB_patched"

# ============================================================================
# LOOP THROUGH ALL MANUSCRIPTS
# ============================================================================

# For UDIADS_BIB: Latin2, Latin14396, Latin16746, Syr341
MANUSCRIPTS=(Syr341 Latin2 Latin14396 Latin16746)

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "STARTING OPTUNA TUNING FOR: ${MANUSCRIPT}"
    echo "========================================================================"
    echo ""
    
    # Output directory for this manuscript
    OUTPUT_DIR="./optuna_results/hybrid2/${DATASET}/${MANUSCRIPT}"
    
    echo "=========================================="
    echo "Starting Optuna Hyperparameter Tuning"
    echo "=========================================="
    echo "Model: ${MODEL}"
    echo "Dataset: ${DATASET}"
    echo "Manuscript: ${MANUSCRIPT}"
    echo "Number of trials: ${N_TRIALS}"
    echo "Max epochs per trial: ${MAX_EPOCHS_PER_TRIAL}"
    echo "Early stopping patience: ${PATIENCE}"
    echo "Output: ${OUTPUT_DIR}"
    echo "=========================================="
    echo ""
    
    # Check if required dependencies are installed
    echo "Checking dependencies..."
    python3 -c "import sympy; import filelock; import optuna" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: Missing required dependencies!"
        echo "Please run: bash fix_dependencies.sh"
        echo "Or install manually:"
        echo "  python3 -m pip install --user sympy filelock fsspec jinja2 networkx python-dateutil"
        echo "  python3 -m pip install --user --force-reinstall 'numpy>=2.0,<2.3.0'"
        exit 1
    fi
    echo "✓ All dependencies OK"
    echo ""
    
    # Run Optuna tuning for this manuscript
    python3 optuna_tune.py \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --manuscript ${MANUSCRIPT} \
        --udiadsbib_root ${UDIADSBIB_ROOT} \
        --divahisdb_root ${DIVAHISDB_ROOT} \
        --use_patched_data \
        --n_trials ${N_TRIALS} \
        --optuna_max_epochs ${MAX_EPOCHS_PER_TRIAL} \
        --optuna_patience ${PATIENCE} \
        --output_dir ${OUTPUT_DIR} \
        --img_size 224 \
        --n_gpu 1 \
        --num_workers 4 \
        --seed 1234
    
    echo ""
    echo "========================================================================"
    echo "COMPLETED OPTUNA TUNING FOR: ${MANUSCRIPT}"
    echo "========================================================================"
    echo "Best parameters: ${OUTPUT_DIR}/optuna_best_params.json"
    echo "Study database: ${OUTPUT_DIR}/optuna_study.db"
    echo "Visualizations: ${OUTPUT_DIR}/*.html"
    echo "========================================================================"
    echo ""
    
    # Display best parameters if available
    if [ -f "${OUTPUT_DIR}/optuna_best_params.json" ]; then
        echo "Best hyperparameters found for ${MANUSCRIPT}:"
        cat "${OUTPUT_DIR}/optuna_best_params.json"
        echo ""
    fi
    
    echo ""
    echo "Moving to next manuscript..."
    echo ""

done

echo ""
echo "========================================================================"
echo "ALL MANUSCRIPTS COMPLETED!"
echo "========================================================================"
echo "Tuned manuscripts: ${MANUSCRIPTS[@]}"
echo ""
echo "Results locations:"
for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo "  - ${MANUSCRIPT}: ./optuna_results/hybrid2/${DATASET}/${MANUSCRIPT}/"
done
echo "========================================================================"
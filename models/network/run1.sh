#!/bin/bash -l
#SBATCH --job-name=vis_baseline_ds      
#SBATCH --output=./gradcam_results/visualize_baseline_ds_%j.out
#SBATCH --error=./gradcam_results/visualize_baseline_ds_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=22:00:00
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

# Memory optimization: Reduce CUDA memory fragmentation
# This helps prevent OOM errors during TTA (Test-Time Augmentation) which processes 4 augmentations at once
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to script directory (where this script is located)
# This ensures we can find visualize_gradcam.py and other modules
SCRIPT_DIR="/home/hpc/iwi5/iwi5250h/Transformer_Models/models/network"
cd "$SCRIPT_DIR"
echo "Changed to script directory: $SCRIPT_DIR"
echo "Current directory: $(pwd)"
echo ""

# ============================================================================
# GRADCAM VISUALIZATION: BASELINE + DEEP SUPERVISION
# ============================================================================
# Generate GradCAM visualizations for Baseline + Deep Supervision models
# 
# Models to Visualize:
#   ✓ Baseline (Simple Skip): ./Result/a1/
#   ✓ Baseline + Deep Supervision: ./Result/a4/
#
# Visualizations Generated:
#   - Component comparison (Baseline vs Baseline+DS)
#   - Encoder GradCAM heatmaps
#   - Predictions comparison
#   - Deep Supervision multi-resolution outputs
#
# Requirements:
#   - pytorch-grad-cam: pip install grad-cam
#   - Trained model checkpoints in Result directories
# ============================================================================

echo "============================================================================"
echo "GRADCAM VISUALIZATION: BASELINE + DEEP SUPERVISION"
echo "============================================================================"
echo "Generating visualizations for:"
echo "  ✓ Baseline Model (Simple Skip Connection)"
echo "  ✓ Baseline + Deep Supervision (MSAGHNet-style multi-resolution)"
echo ""
echo "Visualization Types:"
echo "  ✓ Component comparison (side-by-side GradCAM)"
echo "  ✓ Encoder attention heatmaps"
echo "  ✓ Prediction comparisons"
echo "  ✓ Multi-resolution outputs (for Deep Supervision)"
echo ""
echo "Output Location: ./gradcam_results/"
echo "============================================================================"
echo ""

# Check if grad-cam is installed
python3 -c "import pytorch_grad_cam" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Installing pytorch-grad-cam..."
    pip install grad-cam --quiet
    echo "✓ Installation complete"
    echo ""
fi

# Generate visualizations for all manuscripts
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║  GENERATING VISUALIZATIONS: $MANUSCRIPT"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Models to Visualize:"
    echo "  - Baseline: ./Result/a1/${MANUSCRIPT}"
    echo "  - Baseline + Deep Supervision: ./Result/a4/${MANUSCRIPT}"
    echo ""
    echo "Output Directory: ./gradcam_results/${MANUSCRIPT}"
    echo ""
    
    # Check if model directories exist
    BASELINE_DIR="./Result/a1/${MANUSCRIPT}"
    BASELINE_DS_DIR="./Result/a4/${MANUSCRIPT}"
    
    if [ ! -d "$BASELINE_DIR" ]; then
        echo "⚠️  Warning: Baseline model directory not found: $BASELINE_DIR"
        echo "   Skipping $MANUSCRIPT..."
        continue
    fi
    
    if [ ! -d "$BASELINE_DS_DIR" ]; then
        echo "⚠️  Warning: Baseline+DS model directory not found: $BASELINE_DS_DIR"
        echo "   Will only visualize Baseline model..."
        BASELINE_DS_DIR=""
    fi
    
    # Ensure we're in the script directory (already changed at top of script)
    cd "$SCRIPT_DIR"
    
    # Check if visualize_gradcam.py exists
    if [ ! -f "visualize_gradcam.py" ]; then
        echo "❌ Error: visualize_gradcam.py not found in $SCRIPT_DIR"
        echo "   Current directory: $(pwd)"
        echo "   Files in directory: $(ls -la *.py 2>/dev/null | head -5)"
        continue
    fi
    
    # Run visualization script
    python3 visualize_gradcam.py \
        --dataset_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --baseline_dir "${BASELINE_DIR}" \
        --baseline_ds_dir "${BASELINE_DS_DIR}" \
        --output_dir "./gradcam_results/${MANUSCRIPT}" \
        --num_samples 10
    
    VIS_EXIT_CODE=$?
    
    if [ $VIS_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "╔════════════════════════════════════════════════════════════════════════╗"
        echo "║  ✓ VISUALIZATION COMPLETED: $MANUSCRIPT"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "  Visualizations saved to: ./gradcam_results/${MANUSCRIPT}/"
        echo "  - comparison_image_*.png (Component comparisons)"
        echo "  - attention_heatmap_*.png (if applicable)"
        echo "  - fourier_frequency_*.png (if applicable)"
        echo ""
    else
        echo ""
        echo "╔════════════════════════════════════════════════════════════════════════╗"
        echo "║  ✗ VISUALIZATION FAILED: $MANUSCRIPT (Exit Code: $VIS_EXIT_CODE)"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""
    fi
done

echo ""
echo "============================================================================"
echo "ALL VISUALIZATIONS GENERATED"
echo "============================================================================"
echo "Models Visualized:"
echo "  ✓ Baseline (Simple Skip Connection)"
echo "  ✓ Baseline + Deep Supervision (MSAGHNet-style)"
echo ""
echo "Results Location: ./gradcam_results/"
echo ""
echo "Visualization Files:"
echo "  - comparison_image_*.png: Side-by-side component comparisons"
echo "  - attention_heatmap_*.png: Attention visualizations (if applicable)"
echo "  - fourier_frequency_*.png: Frequency domain plots (if applicable)"
echo ""
echo "Next Steps:"
echo "  1. Review visualizations in ./gradcam_results/"
echo "  2. Select best images for presentation"
echo "  3. Use in slides: GradCAM Analysis, Component Comparisons"
echo "============================================================================"
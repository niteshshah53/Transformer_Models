# Benchmarking state-of-the-art Transformer Models for Handwritten Document Layout Analysis

This repository contains state-of-the-art deep learning models for historical document segmentation, featuring transformer-based architectures and hybrid CNN-transformer combinations optimized for manuscript analysis.

## 📁 Directory Structure

```
Transformer_Models/
├── models/                          # Model-specific implementations
│   ├── sstrans/                     # Smart Swin Transformer
│   │   ├── train.py                 # SSTrans training script
│   │   ├── test.py                  # SSTrans testing script
│   │   ├── trainer.py               # SSTrans-specific trainer
│   │   ├── run.sh                   # SSTrans execution script
│   │   ├── Only_Smart.py            # Smart attention mechanism
│   │   ├── vision_transformer.py    # SSTrans model implementation
│   │   ├── modules.py               # SSTrans modules
│   │   └── ...                      # Other SSTrans-specific files
│   ├── swinunet/                    # Swin Transformer U-Net
│   │   ├── train.py                 # SwinUnet training script
│   │   ├── test.py                  # SwinUnet testing script
│   │   ├── trainer.py               # SwinUnet-specific trainer
│   │   ├── run.sh                   # SwinUnet execution script
│   │   ├── swin_transformer_unet_skip_expand_decoder_sys.py  # Main model
│   │   ├── vision_transformer.py    # Vision transformer components
│   │   └── ...                      # Other SwinUnet-specific files
│   ├── missformer/                  # MissFormer 
│   │   ├── train.py                 # MissFormer training script
│   │   ├── test.py                  # MissFormer testing script
│   │   ├── trainer.py               # MissFormer-specific trainer
│   │   ├── run.sh                   # MissFormer execution script
│   │   ├── MISSFormer.py            # MissFormer model implementation
│   │   ├── segformer.py             # SegFormer backbone
│   │   └── ...                      # Other MissFormer-specific files
│   ├── hybrid/                      # Hybrid CNN-Transformer Models(Encoder: Swin, Decoder: CNN)
│   │   ├── hybrid2/                 # Swin-EfficientNet Hybrid (Enhanced)
│   │   │   ├── model.py             # Main hybrid model
│   │   │   ├── components.py        # Model components
│   │   │   └── Hybrid2_Architecture_Description.tex  # Architecture docs
│   │   ├── train.py                 # Unified training script
│   │   ├── test.py                  # Unified testing script
│   │   ├── trainer.py               # Hybrid-specific trainer
│   │   ├── aggregate_results.py     # Results aggregation script
│   │   ├── run.sh                   # Hybrid execution script
│   │   ├── run1.sh, run2.sh, run3.sh  # Additional run scripts
│   │   └── Result/                  # Training results
│   └── network/                     # Hybrid-1/Network Models (Encoder: CNN, Decoder: Swin)
│       ├── train.py                 # Network training script
│       ├── test.py                  # Network testing script
│       ├── trainer.py               # Network-specific trainer
│       ├── cnn_transformer.py      # CNN-Transformer model
│       ├── vision_transformer_cnn.py  # Vision Transformer CNN model
│       ├── components.py            # Network components
│       ├── visualize_gradcam.py     # GradCAM visualization
│       ├── run.sh                   # Network execution script
│       ├── run1.sh, run2.sh, run3.sh  # Additional run scripts
│       └── Result/                  # Training results
├── common/                          # Shared components
│   ├── datasets/                    # Dataset implementations
│   │   ├── dataset_udiadsbib.py     # U-DIADS-Bib dataset loader (legacy)
│   │   ├── dataset_udiadsbib_2.py   # U-DIADS-Bib dataset loader (enhanced with class-aware augmentation)
│   │   ├── dataset_divahisdb.py     # DivaHisDB dataset loader
│   │   ├── sstrans_transforms.py    # SSTrans-specific transforms
│   │   └── README.md                # Dataset documentation
│   ├── utils/                       # Utility functions
│   │   └── utils.py                 # Common utilities (losses, metrics)
│   └── configs/                     # Configuration files
│       ├── config.py                # Configuration management
│       └── swin_tiny_patch4_window7_224_lite.yaml  # SSTrans config
├── pretrained_ckpt/                 # Pre-trained checkpoints
├── logs/                            # Training logs
├── U-DIADS-Bib-MS/                  # U-DIADS-Bib-MS dataset (original)
├── U-DIADS-Bib-MS_patched/          # U-DIADS-Bib-MS dataset (patched)
├── U-DIADS-Bib-FS/                  # U-DIADS-Bib-FS dataset (original)
├── U-DIADS-Bib-FS_patched/          # U-DIADS-Bib-FS dataset (patched)
├── DivaHisDB/                       # DivaHisDB dataset (original)
├── DivaHisDB_patched/               # DivaHisDB dataset (patched)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Load Python environment (if using module system)
module load python/pytorch2.6py3.12
module load cuda/11.8
module load cudnn

# Activate conda environment
conda activate pytorch2.6-py3.12
```

### Run Individual Models

Each model has its own execution script:

```bash
# SSTrans (Smart Swin Transformer with attention mechanisms)
cd models/sstrans
./run.sh

# SwinUnet (Standard Swin Transformer U-Net)
cd models/swinunet
./run.sh

# MissFormer (Multi-scale Transformer with SegFormer backbone)
cd models/missformer
./run.sh

# Hybrid2 (SwinUnet encoder + Enhanced EfficientNet decoder)
cd models/hybrid
./run.sh

# Network (CNN-Transformer models)
cd models/network
./run.sh
```

### Custom Training

```bash
# Train Hybrid2 with custom parameters
cd models/hybrid
python3 train.py \
    --use_baseline \
    --decoder EfficientNet-B4 \
    --use_smart_skip \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Latin2 \
    --use_patched_data \
    --batch_size 16 \
    --max_epochs 300 \
    --base_lr 0.0001 \
    --patience 150 \
    --output_dir "./Result/a1/Latin2"
```

## 🔧 Model Architectures & Configurations

### SSTrans (Smart Swin Transformer)
- **Architecture**: Enhanced Swin Transformer with smart attention mechanisms
- **Key Features**: 
  - Smart attention masks for improved focus
  - Heavy data augmentation pipeline
  - Advanced normalization strategies
- **Training**: Standardized with validation and early stopping
- **Loss Function**: Combined CE, Focal, and Dice losses
- **Optimizer**: AdamW with weight_decay=0.01
- **Validation**: Sliding window on full images
- **Config File**: Requires `--cfg` parameter pointing to config YAML

### SwinUnet (Swin Transformer U-Net)
- **Architecture**: Standard Swin Transformer with U-Net decoder
- **Key Features**:
  - Skip connections between encoder and decoder
  - Patch merging and expanding operations
  - Window-based self-attention
  - ImageNet-pretrained weights for better initialization
- **Training**: Advanced with class imbalance handling
  - **Class Weighting**: Effective Number of Samples (ENS) method with aggressive rare class boosting
  - **Balanced Sampling**: Optional oversampling of rare classes
  - **Class-Aware Augmentation**: Stronger augmentation for rare classes (Title, Paratext, Decoration, Chapter Headings)
  - **Loss Function**: Combined CE (15%), Focal (55%, γ=5), and Dice (30%) losses
  - **Early Stopping**: Based on mean foreground Dice score (better reflects rare class performance)
  - **Learning Rate Warmup**: Gradual LR increase over first N epochs
  - **Scheduler**: CosineAnnealingWarmRestarts (T_0=50, T_mult=2) optimized for imbalanced data
- **Optimizer**: AdamW with weight_decay=0.01
- **Validation**: Sliding window on full images with per-class Dice tracking
- **Inference**: 
  - **Test-Time Augmentation (TTA)**: 8 augmentations (original, flips, rotations, combinations)
  - **Multi-Scale Testing**: 3 scales (0.75x, 1.0x, 1.25x) with probability averaging
  - **Batch Processing**: Efficient patch batching for faster inference
  - **Probability Accumulation**: Correct overlap handling by accumulating probabilities, not class indices

### MissFormer
- **Architecture**: SegFormer backbone with multi-scale feature fusion
- **Key Features**:
  - Efficient self-attention mechanisms
  - Multi-scale feature aggregation
  - Bridge layers for feature fusion
- **Training**: Advanced with class weights and sliding window validation
- **Loss Function**: Combined CE and Dice losses (with class weights)
- **Optimizer**: AdamW with weight_decay=1e-4
- **Validation**: Advanced sliding window with mask conversion

### Hybrid2 (Swin-EfficientNet Enhanced)
- **Architecture**: SwinUnet encoder + Enhanced EfficientNet decoder
- **Key Features**:
  - **Swin Transformer Encoder**: 4 stages (96→192→384→768 dim)
  - **Bottleneck**: 2 Swin Transformer blocks (768 dim, 24 heads)
  - **EfficientNet-B4 Decoder**: MBConv blocks with channel progression
  - **Smart Skip Connections**: Attention-based feature fusion
  - **Positional Embeddings**: 2D learnable positional embeddings
  - **Deep Supervision**: Optional multi-resolution auxiliary outputs
  - **CBAM Attention**: Optional channel and spatial attention
  - **Feature Refinement**: Gradual channel reduction with residual connections
- **Training**: Advanced with balanced sampling and class-aware augmentation
- **Loss Function**: Combined CE, Focal (γ=2.0), and Dice losses
- **Optimizer**: AdamW with differential learning rates
- **Variants**: B0 (lightweight), B4 (balanced), B5 (heavy)
- **Decoder Options**: EfficientNet-B4, ResNet50, or simple CNN decoder

### Network (EfficientNet-Swin Transformer)
- **Architecture**: Hybrid CNN-Transformer U-Net with EfficientNet/ResNet-50 encoder and Swin Transformer decoder
- **Key Features**:
  - **Encoder Options**: EfficientNet-B4 (default) or ResNet-50 with ImageNet pretrained weights
  - **Swin Transformer Decoder**: 4-stage decoder with window-based self-attention (96→192→384→768 dims)
  - **Feature Adapters**: Convert CNN features to transformer format (external or streaming mode)
  - **Bottleneck**: Optional 2 Swin Transformer blocks or MSFA+MCT hybrid bottleneck
  - **Skip Connection Fusion**: Multiple fusion methods:
    - `simple`: Basic concatenation + linear projection
    - `smart`: Attention-based fusion (cross-attention between encoder/decoder)
    - `fourier`: Fourier feature fusion for frequency domain integration
    - `gcff`: Global Context Feature Fusion (from MSAGHNet)
  - **Deep Supervision**: Optional multi-resolution auxiliary outputs (all upsampled to full resolution)
  - **Multi-Scale Aggregation**: Optional fusion of all encoder scales in bottleneck
  - **SE-MSFE**: Optional Squeeze-Excitation Multi-Scale Feature Extraction (replaces MBConv in EfficientNet)
  - **Normalization**: GroupNorm (default) or LayerNorm for better small-batch training
- **Training**: Advanced with class imbalance handling
  - **Class Weighting**: Effective Number of Samples (ENS) method with capping
  - **Balanced Sampling**: Optional continuous rarity-based oversampling
  - **Loss Function**: Combined CE, Focal (γ=3.0), and Dice losses (configurable weighting)
  - **Optimizer**: AdamW with weight_decay=0.01
  - **Scheduler**: CosineAnnealingWarmRestarts, ReduceLROnPlateau, or OneCycleLR
  - **Mixed Precision**: AMP support for faster training
- **Validation**: Sliding window on full images with per-class metrics
- **Visualization**: GradCAM support for model interpretability

## 📊 Supported Datasets

### U-DIADS-Bib
- **Description**: Historical manuscript segmentation dataset
- **Variants**: 
  - **MS (Full Dataset)**: Full Datasets
  - **FS (Few-Shot)**: Few Images or Few-Shot Dataset
- **Classes**: 6 classes (5 for Syriaque341 manuscripts)
  - Background, Paratext, Decoration, Main Text, Title, Chapter Headings
- **Note**: Syriaque341 manuscripts don't have Chapter Headings (5 classes)
- **Format**: RGB color-coded masks
- **Usage**: `--dataset UDIADS_BIB --use_patched_data --udiadsbib_root "../../U-DIADS-Bib-MS_patched"`

### DIVAHISDB
- **Description**: Historical document analysis dataset
- **Classes**: 4 classes
  - Background, Comment, Decoration, Main Text
- **Format**: Bitmask-encoded masks
- **Usage**: `--dataset DIVAHISDB --use_patched_data`


## 🔧 Key Benefits of Repository Structure

1. **Modularity**: Each model is self-contained with its own implementation
2. **Flexibility**: Easy to experiment with different architectures
3. **Maintainability**: Clear separation between model-specific and shared code
4. **Extensibility**: Simple to add new models or modify existing ones
5. **Reproducibility**: Consistent training and evaluation pipelines
6. **SLURM Integration**: All run scripts include SLURM job submission headers

## 📝 Adding New Models

To add a new model:

1. Create a new folder in `models/`
2. Copy the structure from an existing model (recommend starting with `hybrid/`)
3. Implement your model architecture
4. Modify the training/testing scripts
5. Update the common trainer if needed
6. Create run scripts with SLURM headers if needed

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
- Ensure you're running scripts from the correct directory
- Check that the common directory is in the Python path
- Verify Python environment is properly loaded (`module load python/pytorch2.6py3.12`)
- Add user site-packages to PYTHONPATH: `export PYTHONPATH="${HOME}/.local/lib/python3.12/site-packages:${PYTHONPATH}"`

#### Model-Specific Issues
- **SSTrans**: Requires config file (`--cfg` parameter pointing to YAML config)
- **Hybrid Models**: Check `--decoder` parameter (EfficientNet-B4, ResNet50, or baseline)
- **MissFormer**: Verify SegFormer dependencies
- Check individual `trainer.py` files for model-specific logic

#### Training Issues
- **CUDA Memory**: Reduce batch size if encountering OOM errors
  - Set `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for memory optimization
- **Data Loading**: Ensure dataset paths are correct and accessible
- **Checkpoints**: Verify checkpoint paths and model compatibility
- **SLURM Jobs**: Check job output files in Result directories for detailed error messages

#### Performance Issues
- **Slow Training**: Consider reducing image size or using mixed precision (`--amp_opt_level O1`)
- **Poor Convergence**: Adjust learning rate or try different optimizers
- **Overfitting**: Increase regularization or use more data augmentation
- **Class Imbalance**: Use `--use_balanced_sampler` and `--use_class_aware_aug` flags

## 📈 Model Performance Comparison

| Model | Architecture | Parameters | Memory | Speed | Best For |
|-------|-------------|-----------|--------|-------|----------|
| SSTrans | Smart Swin Transformer | ~28M | Moderate | Fast | Attention-focused tasks |
| SwinUnet | Standard Swin U-Net | ~27M | Moderate | Fast | General segmentation |
| MissFormer | Multi-scale Transformer | ~30M | High | Moderate | Multi-scale features |
| Hybrid2 | Swin-EfficientNet | ~45M | Moderate | Moderate | Enhanced feature extraction |
| Network | CNN-Transformer | Variable | Variable | Variable | Flexible architectures |


This repository provides a comprehensive framework for historical document segmentation with state-of-the-art models and best practices.


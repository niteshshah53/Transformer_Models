# Hybrid Models for Historical Document Segmentation

This directory contains two hybrid models that combine different encoder-decoder architectures:

## Models

### Hybrid1: EfficientNet-Swin
- **Encoder**: EfficientNet-B4 (CNN-based)
- **Decoder**: SwinUnet (Transformer-based)
- **Architecture**: CNN Encoder + Transformer Decoder

### Hybrid2: Swin-EfficientNet Enhanced
- **Encoder**: SwinUnet (Transformer-based)
- **Decoder**: Enhanced EfficientNet-style (CNN-based)
- **Architecture**: Transformer Encoder + CNN Decoder
- **Key Features**: CBAM Attention, Smart Skip Connections, Deep Decoder Blocks

## Directory Structure

```
hybrid/
├── hybrid1/                    # EfficientNet-Swin model
│   ├── hybrid_model.py         # Main model implementation
│   ├── efficientnet_encoder.py # EfficientNet-B4 encoder
│   └── swin_decoder.py         # SwinUnet decoder
├── hybrid2/                    # Swin-EfficientNet model
│   ├── hybrid_model.py         # Main model implementation
│   ├── swin_encoder.py         # SwinUnet encoder
│   └── efficientnet_decoder.py # EfficientNet-style decoder
├── train.py                    # Training script (supports both models)
├── test.py                     # Testing script (supports both models)
├── trainer.py                  # Training logic (shared, SSTrans-aligned)
├── run.sh                      # Run script for hybrid2 (UDIADS_BIB MS)
├── run2.sh                     # Run script for hybrid2 (UDIADS_BIB FS)
├── run3.sh                     # Run script for hybrid2 (DIVAHISDB)
└── README.md                   # This file
```

## Usage

### Training

#### Hybrid1 (EfficientNet-Swin)
```bash
python3 train.py \
    --model hybrid1 \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Latin2 \
    --use_patched_data \
    --batch_size 16 \
    --max_epochs 300 \
    --base_lr 0.0002 \
    --patience 30 \
    --output_dir "./results/hybrid1_latin2"
```

#### Hybrid2 (Swin-EfficientNet)
```bash
python3 train.py \
    --model hybrid2 \
    --efficientnet_variant b4 \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Latin2 \
    --use_patched_data \
    --batch_size 16 \
    --max_epochs 300 \
    --base_lr 0.0002 \
    --patience 30 \
    --output_dir "./results/hybrid2_latin2"
```

### Testing

#### Hybrid1
```bash
python3 test.py \
    --model hybrid1 \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Latin2 \
    --use_patched_data \
    --is_savenii \
    --output_dir "./results/hybrid1_latin2"
```

#### Hybrid2
```bash
python3 test.py \
    --model hybrid2 \
    --efficientnet_variant b4 \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Latin2 \
    --use_patched_data \
    --is_savenii \
    --use_tta \
    --output_dir "./results/hybrid2_latin2"
```

## Command Line Arguments

### Common Arguments
- `--model`: Model type (`hybrid1` or `hybrid2`)
- `--dataset`: Dataset to use (`UDIADS_BIB` or `DIVAHISDB`)
- `--manuscript`: Manuscript name (e.g., `Latin2`, `Syr341`)
- `--use_patched_data`: Use pre-generated patches
- `--batch_size`: Batch size for training/testing
- `--max_epochs`: Maximum number of training epochs
- `--base_lr`: Initial learning rate
- `--patience`: Early stopping patience
- `--output_dir`: Directory to save results

### Hybrid2 Specific Arguments
- `--efficientnet_variant`: EfficientNet variant for decoder (`b0`, `b4`, `b5`)

### Testing Arguments
- `--use_tta`: Enable Test-Time Augmentation for improved accuracy (+2-4% mIoU)
- `--is_savenii`: Save prediction visualizations

## Model Configurations

### Hybrid1 (EfficientNet-Swin)
- **Loss Function**: 0.3 * CE + 0.4 * Focal + 0.3 * Dice (with class weights)
- **Optimizer**: AdamW with weight_decay=0.05
- **Scheduler**: CosineAnnealingWarmRestarts
- **Early Stopping**: Yes (patience=50 epochs)
- **Class Weights**: Computed from pixel frequency

### Hybrid2 (Swin-EfficientNet Enhanced)
- **Loss Function**: 0.3 * CE + 0.4 * Focal + 0.3 * Dice (with class weights)
- **Optimizer**: AdamW with weight_decay=0.05
- **Scheduler**: CosineAnnealingWarmRestarts
- **Early Stopping**: Yes (patience=50 epochs)
- **Class Weights**: Computed from pixel frequency
- **EfficientNet Variants**: B0 (lightweight), B4 (balanced), B5 (heavy)
- **Enhanced Features**: CBAM Attention, Smart Skip Connections, Deep Decoder Blocks

## Supported Datasets

### U-DIADS-Bib
- **Classes**: 6 classes (5 for Syriaque341 manuscripts)
- **Classes**: Background, Paratext, Decoration, Main Text, Title, Chapter Headings
- **Note**: Syriaque341 manuscripts don't have Chapter Headings (5 classes)

### DIVAHISDB
- **Classes**: 4 classes
- **Classes**: Background, Comment, Decoration, Main Text

## Run Scripts

### Quick Start
```bash
# Run Hybrid2 on U-DIADS-Bib MS dataset (Latin2, Latin14396, Latin16746, Syr341)
./run.sh

# Run Hybrid2 on U-DIADS-Bib FS dataset (Latin2FS, Latin14396FS, Latin16746FS, Syr341FS)
./run2.sh

# Run Hybrid2 on DIVAHISDB dataset (CB55, CSG18, CSG863)
./run3.sh
```

### Custom Runs
```bash
# Train Hybrid1 with custom parameters
python3 train.py --model hybrid1 --manuscript Latin2 --batch_size 16

# Train Hybrid2 with EfficientNet-B4 decoder (recommended)
python3 train.py --model hybrid2 --efficientnet_variant b4 --manuscript Latin2

# Test with TTA enabled for better accuracy
python3 test.py --model hybrid2 --efficientnet_variant b4 --use_tta --manuscript Latin2
```

## Model Comparison

| Aspect | Hybrid1 (EfficientNet-Swin) | Hybrid2 (Swin-EfficientNet Enhanced) |
|--------|------------------------------|---------------------------------------|
| Encoder | EfficientNet-B4 (CNN) | SwinUnet (Transformer) |
| Decoder | SwinUnet (Transformer) | Enhanced EfficientNet-style (CNN) |
| Parameters | ~50M | ~45M |
| Memory Usage | Moderate | Moderate |
| Training Speed | Fast | Moderate |
| Inference Speed | Fast | Fast |
| Special Features | Standard architecture | CBAM Attention, Smart Skip Connections |
| Best For | Quick training, good performance | Enhanced feature extraction, attention mechanisms |

## Recent Updates & Improvements

### Training Standardization (SSTrans-aligned)
- **Class Weights**: Dynamic computation from pixel frequency (replaces manual weights)
- **Loss Function**: Unified 0.3 * CE + 0.4 * Focal + 0.3 * Dice with proper class weighting
- **Optimizer**: AdamW with increased weight_decay=0.05 for better regularization
- **Scheduler**: CosineAnnealingWarmRestarts for improved transformer convergence
- **Early Stopping**: Increased patience to 50 epochs for better convergence

### Hybrid2 Enhancements
- **CBAM Attention**: Channel and spatial attention mechanisms for better feature focus
- **Smart Skip Connections**: Attention-based feature fusion instead of simple concatenation
- **Deep Decoder Blocks**: Multi-layer convolutions with attention for better reconstruction
- **Feature Refinement**: Gradual channel reduction with residual connections

### Testing Improvements
- **Test-Time Augmentation (TTA)**: Enabled by default for all run scripts (+2-4% mIoU improvement)
- **Enhanced Inference**: Better handling of edge cases and rare classes
- **Consistent Evaluation**: All datasets now use the same TTA pipeline

## Notes

- Both models use SSTrans-aligned training approach with dynamic class weights
- Hybrid2 supports different EfficientNet variants (B0, B4, B5) for the decoder
- All models support both U-DIADS-Bib and DIVAHISDB datasets
- TTA is enabled by default in all run scripts for improved accuracy
- Results are saved with model-specific naming to avoid conflicts
- Trainer architecture matches SSTrans for consistency across models

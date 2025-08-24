
# How This Repository Works: A Complete Workflow

This repository implements a medical image segmentation pipeline using tra3. **Quantitative Result## 6. Practical Workflow

To run the complete pipeline for a specific manuscript:

1. **Generate Patches**:
   ```bash
   python3 Sliding_window_generate_dataset.py
   ```
   This creates patches for all manuscripts in the `U-DIADS-Bib-MS_patched/` directory.

2. **Train the Model**:
   ```bash
   sbatch training.sh
   ```
   This runs the training for Latin14396 manuscript using the patched data.

3. **Test the Model**:
   ```bash
   sbatch testing.sh
   ```
   This tests the model on Latin14396 manuscript and generates segmentation outputs.the test output file
   - Per-class metrics (Precision, Recall, F1, IoU)
   - Mean metrics across all classes

## 6. Practical Workflow

To run the complete pipeline for a specific manuscript:

1. **Generate Patches**:
   ```bash
   python3 Sliding_window_generate_dataset.py
   ```
   This creates patches for all manuscripts in the `U-DIADS-Bib-MS_patched/` directory.

2. **Train the Model**:
   ```bash
   sbatch training.sh
   ```
   This runs the training for Latin14396 manuscript using the patched data.ls (Swin-Unet and MISSFormer) for the U-DIADS-Bib manuscript dataset. Below is a complete explanation of the workflow from data preparation to model output generation.

## 1. Overall Architecture

The repository contains two main models:

- **Swin-Unet**: A transformer-based U-Net that uses Swin Transformer blocks for feature extraction, combining the strengths of transformers for global context with the U-Net architecture for preserving spatial information.

- **MISSFormer**: A mixed supervision transformer that combines self-supervision and cross-supervision mechanisms.

## 2. Data Flow

### Step 1: Dataset Organization
- The original dataset (`U-DIADS-Bib-MS/`) contains four manuscripts: Latin2, Latin14396, Latin16746, and Syr341
- Each manuscript has images (`img-{manuscript}/`) and pixel-level annotations (`pixel-level-gt-{manuscript}/`)
- Data is split into training/validation/test directories

### Step 2: Patch Generation
- The `Sliding_window_generate_dataset.py` script divides the large manuscript images into 224×224 pixel patches
- This is necessary because:
  1. The Swin-Unet model requires a fixed input size (224×224)
  2. Full manuscript images are too large to process directly
  3. Patching increases the number of training samples
- Patches are stored in `U-DIADS-Bib-MS_patched/` with a similar structure to the original dataset

### Step 3: Training Pipeline

1. **Dataset Loading** (`datasets/dataset_udiadsbib.py`):
   - The `UDiadsBibDataset` class loads either:
     - Original full images (cropping patches on-the-fly)
     - Pre-generated patches from `U-DIADS-Bib-MS_patched/`
   - RGB annotations are converted to class indices using the `rgb_to_class` function
   - Data augmentation is applied during training (flips, rotations, color jitter)

2. **Model Initialization** (`train.py`):
   - Loads the Swin-Unet architecture from `networks/vision_transformer.py`
   - Initializes with pre-trained weights from `pretrained_ckpt/swin_tiny_patch4_window7_224.pth`
   - Model architecture consists of:
     - Patch embedding layer
     - Swin Transformer encoder blocks (for downsampling)
     - Swin Transformer decoder blocks (for upsampling)
     - Skip connections between encoder and decoder
     - Final segmentation head

3. **Training Process** (`trainer.py`):
   - Uses cross-entropy loss for segmentation
   - Adam optimizer with initial learning rate of 0.01
   - Training occurs for a specified number of epochs (default: 300)
   - Best models are saved based on validation performance
## 3. Model Architecture Deep Dive

### Swin-Unet Architecture:

1. **Patch Embedding**:
   - Input images are divided into non-overlapping patches
   - Each patch is embedded into a high-dimensional feature space

2. **Swin Transformer Encoder**:
   - Uses shifted window self-attention mechanism
   - Captures both local and global context efficiently
   - Gradually reduces spatial resolution while increasing feature channels

3. **Swin Transformer Decoder**:
   - Upsamples feature maps
   - Restores spatial resolution for pixel-level segmentation

4. **Skip Connections**:
   - Connect corresponding encoder and decoder layers
   - Help preserve fine-grained spatial details

5. **Segmentation Head**:
   - Final convolution layer that maps features to class probabilities
   - Outputs a segmentation map with 6 classes:
     - Background (0)
     - Paratext (1) - Yellow in GT
     - Decoration (2) - Cyan in GT
     - Main Text (3) - Magenta in GT
     - Title (4) - Red in GT
     - Chapter Headings (5) - Lime in GT

## 4. Testing/Inference Pipeline

### Step 1: Model Loading (`test.py`):
- Loads the trained model from saved checkpoints
- Automatically finds the best model based on the highest epoch number

### Step 2: Inference with Patched Data:
- For each manuscript in the test set:
  1. Processes each 224×224 patch through the model
  2. Gets a segmentation prediction for each patch
  3. Records the patch position to reconstruct the full image later

### Step 3: Image Reconstruction:
- The test.py script contains logic to:
  1. Group patches by their original image
  2. Determine the original image dimensions
  3. Place each predicted patch at its correct position
  4. Combine overlapping predictions (if any) by averaging

### Step 4: Evaluation and Visualization:
- Computes metrics between predicted segmentation and ground truth:
  - Precision, Recall, F1 score, and IoU for each class
  - Mean metrics across all classes
- Generates visualization images:
  - Original input image
  - Predicted segmentation mask (with class colors)
  - Ground truth segmentation mask
  - Saved to `predictions/compare/` directory

## 5. Output Generation

The final outputs from the testing pipeline include:

1. **Segmentation Masks**:
   - Stored in `model_out/udiadsbib_patch224_swinunet_Latin14396/predictions/result/`
   - Full-size reconstructed predictions in the original RGB color scheme
   - Each pixel is colored according to its predicted class

2. **Comparison Visualizations**:
   - Stored in `model_out/udiadsbib_patch224_swinunet_Latin14396/predictions/compare/`
   - 3-panel images showing original, prediction, and ground truth
   - Useful for qualitative evaluation
Quantitative Results:

Logged in the test output file
Per-class metrics (Precision, Recall, F1, IoU)
Mean metrics across all classes
6. Practical Workflow
To run the complete pipeline for a specific manuscript:

Generate Patches:


python3 Sliding_window_generate_dataset.py
This creates patches for all manuscripts in the U-DIADS-Bib-MS_patched/ directory.

Train the Model:


sbatch training.sh
This runs the training for Latin14396 manuscript using the patched data.

Test the Model:


sbatch testing.sh
This tests the model on Latin14396 manuscript and generates segmentation outputs.

## 7. Key Features and Benefits

1. **Patch-based Processing**:
   - Enables processing of large images with fixed-size model
   - Increases training data through patch extraction
   - Allows model to focus on local details

2. **Transformer Architecture**:
   - Better at capturing long-range dependencies than CNNs
   - Handles complex manuscript layouts effectively
   - Preserves fine details important for manuscript analysis

3. **Manuscript-specific Training**:
   - Each manuscript is processed individually
   - Allows the model to adapt to the specific style of each manuscript
   - Results in better segmentation quality

4. **Comprehensive Evaluation**:
   - Multiple metrics to assess segmentation quality
   - Visual comparisons for qualitative assessment
   - Per-class performance analysis

This repository effectively implements a state-of-the-art approach to manuscript segmentation, leveraging the latest advances in transformer-based computer vision to accurately segment different elements in historical manuscripts.


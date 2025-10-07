# Hyperspectral Plastic Classification Pipeline

Complete pipeline for training and deploying 1D CNN models for pixel-wise plastic classification from hyperspectral imaging data.

## Overview

This pipeline processes hyperspectral imagery (458 bands, 450-850nm) to classify 11 types of plastic materials using pixel-wise 1D CNNs.

### Plastic Classes
1. Background
2. 95PU
3. HIPS
4. HVDF-HFP
5. GPSS
6. PU
7. 75PU
8. 85PU
9. PETE
10. PET
11. PMMA

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Raw Training Data (training_dataset/)                      │
│         │                                                    │
│         ▼                                                    │
│  [1] Normalization (pipeline_normalize.py)                  │
│         │  • Percentile-based (2%-98%)                       │
│         │  • Band-wise stretching to 0-255                   │
│         ▼                                                    │
│  Normalized Data (training_dataset_normalized/)             │
│         │                                                    │
│         ▼                                                    │
│  [2] Preprocessing (pipeline_preprocess.py)                 │
│         │  • Wavelength filtering (optional)                 │
│         │  • Spectral binning/selection                      │
│         │  • Spatial binning (optional)                      │
│         │  • Denoising (optional)                            │
│         ▼                                                    │
│  Preprocessed Cube                                           │
│         │                                                    │
│         ▼                                                    │
│  [3] Dataset Creation (pipeline_dataset.py)                 │
│         │  • Load ground truth labels                        │
│         │  • Extract pixel spectra                           │
│         │  • Train/val split (80/20)                         │
│         ▼                                                    │
│  DataLoaders                                                 │
│         │                                                    │
│         ▼                                                    │
│  [4] Model Training (pipeline_train.py)                     │
│         │  • 1D CNN (standard or residual)                   │
│         │  • Cross-entropy loss                              │
│         │  • Adam optimizer                                  │
│         │  • Learning rate scheduling                        │
│         ▼                                                    │
│  Trained Model (output/training/best_model.pth)             │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  INFERENCE PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Raw Inference Data (Inference_dataset1/)                   │
│         │                                                    │
│         ▼                                                    │
│  [1] Normalization                                           │
│         │                                                    │
│         ▼                                                    │
│  [2] Preprocessing (same as training)                       │
│         │                                                    │
│         ▼                                                    │
│  [3] Inference (pipeline_inference.py)                      │
│         │  • Load trained model                              │
│         │  • Pixel-wise prediction                           │
│         │  • Generate probability maps                       │
│         ▼                                                    │
│  Results (output/inference/)                                 │
│    • predictions.png                                         │
│    • probability_maps/                                       │
│    • inference_statistics.json                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Requirements
```bash
pip install torch torchvision numpy pillow scipy tqdm matplotlib
```

### Verify Installation
```bash
python -c "import torch; import numpy; import PIL; print('All dependencies installed!')"
```

## Quick Start

### Run Complete Pipeline
```bash
python run_pipeline.py --mode full
```

This will:
1. Normalize training and inference data
2. Train the model
3. Run inference
4. Generate visualizations and statistics

### Run Individual Steps

**Normalize data only:**
```bash
python run_pipeline.py --mode normalize
```

**Train model only:**
```bash
python run_pipeline.py --mode train
```

**Inference only:**
```bash
python run_pipeline.py --mode inference
```

## Module Documentation

### 1. `pipeline_normalize.py`
Performs percentile-based normalization on raw hyperspectral bands.

**Key Function:**
```python
normalize_dataset(
    input_folder='training_dataset',
    output_folder='training_dataset_normalized',
    lower_percentile=2,
    upper_percentile=98
)
```

**Output:**
- Normalized bands as PNG files
- `normalization_metadata.json` with per-band statistics

### 2. `pipeline_preprocess.py`
Applies various preprocessing operations to hyperspectral cubes.

**Configuration Options:**
```python
PREPROCESS_CONFIG = {
    'wavelength_range': (450, 700),  # Filter to specific wavelengths
    'select_n_bands': 100,            # Select N evenly-spaced bands
    'spectral_binning': 2,            # Average consecutive bands
    'spatial_binning': 2,             # Downsample spatial resolution
    'denoise_enabled': True,
    'denoise_method': 'gaussian',
    'denoise_strength': 1.0
}
```

**Key Functions:**
- `load_hyperspectral_cube()` - Load cube from folder
- `filter_wavelength_range()` - Filter by wavelength
- `bin_spectral_bands()` - Reduce spectral dimensions
- `bin_spatial()` - Reduce spatial dimensions
- `denoise_cube()` - Apply denoising filters

### 3. `pipeline_dataset.py`
PyTorch dataset for pixel-wise classification.

**Features:**
- Extracts spectral signatures from each pixel
- Maps RGB labels to class indices
- Handles class imbalance with weighted sampling
- Train/validation splitting
- Data augmentation (optional noise injection)

**Class Mapping:**
```python
CLASS_MAPPING = {
    (0,0,0): 0,        # Background
    (255,0,0): 1,      # 95PU
    (0,0,255): 2,      # HIPS
    # ... etc
}
```

### 4. `pipeline_model.py`
1D CNN architectures for hyperspectral classification.

**Model Options:**

**Standard CNN:**
- 4 convolutional blocks (64, 128, 256, 256 channels)
- Batch normalization
- Dropout regularization
- Global average pooling
- 2 fully connected layers

**Residual CNN:**
- Residual connections for better gradient flow
- 3 residual blocks
- Suitable for deeper networks

**Usage:**
```python
model = create_model(
    num_bands=229,
    num_classes=11,
    model_type='resnet',  # or 'cnn'
    dropout_rate=0.3
)
```

### 5. `pipeline_train.py`
Complete training pipeline with monitoring and checkpointing.

**Configuration:**
```python
CONFIG = {
    'num_epochs': 50,
    'learning_rate': 0.001,
    'batch_size': 512,
    'model_type': 'resnet',
    'use_class_weights': True,  # Handle imbalanced data
    'output_dir': 'output/training'
}
```

**Features:**
- Automatic train/val split
- Class-weighted loss for imbalanced data
- Learning rate scheduling
- Early stopping
- Training history plots
- Model checkpointing (best + latest)

**Outputs:**
- `best_model.pth` - Best model checkpoint
- `latest_model.pth` - Latest model checkpoint
- `training_history.json` - Training metrics
- `training_history.png` - Loss and accuracy plots

### 6. `pipeline_inference.py`
Inference on new hyperspectral data.

**Features:**
- Batch-wise pixel prediction
- Probability map generation
- Class distribution statistics
- RGB visualization of predictions

**Outputs:**
- `predictions.png` - RGB-coded prediction map
- `probability_maps/` - Per-class probability maps
- `inference_statistics.json` - Pixel counts and confidence scores

### 7. `run_pipeline.py`
Master orchestration script for the entire pipeline.

**Modes:**
- `full` - Complete pipeline (normalize → train → inference)
- `normalize` - Data normalization only
- `train` - Model training only
- `inference` - Inference only

**Features:**
- Dependency checking
- Data folder validation
- Error handling
- Progress reporting

## Configuration Guide

### Preprocessing Configuration

The preprocessing configuration **must be identical** between training and inference. The default configuration uses spectral binning to reduce from 458 to ~229 bands:

```python
PREPROCESS_CONFIG = {
    'wavelength_range': None,      # No wavelength filtering
    'select_n_bands': None,        # No band selection
    'spectral_binning': 2,         # Average 2 consecutive bands
    'spatial_binning': None,       # No spatial downsampling
    'denoise_enabled': False       # No denoising
}
```

### Training Configuration

Key hyperparameters to tune:

```python
# Model architecture
'model_type': 'resnet'      # 'cnn' or 'resnet'
'dropout_rate': 0.3         # 0.2-0.5 typical range

# Training
'num_epochs': 50            # Increase if underfitting
'learning_rate': 0.001      # Decrease if loss unstable
'batch_size': 512           # Adjust based on memory

# Data
'ignore_background': True   # Exclude background pixels
'augment': True             # Add noise augmentation
'use_class_weights': True   # Balance class influence
```

## Expected Performance

### Training Metrics
- **Training time:** ~10-30 minutes on GPU (depending on data size)
- **Expected accuracy:** 85-95% validation accuracy for well-separated classes
- **Memory usage:** ~2-4 GB GPU memory with batch_size=512

### Inference Metrics
- **Inference time:** ~1-5 minutes on GPU for full-resolution image
- **Output size:** Same as input resolution (e.g., 5496 x 3672 pixels)

## Troubleshooting

### Out of Memory Errors
```python
# Reduce batch size in training config
CONFIG['batch_size'] = 256  # or 128

# Reduce spatial resolution with spatial binning
PREPROCESS_CONFIG['spatial_binning'] = 2  # or 4
```

### Low Accuracy
```python
# Try residual model
CONFIG['model_type'] = 'resnet'

# Increase training epochs
CONFIG['num_epochs'] = 100

# Use class weights for imbalanced data
CONFIG['use_class_weights'] = True

# Add denoising
PREPROCESS_CONFIG['denoise_enabled'] = True
```

### Slow Training
```python
# Reduce spectral dimensions
PREPROCESS_CONFIG['spectral_binning'] = 4  # More aggressive binning

# Or select fewer bands
PREPROCESS_CONFIG['select_n_bands'] = 50

# Reduce spatial resolution
PREPROCESS_CONFIG['spatial_binning'] = 2
```

## File Structure

```
plastic-type-classification/
├── training_dataset/              # Raw training data (458 bands)
│   ├── ImagesStack1.png
│   ├── ImagesStack2.png
│   ├── ...
│   └── header.json
├── Ground_Truth/                  # Training labels
│   ├── labels.png
│   └── labels.json
├── Inference_dataset1/            # Raw inference data
│   ├── ImagesStack1.png
│   ├── ...
│   └── header.json
├── training_dataset_normalized/   # Normalized training data
├── Inference_dataset1_normalized/ # Normalized inference data
├── output/
│   ├── training/                  # Training outputs
│   │   ├── best_model.pth
│   │   ├── latest_model.pth
│   │   ├── training_history.json
│   │   └── training_history.png
│   └── inference/                 # Inference outputs
│       ├── predictions.png
│       ├── probability_maps/
│       └── inference_statistics.json
├── pipeline_normalize.py          # Step 1: Normalization
├── pipeline_preprocess.py         # Step 2: Preprocessing
├── pipeline_dataset.py            # Step 3: Dataset
├── pipeline_model.py              # Step 4: Model
├── pipeline_train.py              # Step 5: Training
├── pipeline_inference.py          # Step 6: Inference
├── run_pipeline.py                # Main orchestrator
└── README_PIPELINE.md             # This file
```

## Citation

If you use this pipeline, please cite:
```
Hyperspectral Plastic Classification Pipeline
1D CNN for pixel-wise plastic type identification
```

## License

[Your License Here]

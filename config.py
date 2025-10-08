"""
Configuration file for hyperspectral plastic classification pipeline

Modify parameters here to customize the pipeline behavior.
"""

# =====================================================================
# DATA PATHS
# =====================================================================

# Input folders
TRAINING_DATA_FOLDER = 'training_dataset'
INFERENCE_DATA_FOLDER = 'Inference_dataset1'
GROUND_TRUTH_FOLDER = 'Ground_Truth'

# Output folders
NORMALIZED_TRAINING_FOLDER = 'training_dataset_normalized'
NORMALIZED_INFERENCE_FOLDER = 'Inference_dataset1_normalized'
OUTPUT_DIR = 'output'

# =====================================================================
# NORMALIZATION PARAMETERS
# =====================================================================

NORMALIZATION = {
    'lower_percentile': 2,    # Lower percentile for clipping (0-100)
    'upper_percentile': 98,   # Upper percentile for clipping (0-100)
}

# =====================================================================
# PREPROCESSING PARAMETERS
# =====================================================================

PREPROCESSING = {
    # Wavelength filtering
    'wavelength_range': None,      # e.g., (450, 700) or None to disable

    # Band selection/reduction (mutually exclusive)
    'select_n_bands': None,        # e.g., 100 to select 100 evenly-spaced bands, or None
    'spectral_binning': 2,         # e.g., 2 to average 2 consecutive bands, or None

    # Spatial downsampling
    'spatial_binning': None,       # e.g., 2 for 2x2 blocks, or None

    # Denoising
    'denoise_enabled': False,      # True to enable denoising
    'denoise_method': 'gaussian',  # 'gaussian' or 'median'
    'denoise_strength': 1.0,       # Strength parameter (sigma for gaussian, size for median)
}

# =====================================================================
# DATASET PARAMETERS
# =====================================================================

DATASET = {
    'augment': True,               # Apply data augmentation (noise injection)
    'val_ratio': 0.2,              # Validation set ratio (0.0-1.0)
    'batch_size': 512,             # Batch size for training
    'num_workers': 4,              # Number of workers for data loading
}

# =====================================================================
# MODEL PARAMETERS
# =====================================================================

MODEL = {
    'model_type': 'resnet',        # 'cnn' or 'resnet'
    'dropout_rate': 0.3,           # Dropout rate (0.0-1.0)
}

# =====================================================================
# TRAINING PARAMETERS
# =====================================================================

TRAINING = {
    'num_epochs': 50,              # Number of training epochs
    'learning_rate': 0.001,        # Initial learning rate
    'weight_decay': 1e-4,          # L2 regularization weight
    'use_class_weights': True,     # Use class weights to handle imbalanced data
}

# =====================================================================
# INFERENCE PARAMETERS
# =====================================================================

INFERENCE = {
    'checkpoint_path': 'output/training/best_model.pth',  # Path to trained model
    'batch_size': 512,                                     # Batch size for inference
    'save_probability_maps': True,                         # Save per-class probability maps
}

# =====================================================================
# CLASS INFORMATION (DO NOT MODIFY)
# =====================================================================

CLASS_MAPPING = {
    (0, 0, 0): 0,        # Background
    (255, 0, 0): 1,      # 95PU
    (0, 0, 255): 2,      # HIPS
    (255, 125, 125): 3,  # HVDF-HFP
    (255, 255, 0): 4,    # GPSS
    (0, 125, 125): 5,    # PU
    (0, 200, 255): 6,    # 75PU
    (255, 0, 255): 7,    # 85PU
    (0, 255, 0): 8,      # PETE
    (255, 125, 0): 9,    # PET
    (255, 0, 100): 10    # PMMA
}

CLASS_NAMES = [
    'Background', '95PU', 'HIPS', 'HVDF-HFP', 'GPSS',
    'PU', '75PU', '85PU', 'PETE', 'PET', 'PMMA'
]

NUM_CLASSES = 11

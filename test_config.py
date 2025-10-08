"""
Quick test to verify configuration parameters are properly applied
"""

import pipeline_train

# Override CONFIG
pipeline_train.CONFIG = {
    'data_folder': 'training_dataset_normalized',
    'label_path': 'Ground_Truth/labels.png',
    'preprocess': {
        'wavelength_range': None,
        'select_n_bands': None,
        'spectral_binning': 10,  # Should produce ~45 bands
        'spatial_binning': 4,    # Should reduce resolution by 16x
        'denoise_enabled': True,
        'denoise_method': 'gaussian',
        'denoise_strength': 3.0
    },
    'augment': True,
    'val_ratio': 0.2,
    'batch_size': 512,
    'num_workers': 4,
    'model_type': 'resnet',
    'dropout_rate': 0.3,
    'num_epochs': 2,  # Just 2 epochs for testing
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'use_class_weights': True,
    'output_dir': 'output/test_training'
}

print("Testing if CONFIG override works...")
print(f"Spectral binning: {pipeline_train.CONFIG['preprocess']['spectral_binning']}")
print(f"Spatial binning: {pipeline_train.CONFIG['preprocess']['spatial_binning']}")
print(f"Denoising: {pipeline_train.CONFIG['preprocess']['denoise_enabled']}")
print()

# This should show the preprocessing being applied
from pipeline_preprocess import load_hyperspectral_cube, preprocess_cube

print("Loading and preprocessing cube...")
cube, wavelengths, header = load_hyperspectral_cube('training_dataset_normalized')
print(f"Original cube shape: {cube.shape}")

cube, wavelengths = preprocess_cube(cube, wavelengths, pipeline_train.CONFIG['preprocess'])
print(f"After preprocessing shape: {cube.shape}")
print()

expected_bands = 458 // 10  # ~45
expected_height = 1176 // 4  # 294
expected_width = 2092 // 4   # 523

print(f"Expected: ({expected_height}, {expected_width}, {expected_bands})")
print(f"Actual:   {cube.shape}")

if cube.shape[2] == expected_bands:
    print("✓ Spectral binning working correctly!")
else:
    print(f"✗ Spectral binning NOT working (expected {expected_bands} bands, got {cube.shape[2]})")

if cube.shape[0] == expected_height and cube.shape[1] == expected_width:
    print("✓ Spatial binning working correctly!")
else:
    print(f"✗ Spatial binning NOT working (expected {expected_height}x{expected_width}, got {cube.shape[0]}x{cube.shape[1]})")

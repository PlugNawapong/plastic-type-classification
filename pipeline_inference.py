"""
Inference Pipeline
Perform inference on new hyperspectral data
"""

import os
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from pipeline_preprocess import load_hyperspectral_cube, preprocess_cube
from pipeline_dataset import CLASS_NAMES, CLASS_MAPPING
from pipeline_model import create_model


# Module-level CONFIG that can be overridden by run_pipeline_config.py
CONFIG = None

# Inverse class mapping (class_id -> RGB)
CLASS_TO_RGB = {v: k for k, v in CLASS_MAPPING.items()}


def load_trained_model(checkpoint_path, device='cpu'):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        model: Loaded PyTorch model
        config: Training configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Determine number of bands from preprocessing config
    # This should match the training preprocessing
    print("Loading model configuration...")
    print(f"Model type: {config['model_type']}")
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")

    return checkpoint, config


def perform_inference(model, cube, device='cpu', batch_size=512):
    """
    Perform pixel-wise inference on hyperspectral cube.

    Args:
        model: Trained PyTorch model
        cube: 3D numpy array (H x W x Bands)
        device: Device for inference
        batch_size: Batch size for inference

    Returns:
        predictions: 2D numpy array (H x W) with class predictions
        probabilities: 3D numpy array (H x W x num_classes) with class probabilities
    """
    model.eval()
    height, width, num_bands = cube.shape

    # Reshape cube to (num_pixels, num_bands)
    pixels = cube.reshape(-1, num_bands).astype(np.float32) / 255.0

    # Perform inference in batches
    predictions = np.zeros(height * width, dtype=np.int64)
    probabilities = np.zeros((height * width, 11), dtype=np.float32)

    with torch.no_grad():
        for i in tqdm(range(0, len(pixels), batch_size), desc='Inference'):
            batch = pixels[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).to(device)

            # Forward pass
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)

            # Store predictions
            _, preds = torch.max(outputs, 1)
            predictions[i:i+batch_size] = preds.cpu().numpy()
            probabilities[i:i+batch_size] = probs.cpu().numpy()

    # Reshape to image dimensions
    predictions = predictions.reshape(height, width)
    probabilities = probabilities.reshape(height, width, 11)

    return predictions, probabilities


def save_prediction_image(predictions, output_path):
    """
    Save prediction as RGB image using class color mapping.

    Args:
        predictions: 2D array (H x W) with class indices
        output_path: Path to save image
    """
    height, width = predictions.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, rgb in CLASS_TO_RGB.items():
        mask = predictions == class_id
        rgb_image[mask] = rgb

    Image.fromarray(rgb_image).save(output_path)


def save_class_probability_maps(probabilities, output_dir):
    """
    Save individual probability maps for each class.

    Args:
        probabilities: 3D array (H x W x num_classes)
        output_dir: Directory to save probability maps
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_id, class_name in enumerate(CLASS_NAMES):
        prob_map = (probabilities[:, :, class_id] * 255).astype(np.uint8)
        Image.fromarray(prob_map).save(output_dir / f'prob_class{class_id}_{class_name}.png')


def calculate_class_statistics(predictions, probabilities):
    """
    Calculate statistics about predicted classes.

    Args:
        predictions: 2D array (H x W) with class indices
        probabilities: 3D array (H x W x num_classes)

    Returns:
        stats: Dictionary with statistics
    """
    total_pixels = predictions.size
    unique, counts = np.unique(predictions, return_counts=True)

    stats = {
        'total_pixels': int(total_pixels),
        'class_distribution': {}
    }

    for class_id, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        mean_confidence = probabilities[:, :, class_id][predictions == class_id].mean()

        stats['class_distribution'][int(class_id)] = {
            'class_name': CLASS_NAMES[class_id],
            'pixel_count': int(count),
            'percentage': float(percentage),
            'mean_confidence': float(mean_confidence)
        }

    return stats


def main():
    """Main inference pipeline."""

    global CONFIG

    # =====================================================================
    # CONFIGURATION
    # =====================================================================

    # Use module-level CONFIG if set (by run_pipeline_config.py), otherwise use defaults
    if CONFIG is None:
        CONFIG = {
            # Model
            'checkpoint_path': 'output/training/best_model.pth',

            # Inference data
            'inference_folder': 'Inference_dataset1',

            # Preprocessing (must match training!)
            'preprocess': {
                'wavelength_range': None,
                'select_n_bands': None,
                'spectral_binning': 2,  # Must match training
                'spatial_binning': None,
                'denoise_enabled': False,
                'denoise_method': 'gaussian',
                'denoise_strength': 1.0
            },

            # Output
            'output_dir': 'output/inference',
            'save_probability_maps': True,
            'batch_size': 512
        }

    # =====================================================================
    # SETUP
    # =====================================================================

    print("="*60)
    print("HYPERSPECTRAL PLASTIC CLASSIFICATION - INFERENCE PIPELINE")
    print("="*60)

    # Device (Apple Silicon M4 GPU support via MPS)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"\n✓ Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"\n⚠ Using CPU (GPU not available)")

    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # LOAD MODEL
    # =====================================================================

    print("\n[1/5] Loading trained model...")
    checkpoint, train_config = load_trained_model(CONFIG['checkpoint_path'], device)

    # Create model with same architecture
    # Get number of bands from checkpoint
    model_state = checkpoint['model_state_dict']
    # Extract num_bands from first conv layer weight shape
    num_bands = None
    for key in model_state.keys():
        if 'conv' in key and 'weight' in key:
            # Shape is (out_channels, in_channels, kernel_size)
            # For 1D conv on spectral data, we need to infer from the network
            break

    # Load inference data first to get correct dimensions
    print("\n[2/5] Loading inference cube...")
    cube, wavelengths, header = load_hyperspectral_cube(CONFIG['inference_folder'])

    print("\n[3/5] Preprocessing cube...")
    cube, wavelengths = preprocess_cube(cube, wavelengths, CONFIG['preprocess'])

    # Now create model with correct num_bands
    num_bands = cube.shape[2]
    print(f"\n[4/5] Creating model with {num_bands} bands...")

    model = create_model(
        num_bands=num_bands,
        num_classes=11,
        model_type=train_config['model_type'],
        dropout_rate=train_config['dropout_rate']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"Training accuracy: {checkpoint['best_val_acc']:.2f}%")

    # =====================================================================
    # PERFORM INFERENCE
    # =====================================================================

    print("\n[5/5] Performing inference...")
    predictions, probabilities = perform_inference(
        model, cube, device, batch_size=CONFIG['batch_size']
    )

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================

    print("\nSaving results...")

    # Save prediction image
    pred_image_path = output_dir / 'predictions.png'
    save_prediction_image(predictions, pred_image_path)
    print(f"✓ Prediction image saved: {pred_image_path}")

    # Save probability maps
    if CONFIG['save_probability_maps']:
        prob_dir = output_dir / 'probability_maps'
        save_class_probability_maps(probabilities, prob_dir)
        print(f"✓ Probability maps saved: {prob_dir}")

    # Calculate and save statistics
    stats = calculate_class_statistics(predictions, probabilities)
    stats_path = output_dir / 'inference_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics saved: {stats_path}")

    # Print statistics
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    print(f"\nTotal pixels: {stats['total_pixels']:,}")
    print("\nClass distribution:")
    for class_id, info in sorted(stats['class_distribution'].items()):
        print(f"  {info['class_name']:12s}: {info['pixel_count']:8,} pixels "
              f"({info['percentage']:5.2f}%) - Confidence: {info['mean_confidence']:.3f}")

    print("\n" + "="*60)
    print("Inference completed successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

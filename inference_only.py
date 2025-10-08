"""
Quick Inference Script for Local Mac M4
========================================

Run inference on already-normalized data with a trained model.

Usage:
    python inference_only.py

Requirements:
    - Normalized inference data in a folder (e.g., Inference_dataset1/)
    - Trained model file (e.g., best_model.pth)
    - Ground truth labels for preprocessing info (optional, for spatial binning)

The script will:
    1. Load the trained model
    2. Load normalized inference data
    3. Run inference on all pixels
    4. Save predictions as PNG with color mapping
    5. Generate statistics and probability maps
"""

import os
# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm

# Import from pipeline modules
from pipeline_preprocess import load_hyperspectral_cube, preprocess_cube
from pipeline_model import create_model
from config import CLASS_MAPPING, CLASS_NAMES, NUM_CLASSES


# ============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================================================

# Paths (edit these to match your setup)
INFERENCE_FOLDER = 'Inference_dataset1'  # Your normalized inference data folder
MODEL_PATH = 'best_model.pth'            # Path to your trained model
OUTPUT_DIR = 'inference_output'          # Where to save results

# Preprocessing settings (must match training settings!)
PREPROCESSING = {
    'wavelength_range': None,      # e.g., (450, 700) or None
    'select_n_bands': None,        # e.g., 100 or None
    'spectral_binning': 2,         # Must match training! (2, 5, 10, or None)
    'spatial_binning': None,       # Must match training! (2, 4, 8, or None)
    'denoise_enabled': False,      # Must match training!
    'denoise_method': 'gaussian',
    'denoise_strength': 1.0,
}

# Model settings (must match training settings!)
MODEL_TYPE = 'resnet'  # 'cnn', 'resnet', 'deep', 'inception', 'lstm', 'transformer'
DROPOUT_RATE = 0.3

# Inference settings
BATCH_SIZE = 1024  # Larger batch = faster inference
DEVICE = 'mps'     # 'mps' for M4, 'cuda' for GPU, 'cpu' for CPU


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_device():
    """Setup device (MPS for M4, CUDA for GPU, or CPU)."""
    if DEVICE == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✓ Using Apple Silicon GPU (MPS)")
    elif DEVICE == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"✓ Using CPU")
    return device


def load_model(model_path, num_bands, device):
    """Load trained model from checkpoint."""
    print(f"\nLoading model from: {model_path}")

    # Create model architecture
    model = create_model(num_bands, NUM_CLASSES, MODEL_TYPE, DROPOUT_RATE)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
        print(f"  Training accuracy: {checkpoint.get('val_acc', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ Model weights loaded")

    model.to(device)
    model.eval()

    return model


def create_rgb_from_labels(predictions, class_mapping, class_names):
    """Convert class predictions to RGB image."""
    height, width = predictions.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Create reverse mapping (class_id -> RGB)
    class_to_rgb = {}
    for rgb, class_id in class_mapping.items():
        class_to_rgb[class_id] = rgb

    # Map each pixel
    for class_id in range(NUM_CLASSES):
        if class_id in class_to_rgb:
            mask = predictions == class_id
            rgb_image[mask] = class_to_rgb[class_id]

    return rgb_image


def run_inference_batch(model, cube, device, batch_size=1024):
    """Run inference on entire cube in batches."""
    height, width, num_bands = cube.shape
    total_pixels = height * width

    # Reshape cube to (num_pixels, num_bands)
    cube_flat = cube.reshape(-1, num_bands).astype(np.float32)

    # Prepare output arrays
    predictions = np.zeros(total_pixels, dtype=np.int64)
    probabilities = np.zeros((total_pixels, NUM_CLASSES), dtype=np.float32)

    print(f"\nRunning inference on {total_pixels:,} pixels...")

    with torch.no_grad():
        num_batches = (total_pixels + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Inference"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_pixels)

            # Get batch
            batch = torch.from_numpy(cube_flat[start_idx:end_idx]).float().to(device)

            # Forward pass
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Store results
            predictions[start_idx:end_idx] = preds.cpu().numpy()
            probabilities[start_idx:end_idx] = probs.cpu().numpy()

    # Reshape to original dimensions
    predictions = predictions.reshape(height, width)
    probabilities = probabilities.reshape(height, width, NUM_CLASSES)

    return predictions, probabilities


def save_results(predictions, probabilities, output_dir):
    """Save predictions as RGB image and statistics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Save RGB prediction image
    print("\nSaving results...")
    rgb_image = create_rgb_from_labels(predictions, CLASS_MAPPING, CLASS_NAMES)
    rgb_pil = Image.fromarray(rgb_image)
    rgb_pil.save(output_path / 'predictions.png')
    print(f"✓ Saved: {output_path / 'predictions.png'}")

    # 2. Save raw predictions (numpy array)
    np.save(output_path / 'predictions.npy', predictions)
    print(f"✓ Saved: {output_path / 'predictions.npy'}")

    # 3. Generate and save statistics
    total_pixels = predictions.size
    class_distribution = {}

    for class_id in range(NUM_CLASSES):
        mask = predictions == class_id
        pixel_count = np.sum(mask)

        if pixel_count > 0:
            mean_conf = np.mean(probabilities[mask, class_id])
            percentage = (pixel_count / total_pixels) * 100

            class_distribution[str(class_id)] = {
                'class_id': int(class_id),
                'class_name': CLASS_NAMES[class_id],
                'pixel_count': int(pixel_count),
                'percentage': float(percentage),
                'mean_confidence': float(mean_conf)
            }

    stats = {
        'total_pixels': int(total_pixels),
        'image_shape': predictions.shape,
        'num_classes': NUM_CLASSES,
        'class_distribution': class_distribution
    }

    with open(output_path / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved: {output_path / 'statistics.json'}")

    # 4. Save probability maps for each class (optional)
    prob_dir = output_path / 'probability_maps'
    prob_dir.mkdir(exist_ok=True)

    for class_id in range(NUM_CLASSES):
        prob_map = (probabilities[:, :, class_id] * 255).astype(np.uint8)
        prob_img = Image.fromarray(prob_map)
        prob_img.save(prob_dir / f'class_{class_id}_{CLASS_NAMES[class_id]}.png')

    print(f"✓ Saved probability maps: {prob_dir}/")

    # 5. Print summary statistics
    print("\n" + "="*60)
    print("INFERENCE STATISTICS")
    print("="*60)
    print(f"Total pixels: {total_pixels:,}\n")
    print(f"{'Class':<15} {'Pixels':>12} {'Percentage':>12} {'Confidence':>12}")
    print("-"*60)

    for class_id in sorted(class_distribution.keys(), key=int):
        info = class_distribution[class_id]
        print(f"{info['class_name']:<15} {info['pixel_count']:>12,} "
              f"{info['percentage']:>11.2f}% {info['mean_confidence']:>11.3f}")

    print("="*60)


# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

def main():
    """Main inference pipeline."""
    print("="*60)
    print("HYPERSPECTRAL INFERENCE - MAC M4 OPTIMIZED")
    print("="*60)

    # 1. Setup device
    device = setup_device()

    # 2. Check paths exist
    print(f"\nChecking paths...")
    if not Path(INFERENCE_FOLDER).exists():
        print(f"❌ Error: Inference folder not found: {INFERENCE_FOLDER}")
        print(f"   Please update INFERENCE_FOLDER path in the script.")
        return

    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model file not found: {MODEL_PATH}")
        print(f"   Please update MODEL_PATH path in the script.")
        return

    print(f"✓ Inference folder: {INFERENCE_FOLDER}")
    print(f"✓ Model file: {MODEL_PATH}")

    # 3. Load inference data
    print(f"\n[1/4] Loading inference data...")
    cube = load_hyperspectral_cube(INFERENCE_FOLDER)
    print(f"✓ Loaded cube: {cube.shape} (H x W x Bands)")

    # 4. Preprocess (if needed)
    print(f"\n[2/4] Preprocessing...")
    original_shape = cube.shape
    cube = preprocess_cube(cube, PREPROCESSING)
    print(f"✓ Preprocessed cube: {cube.shape}")

    if cube.shape != original_shape:
        print(f"  Note: Shape changed from {original_shape} to {cube.shape}")

    num_bands = cube.shape[2]

    # 5. Load model
    print(f"\n[3/4] Loading model...")
    model = load_model(MODEL_PATH, num_bands, device)
    print(f"✓ Model ready: {MODEL_TYPE} with {num_bands} input bands")

    # 6. Run inference
    print(f"\n[4/4] Running inference...")
    predictions, probabilities = run_inference_batch(
        model, cube, device, batch_size=BATCH_SIZE
    )
    print(f"✓ Inference complete!")

    # 7. Save results
    save_results(predictions, probabilities, OUTPUT_DIR)

    print("\n" + "="*60)
    print("✅ INFERENCE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Results saved to: {OUTPUT_DIR}/")
    print(f"  - predictions.png      (RGB visualization)")
    print(f"  - predictions.npy      (raw predictions)")
    print(f"  - statistics.json      (class statistics)")
    print(f"  - probability_maps/    (per-class confidence maps)")
    print("="*60)


if __name__ == '__main__':
    main()

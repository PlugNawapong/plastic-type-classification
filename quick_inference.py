#!/usr/bin/env python3
"""
Quick Inference Script
Simple command-line interface for running inference
Usage: python quick_inference.py --checkpoint MODEL.pth --data DATA_FOLDER
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from pipeline_preprocess import load_hyperspectral_cube, preprocess_cube
from pipeline_dataset import CLASS_NAMES, CLASS_MAPPING
from pipeline_model import create_model


# Inverse class mapping
CLASS_TO_RGB = {v: k for k, v in CLASS_MAPPING.items()}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Quick inference on hyperspectral data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to hyperspectral data folder (normalized)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/quick_inference',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=512,
        help='Batch size for inference'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'mps', 'cuda', 'cpu'],
        help='Device to use (auto, mps, cuda, or cpu)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for filtering predictions'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    
    return parser.parse_args()


def get_device(device_arg):
    """Get computation device."""
    if device_arg == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_arg)


def load_model(checkpoint_path, num_bands, device):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    train_config = checkpoint.get('config', {})
    model_type = train_config.get('model_type', 'spectral_cnn')
    dropout_rate = train_config.get('dropout_rate', 0.5)
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    print(f"  Model type: {model_type}")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    
    # Create and load model
    model = create_model(
        num_bands=num_bands,
        num_classes=len(CLASS_NAMES),
        model_type=model_type,
        dropout_rate=dropout_rate
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  Model loaded successfully")
    
    return model, train_config


def predict_cube(model, cube, device, batch_size=512):
    """Perform inference on hyperspectral cube."""
    h, w, bands = cube.shape
    pixels = cube.reshape(-1, bands)
    num_pixels = pixels.shape[0]
    
    predictions = np.zeros(num_pixels, dtype=np.int32)
    confidences = np.zeros(num_pixels, dtype=np.float32)
    
    print(f"Running inference on {num_pixels:,} pixels...")
    
    with torch.no_grad():
        for i in range(0, num_pixels, batch_size):
            batch_end = min(i + batch_size, num_pixels)
            batch = pixels[i:batch_end]
            
            batch_tensor = torch.FloatTensor(batch).to(device)
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            conf, pred = probs.max(dim=1)
            predictions[i:batch_end] = pred.cpu().numpy()
            confidences[i:batch_end] = conf.cpu().numpy()
            
            if (i // batch_size) % 10 == 0:
                progress = 100.0 * i / num_pixels
                print(f"  Progress: {progress:.1f}%", end='\r')
    
    print(f"  Progress: 100.0%")
    
    prediction_map = predictions.reshape(h, w)
    confidence_map = confidences.reshape(h, w)
    
    return prediction_map, confidence_map


def create_rgb_visualization(prediction_map):
    """Convert prediction map to RGB image."""
    h, w = prediction_map.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(len(CLASS_NAMES)):
        if class_id in CLASS_TO_RGB:
            mask = prediction_map == class_id
            rgb_image[mask] = CLASS_TO_RGB[class_id]
    
    return rgb_image


def save_results(prediction_map, confidence_map, rgb_image, output_dir):
    """Save inference results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numpy arrays
    np.save(output_dir / 'prediction_map.npy', prediction_map)
    np.save(output_dir / 'confidence_map.npy', confidence_map)
    
    # Save RGB image
    rgb_pil = Image.fromarray(rgb_image)
    rgb_pil.save(output_dir / 'prediction_rgb.png')
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - prediction_map.npy")
    print(f"  - confidence_map.npy")
    print(f"  - prediction_rgb.png")


def create_visualization(prediction_map, confidence_map, rgb_image, output_dir, threshold):
    """Create and save visualization."""
    output_dir = Path(output_dir)
    
    fig = plt.figure(figsize=(18, 5))
    
    # Predicted classes
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(rgb_image)
    ax1.set_title('Predicted Classes', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Confidence map
    ax2 = plt.subplot(1, 3, 2)
    im = ax2.imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
    ax2.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, label='Confidence')
    
    # High confidence only
    ax3 = plt.subplot(1, 3, 3)
    filtered_rgb = rgb_image.copy()
    low_conf_mask = confidence_map < threshold
    filtered_rgb[low_conf_mask] = [128, 128, 128]  # Gray for low confidence
    ax3.imshow(filtered_rgb)
    ax3.set_title(f'High Confidence (>{threshold})', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'visualization.png', dpi=150, bbox_inches='tight')
    print(f"  - visualization.png")
    plt.close()


def print_statistics(prediction_map, confidence_map, threshold):
    """Print inference statistics."""
    print("\n" + "="*60)
    print("INFERENCE STATISTICS")
    print("="*60)
    
    # Class distribution
    print("\nPredicted class distribution:")
    unique, counts = np.unique(prediction_map, return_counts=True)
    total_pixels = prediction_map.size
    
    for class_id, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
        if class_id < len(CLASS_NAMES):
            percentage = 100.0 * count / total_pixels
            print(f"  {CLASS_NAMES[class_id]:20s}: {count:8,} pixels ({percentage:5.2f}%)")
    
    # Confidence statistics
    print(f"\nConfidence statistics:")
    print(f"  Mean:   {confidence_map.mean():.4f}")
    print(f"  Median: {np.median(confidence_map):.4f}")
    print(f"  Std:    {confidence_map.std():.4f}")
    print(f"  Min:    {confidence_map.min():.4f}")
    print(f"  Max:    {confidence_map.max():.4f}")
    
    # Confidence filtering
    low_conf_pixels = (confidence_map < threshold).sum()
    low_conf_pct = 100.0 * low_conf_pixels / total_pixels
    high_conf_pixels = total_pixels - low_conf_pixels
    high_conf_pct = 100.0 - low_conf_pct
    
    print(f"\nConfidence threshold: {threshold}")
    print(f"  High confidence (>={threshold}): {high_conf_pixels:,} pixels ({high_conf_pct:.2f}%)")
    print(f"  Low confidence (<{threshold}):  {low_conf_pixels:,} pixels ({low_conf_pct:.2f}%)")


def main():
    """Main function."""
    args = parse_args()
    
    print("="*60)
    print("QUICK HYPERSPECTRAL INFERENCE")
    print("="*60)
    
    # Get device
    device = get_device(args.device)
    print(f"\nUsing device: {device}")
    
    # Load data
    print(f"\nLoading hyperspectral data: {args.data}")
    cube, wavelengths, header = load_hyperspectral_cube(args.data)
    print(f"  Cube shape: {cube.shape}")
    print(f"  Wavelengths: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    
    # Load model
    print()
    num_bands = cube.shape[2]
    model, train_config = load_model(args.checkpoint, num_bands, device)
    
    # Perform inference
    print()
    prediction_map, confidence_map = predict_cube(
        model, cube, device, batch_size=args.batch_size
    )
    
    # Print statistics
    print_statistics(prediction_map, confidence_map, args.threshold)
    
    # Create RGB visualization
    print("\nCreating visualizations...")
    rgb_image = create_rgb_visualization(prediction_map)
    
    # Save results
    save_results(prediction_map, confidence_map, rgb_image, args.output)
    
    # Create visualization
    if not args.no_viz:
        create_visualization(
            prediction_map, confidence_map, rgb_image, 
            args.output, args.threshold
        )
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == '__main__':
    main()

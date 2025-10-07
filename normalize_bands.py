"""
Band-wise Normalization of Hyperspectral Data

This script performs percentile-based normalization on each spectral band
and exports the normalized images to a new folder.

Usage:
    python normalize_bands.py

Configuration:
    Edit the configuration section below to customize:
    - Input folder
    - Output folder
    - Percentile values
"""

import numpy as np
import json
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import os
from tqdm import tqdm

# ========================================
# CONFIGURATION
# ========================================
INPUT_FOLDER = 'training_dataset'
OUTPUT_FOLDER = 'normalized_dataset'
LOWER_PERCENTILE = 2
UPPER_PERCENTILE = 98
# ========================================


def normalize_band_percentile(band, lower_percentile=2, upper_percentile=98):
    """
    Normalize a single band using percentile-based stretching

    Args:
        band: 2D numpy array (height, width)
        lower_percentile: Lower percentile for clipping (default: 2)
        upper_percentile: Upper percentile for clipping (default: 98)

    Returns:
        Normalized band with values stretched to 0-255
    """
    # Calculate percentiles
    p_low = np.percentile(band, lower_percentile)
    p_high = np.percentile(band, upper_percentile)

    # Avoid division by zero
    if p_high - p_low < 1e-6:
        return band

    # Clip and normalize to 0-255
    band_clipped = np.clip(band, p_low, p_high)
    band_normalized = ((band_clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)

    return band_normalized


def load_hyperspectral_data(input_folder):
    """
    Load hyperspectral cube from folder

    Args:
        input_folder: Path to folder containing images and header.json

    Returns:
        tuple: (cube, wavelengths, num_bands, header)
    """
    print(f"\nLoading hyperspectral data from: {input_folder}/")
    print("-" * 70)

    # Check if folder exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Folder '{input_folder}' not found!")

    # Load header information
    header_path = os.path.join(input_folder, 'header.json')
    if not os.path.exists(header_path):
        raise FileNotFoundError(f"header.json not found in {input_folder}/")

    with open(header_path, 'r') as f:
        header = json.load(f)

    wavelengths = np.array(header['wavelength (nm)'])
    num_bands = len(wavelengths)

    print(f"✓ Header loaded")
    print(f"  Number of spectral bands: {num_bands}")
    print(f"  Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm")

    # Load images
    image_folder = Path(input_folder)

    # Load first image to get dimensions
    first_img = Image.open(image_folder / 'ImagesStack001.png')
    height, width = first_img.size[1], first_img.size[0]
    print(f"  Image dimensions: {width} x {height} pixels")

    # Initialize hyperspectral cube
    cube = np.zeros((height, width, num_bands), dtype=np.uint8)

    print(f"\nLoading {num_bands} images...")
    # Load all images
    for i in tqdm(range(1, num_bands + 1), desc="Loading", unit="img"):
        img_path = image_folder / f'ImagesStack{i:03d}.png'

        if not img_path.exists():
            print(f"⚠ Warning: {img_path} not found, skipping...")
            continue

        img = Image.open(img_path).convert('L')
        cube[:, :, i-1] = np.array(img)

    print(f"\n✓ Hyperspectral cube loaded: {cube.shape}")
    print(f"  Memory size: {cube.nbytes / (1024**2):.1f} MB")

    return cube, wavelengths, num_bands, header


def normalize_all_bands(cube, lower_percentile, upper_percentile):
    """
    Normalize all bands in the hyperspectral cube

    Args:
        cube: 3D numpy array (height, width, bands)
        lower_percentile: Lower percentile for clipping
        upper_percentile: Upper percentile for clipping

    Returns:
        Normalized cube
    """
    print(f"\nNormalizing all bands using {lower_percentile}th and {upper_percentile}th percentiles...")
    print("-" * 70)

    num_bands = cube.shape[2]
    normalized_cube = np.zeros_like(cube, dtype=np.uint8)

    # Normalize each band
    for i in tqdm(range(num_bands), desc="Normalizing", unit="band"):
        normalized_cube[:, :, i] = normalize_band_percentile(
            cube[:, :, i],
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile
        )

    print(f"\n✓ All bands normalized")
    print(f"  Normalized cube shape: {normalized_cube.shape}")

    return normalized_cube


def export_normalized_images(normalized_cube, output_folder):
    """
    Export normalized images to folder

    Args:
        normalized_cube: 3D numpy array of normalized data
        output_folder: Path to output folder
    """
    print(f"\nExporting normalized images to: {output_folder}/")
    print("-" * 70)

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    num_bands = normalized_cube.shape[2]

    # Export each normalized band as PNG
    for i in tqdm(range(num_bands), desc="Exporting", unit="img"):
        normalized_band = normalized_cube[:, :, i]

        # Save as PNG
        img = Image.fromarray(normalized_band, mode='L')
        img.save(output_path / f'ImagesStack{i+1:03d}.png')

    print(f"\n✓ All {num_bands} normalized images exported")


def generate_statistics(cube, normalized_cube, wavelengths, output_folder):
    """
    Generate and save normalization statistics

    Args:
        cube: Original cube
        normalized_cube: Normalized cube
        wavelengths: Array of wavelengths
        output_folder: Path to output folder
    """
    print(f"\nGenerating normalization statistics...")
    print("-" * 70)

    num_bands = cube.shape[2]

    stats = {
        'band': [],
        'wavelength': [],
        'original_min': [],
        'original_max': [],
        'original_mean': [],
        'normalized_min': [],
        'normalized_max': [],
        'normalized_mean': []
    }

    for i in range(num_bands):
        stats['band'].append(i + 1)
        stats['wavelength'].append(float(wavelengths[i]))
        stats['original_min'].append(int(cube[:, :, i].min()))
        stats['original_max'].append(int(cube[:, :, i].max()))
        stats['original_mean'].append(float(cube[:, :, i].mean()))
        stats['normalized_min'].append(int(normalized_cube[:, :, i].min()))
        stats['normalized_max'].append(int(normalized_cube[:, :, i].max()))
        stats['normalized_mean'].append(float(normalized_cube[:, :, i].mean()))

    # Save statistics to JSON
    output_path = Path(output_folder)
    stats_file = output_path / 'normalization_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Statistics saved to '{stats_file}'")

    # Display sample statistics
    print("\nSample statistics (first 5 bands):")
    print(f"{'Band':>4} {'Wavelength':>10} {'Orig Min':>9} {'Orig Max':>9} {'Norm Min':>9} {'Norm Max':>9}")
    print("-" * 65)
    for i in range(min(5, num_bands)):
        print(f"{stats['band'][i]:4d} {stats['wavelength'][i]:10.2f} "
              f"{stats['original_min'][i]:9d} {stats['original_max'][i]:9d} "
              f"{stats['normalized_min'][i]:9d} {stats['normalized_max'][i]:9d}")

    return stats


def create_comparison_plots(cube, normalized_cube, wavelengths, output_folder):
    """
    Create and save comparison plots

    Args:
        cube: Original cube
        normalized_cube: Normalized cube
        wavelengths: Array of wavelengths
        output_folder: Path to output folder
    """
    print(f"\nCreating comparison plots...")
    print("-" * 70)

    num_bands = cube.shape[2]

    # Sample band for detailed comparison
    sample_band_idx = num_bands // 2
    original_band = cube[:, :, sample_band_idx]
    normalized_band = normalized_cube[:, :, sample_band_idx]

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original image
    axes[0, 0].imshow(original_band, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title(f'Original Band {sample_band_idx+1}\n({wavelengths[sample_band_idx]:.2f} nm)')
    axes[0, 0].axis('off')

    # Normalized image
    axes[0, 1].imshow(normalized_band, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(f'Normalized Band {sample_band_idx+1}\n({wavelengths[sample_band_idx]:.2f} nm)')
    axes[0, 1].axis('off')

    # Original histogram
    axes[1, 0].hist(original_band.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7)
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlabel('Intensity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)

    # Normalized histogram
    axes[1, 1].hist(normalized_band.ravel(), bins=256, range=(0, 255), color='green', alpha=0.7)
    axes[1, 1].set_title('Normalized Histogram')
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_folder)
    plt.savefig(output_path / 'sample_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Sample comparison saved to '{output_path / 'sample_comparison.png'}'")
    plt.close()

    # Plot statistics across all bands
    stats_orig_min = [cube[:, :, i].min() for i in range(num_bands)]
    stats_orig_max = [cube[:, :, i].max() for i in range(num_bands)]
    stats_orig_mean = [cube[:, :, i].mean() for i in range(num_bands)]
    stats_norm_min = [normalized_cube[:, :, i].min() for i in range(num_bands)]
    stats_norm_max = [normalized_cube[:, :, i].max() for i in range(num_bands)]
    stats_norm_mean = [normalized_cube[:, :, i].mean() for i in range(num_bands)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    bands_array = np.arange(1, num_bands + 1)

    # Min values
    axes[0].plot(bands_array, stats_orig_min, label='Original', linewidth=1.5, alpha=0.7)
    axes[0].plot(bands_array, stats_norm_min, label='Normalized', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('Minimum Intensity')
    axes[0].set_title('Minimum Intensity per Band')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Max values
    axes[1].plot(bands_array, stats_orig_max, label='Original', linewidth=1.5, alpha=0.7)
    axes[1].plot(bands_array, stats_norm_max, label='Normalized', linewidth=1.5, alpha=0.7)
    axes[1].set_ylabel('Maximum Intensity')
    axes[1].set_title('Maximum Intensity per Band')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Mean values
    axes[2].plot(bands_array, stats_orig_mean, label='Original', linewidth=1.5, alpha=0.7)
    axes[2].plot(bands_array, stats_norm_mean, label='Normalized', linewidth=1.5, alpha=0.7)
    axes[2].set_ylabel('Mean Intensity')
    axes[2].set_xlabel('Band Number')
    axes[2].set_title('Mean Intensity per Band')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'normalization_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Full comparison saved to '{output_path / 'normalization_comparison.png'}'")
    plt.close()


def main():
    """Main function"""
    print("\n" + "="*70)
    print("HYPERSPECTRAL BAND NORMALIZATION")
    print("="*70)
    print(f"Input folder:  {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Percentiles:   {LOWER_PERCENTILE}% - {UPPER_PERCENTILE}%")
    print("="*70)

    try:
        # Load data
        cube, wavelengths, num_bands, header = load_hyperspectral_data(INPUT_FOLDER)

        # Normalize all bands
        normalized_cube = normalize_all_bands(cube, LOWER_PERCENTILE, UPPER_PERCENTILE)

        # Export normalized images
        export_normalized_images(normalized_cube, OUTPUT_FOLDER)

        # Copy header file
        print(f"\nCopying header file...")
        shutil.copy(
            os.path.join(INPUT_FOLDER, 'header.json'),
            os.path.join(OUTPUT_FOLDER, 'header.json')
        )
        print(f"✓ Header file copied to '{OUTPUT_FOLDER}/header.json'")

        # Generate statistics
        stats = generate_statistics(cube, normalized_cube, wavelengths, OUTPUT_FOLDER)

        # Create comparison plots
        create_comparison_plots(cube, normalized_cube, wavelengths, OUTPUT_FOLDER)

        # Summary
        print("\n" + "="*70)
        print("NORMALIZATION COMPLETE!")
        print("="*70)
        print(f"✓ {num_bands} bands normalized and exported")
        print(f"✓ Output location: {OUTPUT_FOLDER}/")
        print(f"✓ Statistics saved: {OUTPUT_FOLDER}/normalization_stats.json")
        print(f"✓ Comparison plots: {OUTPUT_FOLDER}/normalization_comparison.png")
        print(f"✓ Sample comparison: {OUTPUT_FOLDER}/sample_comparison.png")
        print("="*70 + "\n")

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check the INPUT_FOLDER variable at the top of this script.")
        return 1
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

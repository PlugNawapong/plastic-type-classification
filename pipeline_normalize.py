"""
Data Normalization Module
Applies percentile-based normalization to hyperspectral bands
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def normalize_band_percentile(band, lower_percentile=2, upper_percentile=98):
    """
    Normalize a single band using percentile-based clipping and stretching.

    Args:
        band: 2D numpy array (H x W)
        lower_percentile: Lower percentile for clipping (default: 2)
        upper_percentile: Upper percentile for clipping (default: 98)

    Returns:
        Normalized band as uint8 (0-255)
    """
    p_low = np.percentile(band, lower_percentile)
    p_high = np.percentile(band, upper_percentile)

    band_clipped = np.clip(band, p_low, p_high)

    if p_high > p_low:
        band_normalized = ((band_clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    else:
        band_normalized = np.zeros_like(band, dtype=np.uint8)

    return band_normalized


def normalize_dataset(input_folder, output_folder, lower_percentile=2, upper_percentile=98):
    """
    Normalize all bands in a hyperspectral dataset.

    Args:
        input_folder: Path to folder containing ImagesStack*.png files
        output_folder: Path to folder for saving normalized bands
        lower_percentile: Lower percentile for clipping
        upper_percentile: Upper percentile for clipping

    Returns:
        Dictionary with normalization metadata
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load header.json for metadata
    header_file = input_path / 'header.json'
    if header_file.exists():
        with open(header_file, 'r') as f:
            header = json.load(f)
        # Copy header to output folder
        with open(output_path / 'header.json', 'w') as f:
            json.dump(header, f, indent=2)
        num_bands = len(header['img no.'])
    else:
        # Count files
        band_files = sorted(input_path.glob('ImagesStack*.png'))
        num_bands = len(band_files)

    print(f"Normalizing {num_bands} bands from {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Percentile range: {lower_percentile}% - {upper_percentile}%")

    # Process each band
    metadata = {
        'num_bands': num_bands,
        'lower_percentile': lower_percentile,
        'upper_percentile': upper_percentile,
        'band_stats': []
    }

    for i in tqdm(range(1, num_bands + 1), desc="Normalizing bands"):
        # Load band
        input_file = input_path / f'ImagesStack{i:03d}.png'
        band = np.array(Image.open(input_file))

        # Normalize
        band_normalized = normalize_band_percentile(band, lower_percentile, upper_percentile)

        # Save
        output_file = output_path / f'ImagesStack{i:03d}.png'
        Image.fromarray(band_normalized).save(output_file)

        # Store stats
        metadata['band_stats'].append({
            'band_id': i,
            'original_min': float(band.min()),
            'original_max': float(band.max()),
            'original_mean': float(band.mean()),
            'normalized_min': int(band_normalized.min()),
            'normalized_max': int(band_normalized.max()),
            'normalized_mean': float(band_normalized.mean())
        })

    # Save metadata
    metadata_file = output_path / 'normalization_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nNormalization complete!")
    print(f"Metadata saved to {metadata_file}")

    return metadata


if __name__ == '__main__':
    # Configuration
    INPUT_FOLDER = 'training_dataset'
    OUTPUT_FOLDER = 'training_dataset_normalized'
    LOWER_PERCENTILE = 2
    UPPER_PERCENTILE = 98

    # Run normalization
    metadata = normalize_dataset(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        lower_percentile=LOWER_PERCENTILE,
        upper_percentile=UPPER_PERCENTILE
    )

    print(f"\nProcessed {metadata['num_bands']} bands")

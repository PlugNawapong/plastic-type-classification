"""
Preprocessing Module
Applies various preprocessing operations to hyperspectral cube
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter, median_filter


def load_hyperspectral_cube(data_folder):
    """
    Load hyperspectral cube from folder containing ImagesStack*.png files.

    Returns:
        cube: 3D numpy array (H x W x Bands)
        wavelengths: 1D numpy array of wavelengths
        header: Dictionary with metadata
    """
    data_path = Path(data_folder)

    # Load header
    header_file = data_path / 'header.json'
    with open(header_file, 'r') as f:
        header = json.load(f)

    wavelengths = np.array(header['wavelength (nm)'])
    num_bands = len(wavelengths)

    print(f"Loading {num_bands} bands from {data_folder}...")

    # Load first band to get dimensions
    first_band = np.array(Image.open(data_path / 'ImagesStack001.png'))
    height, width = first_band.shape

    # Initialize cube
    cube = np.zeros((height, width, num_bands), dtype=np.uint8)

    # Load all bands
    for i in range(1, num_bands + 1):
        band_file = data_path / f'ImagesStack{i:03d}.png'
        cube[:, :, i-1] = np.array(Image.open(band_file))

    print(f"Loaded cube shape: {cube.shape}")

    return cube, wavelengths, header


def filter_wavelength_range(cube, wavelengths, wavelength_range):
    """
    Filter cube to specific wavelength range.

    Args:
        cube: 3D array (H x W x Bands)
        wavelengths: 1D array of wavelengths
        wavelength_range: Tuple (min_wavelength, max_wavelength)

    Returns:
        filtered_cube: 3D array with selected bands
        filtered_wavelengths: 1D array of selected wavelengths
    """
    min_wl, max_wl = wavelength_range
    mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
    filtered_cube = cube[:, :, mask]
    filtered_wavelengths = wavelengths[mask]
    print(f"Wavelength filter: {len(filtered_wavelengths)} bands in range [{min_wl}, {max_wl}] nm")
    return filtered_cube, filtered_wavelengths


def select_n_bands(cube, wavelengths, n_bands):
    """
    Select N evenly-spaced bands from the cube.

    Args:
        cube: 3D array (H x W x Bands)
        wavelengths: 1D array of wavelengths
        n_bands: Number of bands to select

    Returns:
        selected_cube: 3D array with N bands
        selected_wavelengths: 1D array of N wavelengths
    """
    total_bands = cube.shape[2]
    indices = np.linspace(0, total_bands - 1, n_bands, dtype=int)
    selected_cube = cube[:, :, indices]
    selected_wavelengths = wavelengths[indices]
    print(f"Band selection: {n_bands} bands selected from {total_bands}")
    return selected_cube, selected_wavelengths


def bin_spectral_bands(cube, wavelengths, bin_size):
    """
    Average consecutive spectral bands to reduce noise and dimensionality.

    Args:
        cube: 3D array (H x W x Bands)
        wavelengths: 1D array of wavelengths
        bin_size: Number of consecutive bands to average

    Returns:
        binned_cube: 3D array with reduced bands
        binned_wavelengths: 1D array of averaged wavelengths
    """
    height, width, num_bands = cube.shape
    num_bins = num_bands // bin_size

    binned_cube = np.zeros((height, width, num_bins), dtype=np.float32)
    binned_wavelengths = np.zeros(num_bins)

    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size
        binned_cube[:, :, i] = np.mean(cube[:, :, start_idx:end_idx], axis=2)
        binned_wavelengths[i] = np.mean(wavelengths[start_idx:end_idx])

    print(f"Spectral binning: {num_bands} bands -> {num_bins} bands (bin_size={bin_size})")
    return binned_cube.astype(np.uint8), binned_wavelengths


def bin_spatial(cube, bin_size):
    """
    Average spatial pixels in blocks to reduce resolution.

    Args:
        cube: 3D array (H x W x Bands)
        bin_size: Block size (e.g., 2 for 2x2 blocks)

    Returns:
        binned_cube: 3D array with reduced spatial dimensions
    """
    height, width, num_bands = cube.shape
    new_height = height // bin_size
    new_width = width // bin_size

    binned_cube = np.zeros((new_height, new_width, num_bands), dtype=np.float32)

    for i in range(new_height):
        for j in range(new_width):
            h_start = i * bin_size
            h_end = h_start + bin_size
            w_start = j * bin_size
            w_end = w_start + bin_size
            binned_cube[i, j, :] = np.mean(cube[h_start:h_end, w_start:w_end, :], axis=(0, 1))

    print(f"Spatial binning: ({height}, {width}) -> ({new_height}, {new_width}) (bin_size={bin_size})")
    return binned_cube.astype(np.uint8)


def bin_spatial_labels(labels, bin_size):
    """
    Downsample label image using majority voting within each block.

    Args:
        labels: 2D array (H x W) with class labels
        bin_size: Block size (e.g., 2 for 2x2 blocks)

    Returns:
        binned_labels: 2D array with reduced spatial dimensions
    """
    from scipy.stats import mode

    height, width = labels.shape
    new_height = height // bin_size
    new_width = width // bin_size

    binned_labels = np.zeros((new_height, new_width), dtype=labels.dtype)

    for i in range(new_height):
        for j in range(new_width):
            h_start = i * bin_size
            h_end = h_start + bin_size
            w_start = j * bin_size
            w_end = w_start + bin_size

            # Use majority voting (mode) for labels
            block = labels[h_start:h_end, w_start:w_end]
            binned_labels[i, j] = mode(block.flatten(), keepdims=False)[0]

    print(f"Label spatial binning: ({height}, {width}) -> ({new_height}, {new_width}) (bin_size={bin_size})")
    return binned_labels


def denoise_cube(cube, method='gaussian', strength=1.0):
    """
    Apply denoising filter to each band.

    Args:
        cube: 3D array (H x W x Bands)
        method: 'gaussian', 'median', or 'bilateral'
        strength: Denoising strength (sigma for gaussian, size for median)

    Returns:
        denoised_cube: 3D array with denoised bands
    """
    height, width, num_bands = cube.shape
    denoised_cube = np.zeros_like(cube)

    print(f"Denoising with {method} filter (strength={strength})...")

    if method == 'gaussian':
        for i in range(num_bands):
            denoised_cube[:, :, i] = gaussian_filter(cube[:, :, i], sigma=strength)
    elif method == 'median':
        size = int(strength * 2 + 1)
        for i in range(num_bands):
            denoised_cube[:, :, i] = median_filter(cube[:, :, i], size=size)
    else:
        raise ValueError(f"Unsupported denoising method: {method}")

    print(f"Denoising complete")
    return denoised_cube


def preprocess_cube(cube, wavelengths, config):
    """
    Apply preprocessing pipeline to hyperspectral cube.

    Args:
        cube: 3D array (H x W x Bands)
        wavelengths: 1D array of wavelengths
        config: Dictionary with preprocessing options

    Returns:
        processed_cube: 3D array
        processed_wavelengths: 1D array
    """
    print("\n=== Preprocessing Pipeline ===")

    # Step 1: Wavelength range filtering
    if config.get('wavelength_range'):
        cube, wavelengths = filter_wavelength_range(
            cube, wavelengths, config['wavelength_range']
        )

    # Step 2: Band selection (mutually exclusive with spectral binning)
    if config.get('select_n_bands'):
        cube, wavelengths = select_n_bands(
            cube, wavelengths, config['select_n_bands']
        )
    elif config.get('spectral_binning'):
        cube, wavelengths = bin_spectral_bands(
            cube, wavelengths, config['spectral_binning']
        )

    # Step 3: Spatial binning
    if config.get('spatial_binning'):
        cube = bin_spatial(cube, config['spatial_binning'])

    # Step 4: Denoising
    if config.get('denoise_enabled'):
        cube = denoise_cube(
            cube,
            method=config.get('denoise_method', 'gaussian'),
            strength=config.get('denoise_strength', 1.0)
        )

    print(f"\nFinal cube shape: {cube.shape}")
    print(f"Final wavelengths: {len(wavelengths)} bands")

    return cube, wavelengths


if __name__ == '__main__':
    # Configuration
    DATA_FOLDER = 'training_dataset_normalized'

    PREPROCESS_CONFIG = {
        'wavelength_range': None,  # e.g., (450, 700)
        'select_n_bands': None,  # e.g., 100
        'spectral_binning': None,  # e.g., 2
        'spatial_binning': None,  # e.g., 2
        'denoise_enabled': False,
        'denoise_method': 'gaussian',
        'denoise_strength': 1.0
    }

    # Load cube
    cube, wavelengths, header = load_hyperspectral_cube(DATA_FOLDER)

    # Preprocess
    processed_cube, processed_wavelengths = preprocess_cube(
        cube, wavelengths, PREPROCESS_CONFIG
    )

    print(f"\nPreprocessed cube ready for training")
    print(f"Shape: {processed_cube.shape}")

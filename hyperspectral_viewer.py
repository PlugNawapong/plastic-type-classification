"""
Hyperspectral Cube Viewer
Interactive visualization tool for hyperspectral data with spectral signature plotting.

Usage:
    python hyperspectral_viewer.py

Configuration:
    Edit the configuration section below to customize:
    - DATA_FOLDER: Select which dataset to view
    - WAVELENGTH_RANGE: Filter to specific wavelength range (e.g., (450, 700) for 450-700 nm)
                        Set to None to view all wavelengths
    - SELECT_N_BANDS: Select N evenly-spaced bands from the range (reduces data size)
                      Set to None to use all bands in the range
    - SPECTRAL_BINNING: Average consecutive bands to reduce noise (e.g., 2 = average pairs)
                        Mutually exclusive with SELECT_N_BANDS
    - SPATIAL_BINNING: Average spatial pixels to reduce image size (e.g., 2 = 2x2 pixel bins)
                       Can be combined with spectral binning
    - DENOISE_ENABLED: Enable/disable noise reduction filter
    - DENOISE_METHOD: Denoising algorithm ('gaussian', 'median', 'bilateral')
    - DENOISE_STRENGTH: Strength of denoising (higher = more smoothing)

Examples:
    # View visible spectrum with both spectral and spatial binning
    DATA_FOLDER = 'training_dataset'
    WAVELENGTH_RANGE = (450, 700)
    SELECT_N_BANDS = None
    SPECTRAL_BINNING = 2   # 284 bands → 142 bands
    SPATIAL_BINNING = 2    # 5496x3672 → 2748x1836
    DENOISE_ENABLED = False

    # View 50 evenly-spaced bands with spatial binning
    DATA_FOLDER = 'training_dataset'
    WAVELENGTH_RANGE = (450, 700)
    SELECT_N_BANDS = 50
    SPECTRAL_BINNING = None
    SPATIAL_BINNING = 4    # 4x4 pixel bins
    DENOISE_ENABLED = False

    # View all bands with 3-band spectral binning only
    DATA_FOLDER = 'normalized_dataset'
    WAVELENGTH_RANGE = None
    SELECT_N_BANDS = None
    SPECTRAL_BINNING = 3   # 458 bands → 152 bands
    SPATIAL_BINNING = None
    DENOISE_ENABLED = False
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import json
from PIL import Image
from pathlib import Path
import os
from scipy.ndimage import gaussian_filter, median_filter
from scipy.ndimage import uniform_filter

# ========================================
# CONFIGURATION
# ========================================
DATA_FOLDER = 'normalized_dataset_new'  # Options: 'training_dataset', 'normalized_dataset'

# Wavelength range filter (set to None to disable)
# Example: (450, 700) to view only 450-700 nm range
WAVELENGTH_RANGE = (450,700)  # Options: None, (min_wl, max_wl)

# Band selection - select specific number of bands within wavelength range
# Set to None to use all bands in the range, or specify number of bands to uniformly sample
# Example: SELECT_N_BANDS = 50 will select 50 evenly-spaced bands from the wavelength range
SELECT_N_BANDS = None  # Options: None, or integer (e.g., 50, 100)

# Spectral binning - average consecutive bands to reduce noise and data size
# Set to None to disable, or specify bin size (e.g., 2 = average every 2 bands, 3 = every 3 bands)
# Example: SPECTRAL_BINNING = 2 will reduce 458 bands to 229 bands by averaging pairs
SPECTRAL_BINNING = 10  # Options: None, or integer (e.g., 2, 3, 5)

# Spatial binning - average spatial pixels to reduce image size and noise
# Set to None to disable, or specify bin size (e.g., 2 = 2x2 pixel bins, 4 = 4x4 pixel bins)
# Example: SPATIAL_BINNING = 2 will reduce 5496x3672 to 2748x1836 by averaging 2x2 blocks
SPATIAL_BINNING = 4  # Options: None, or integer (e.g., 2, 4, 8)

# Noise reduction (denoising) filter
DENOISE_ENABLED = True  # Set to True to enable denoising
DENOISE_METHOD = 'gaussian'  # Options: 'gaussian', 'median', 'bilateral'
DENOISE_STRENGTH = 3.0  # Strength parameter (higher = more smoothing)
# ========================================


class HyperspectralViewer:
    def __init__(self, cube, wavelengths, data_folder):
        self.cube = cube
        self.wavelengths = wavelengths
        self.data_folder = data_folder
        self.current_band = 0
        self.selected_pixels = []

        # Create figure
        self.fig = plt.figure(figsize=(15, 7))
        self.fig.canvas.manager.set_window_title(f'Hyperspectral Viewer - {data_folder}')

        # Image display (left)
        self.ax_img = plt.subplot(121)
        self.im = self.ax_img.imshow(cube[:, :, 0], cmap='gray', vmin=0, vmax=255)
        self.ax_img.set_title(f'Band 1 - {wavelengths[0]:.2f} nm', fontsize=12, fontweight='bold')
        self.ax_img.set_xlabel('Width (pixels)', fontsize=10)
        self.ax_img.set_ylabel('Height (pixels)', fontsize=10)
        self.colorbar = plt.colorbar(self.im, ax=self.ax_img, label='Intensity')

        # Spectral signature plot (right)
        self.ax_spec = plt.subplot(122)
        self.ax_spec.set_xlabel('Wavelength (nm)', fontsize=11)
        self.ax_spec.set_ylabel('Intensity', fontsize=11)
        self.ax_spec.set_title('Spectral Signatures (Click on image)', fontsize=12, fontweight='bold')
        self.ax_spec.grid(True, alpha=0.3)
        self.ax_spec.set_xlim(wavelengths[0], wavelengths[-1])
        self.ax_spec.set_ylim(0, 255)

        # Add slider
        plt.subplots_adjust(bottom=0.15)
        ax_slider = plt.axes([0.15, 0.05, 0.3, 0.03])
        self.slider = Slider(ax_slider, 'Band', 1, len(wavelengths),
                            valinit=1, valstep=1, color='steelblue')
        self.slider.on_changed(self.update_band)

        # Connect click event
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.spec_lines = []
        self.markers = []

        # Add instruction text
        instruction_text = (
            "LEFT CLICK: View spectral signature | "
            "RIGHT CLICK: Clear all | "
            "SLIDER: Change band"
        )
        self.fig.text(0.5, 0.01, instruction_text, ha='center', fontsize=9,
                     style='italic', color='darkblue')

        plt.tight_layout()
        print("\n" + "="*70)
        print("HYPERSPECTRAL VIEWER READY")
        print("="*70)
        print(f"Data source: {data_folder}/")
        print(f"Viewing band 1 of {len(wavelengths)} ({wavelengths[0]:.2f} nm)")
        print("\nControls:")
        print("  • Use SLIDER to navigate through spectral bands")
        print("  • LEFT CLICK on image to view pixel's spectral signature")
        print("  • RIGHT CLICK anywhere to clear all selections")
        print("  • Close window or press Ctrl+C to exit")
        print("="*70 + "\n")

    def update_band(self, val):
        """Update displayed band when slider changes"""
        self.current_band = int(self.slider.val) - 1
        self.im.set_data(self.cube[:, :, self.current_band])
        self.ax_img.set_title(
            f'Band {self.current_band + 1} - {self.wavelengths[self.current_band]:.2f} nm',
            fontsize=12, fontweight='bold'
        )

        # Update markers on current band
        for marker in self.markers:
            marker.remove()
        self.markers = []

        for x, y, color in self.selected_pixels:
            marker, = self.ax_img.plot(x, y, 'o', color=color, markersize=8,
                                      markeredgecolor='white', markeredgewidth=1.5)
            self.markers.append(marker)

        self.fig.canvas.draw_idle()

    def onclick(self, event):
        """Handle click events"""
        if event.inaxes == self.ax_img and event.button == 1:  # Left click on image
            try:
                x, y = int(event.xdata), int(event.ydata)
            except:
                return  # Click outside valid area

            # Check bounds
            if 0 <= x < self.cube.shape[1] and 0 <= y < self.cube.shape[0]:
                # Extract spectral signature
                signature = self.cube[y, x, :]

                # Generate color for this pixel
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
                color = colors[len(self.selected_pixels) % len(colors)]

                # Store pixel location
                self.selected_pixels.append((x, y, color))

                # Plot spectral signature
                line, = self.ax_spec.plot(self.wavelengths, signature,
                                         label=f'Pixel ({x}, {y})',
                                         color=color, linewidth=2, alpha=0.8)
                self.spec_lines.append(line)

                # Add marker on image
                marker, = self.ax_img.plot(x, y, 'o', color=color, markersize=8,
                                          markeredgecolor='white', markeredgewidth=1.5)
                self.markers.append(marker)

                # Update legend
                self.ax_spec.legend(loc='upper right', fontsize=9)

                self.fig.canvas.draw_idle()

                print(f"Added pixel ({x}, {y}) - {len(self.selected_pixels)} total")

        elif event.button == 3:  # Right click anywhere to clear
            # Clear all spectral signatures
            for line in self.spec_lines:
                line.remove()
            for marker in self.markers:
                marker.remove()

            self.spec_lines = []
            self.markers = []
            num_cleared = len(self.selected_pixels)
            self.selected_pixels = []

            if self.ax_spec.get_legend():
                self.ax_spec.get_legend().remove()

            self.fig.canvas.draw_idle()

            if num_cleared > 0:
                print(f"Cleared {num_cleared} pixel(s)")

    def show(self):
        """Display the viewer"""
        plt.show()


def load_hyperspectral_data(data_folder):
    """
    Load hyperspectral cube from folder

    Args:
        data_folder: Path to folder containing ImagesStack*.png and header.json

    Returns:
        tuple: (cube, wavelengths, num_bands)
    """
    print(f"\nLoading hyperspectral data from: {data_folder}/")
    print("-" * 70)

    # Check if folder exists
    if not os.path.exists(data_folder):
        available_folders = [f for f in os.listdir('.') if os.path.isdir(f) and 'dataset' in f.lower()]
        print(f"⚠ Error: Folder '{data_folder}' not found!")
        print(f"\nAvailable dataset folders:")
        for folder in available_folders:
            print(f"  - {folder}")
        raise FileNotFoundError(f"Please set DATA_FOLDER to one of the available folders above")

    # Load header information
    header_path = os.path.join(data_folder, 'header.json')
    if not os.path.exists(header_path):
        raise FileNotFoundError(f"header.json not found in {data_folder}/")

    with open(header_path, 'r') as f:
        header = json.load(f)

    wavelengths = np.array(header['wavelength (nm)'])
    num_bands = len(wavelengths)

    print(f"✓ Header loaded")
    print(f"  Number of spectral bands: {num_bands}")
    print(f"  Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm")
    print(f"  Spectral resolution: {np.mean(np.diff(wavelengths)):.2f} nm/band")

    # Load images
    image_folder = Path(data_folder)

    # Load first image to get dimensions
    first_img = Image.open(image_folder / 'ImagesStack001.png')
    height, width = first_img.size[1], first_img.size[0]
    print(f"\nImage dimensions: {width} x {height} pixels")

    # Initialize hyperspectral cube (height, width, bands)
    cube = np.zeros((height, width, num_bands), dtype=np.uint8)

    print(f"Loading {num_bands} images...")
    # Load all images
    for i in range(1, num_bands + 1):
        img_path = image_folder / f'ImagesStack{i:03d}.png'

        if not img_path.exists():
            print(f"⚠ Warning: {img_path} not found, skipping...")
            continue

        img = Image.open(img_path).convert('L')  # Convert to grayscale
        cube[:, :, i-1] = np.array(img)

        # Progress indicator
        if i % 100 == 0 or i == num_bands:
            print(f"  Loaded {i}/{num_bands} images ({100*i/num_bands:.1f}%)")

    print(f"\n✓ Hyperspectral cube loaded: {cube.shape}")
    print(f"  Memory size: {cube.nbytes / (1024**2):.1f} MB")

    return cube, wavelengths, num_bands


def filter_wavelength_range(cube, wavelengths, wavelength_range):
    """
    Filter hyperspectral cube to only include bands within specified wavelength range

    Args:
        cube: 3D numpy array (height, width, bands)
        wavelengths: Array of wavelengths
        wavelength_range: Tuple (min_wl, max_wl) or None

    Returns:
        tuple: (filtered_cube, filtered_wavelengths, band_indices)
    """
    if wavelength_range is None:
        # No filtering - return all bands
        return cube, wavelengths, np.arange(len(wavelengths))

    min_wl, max_wl = wavelength_range

    # Find bands within range
    band_mask = (wavelengths >= min_wl) & (wavelengths <= max_wl)
    band_indices = np.where(band_mask)[0]

    if len(band_indices) == 0:
        raise ValueError(f"No bands found in wavelength range {min_wl}-{max_wl} nm")

    # Filter cube and wavelengths
    filtered_cube = cube[:, :, band_indices]
    filtered_wavelengths = wavelengths[band_indices]

    print(f"\n✓ Wavelength filtering applied:")
    print(f"  Range: {min_wl} - {max_wl} nm")
    print(f"  Selected bands: {len(band_indices)} out of {len(wavelengths)}")
    print(f"  Band indices: {band_indices[0]+1} to {band_indices[-1]+1}")
    print(f"  Actual range: {filtered_wavelengths[0]:.2f} - {filtered_wavelengths[-1]:.2f} nm")

    return filtered_cube, filtered_wavelengths, band_indices


def select_n_bands(cube, wavelengths, n_bands):
    """
    Select N evenly-spaced bands from the cube

    Args:
        cube: 3D numpy array (height, width, bands)
        wavelengths: Array of wavelengths
        n_bands: Number of bands to select

    Returns:
        tuple: (selected_cube, selected_wavelengths, selected_indices)
    """
    total_bands = cube.shape[2]

    if n_bands is None or n_bands >= total_bands:
        # No selection needed
        return cube, wavelengths, np.arange(total_bands)

    if n_bands < 1:
        raise ValueError(f"SELECT_N_BANDS must be at least 1, got {n_bands}")

    # Select evenly-spaced indices
    selected_indices = np.linspace(0, total_bands - 1, n_bands, dtype=int)

    # Remove duplicates (can happen with small ranges)
    selected_indices = np.unique(selected_indices)

    # Filter cube and wavelengths
    selected_cube = cube[:, :, selected_indices]
    selected_wavelengths = wavelengths[selected_indices]

    print(f"\n✓ Band selection applied:")
    print(f"  Requested bands: {n_bands}")
    print(f"  Selected bands: {len(selected_indices)} out of {total_bands}")
    print(f"  Selected indices: {selected_indices[0]} to {selected_indices[-1]}")
    print(f"  Wavelength range: {selected_wavelengths[0]:.2f} - {selected_wavelengths[-1]:.2f} nm")
    print(f"  Average spacing: {np.mean(np.diff(selected_wavelengths)):.2f} nm")

    return selected_cube, selected_wavelengths, selected_indices


def bin_spectral_bands(cube, wavelengths, bin_size):
    """
    Bin (average) consecutive spectral bands to reduce noise and data size

    Args:
        cube: 3D numpy array (height, width, bands)
        wavelengths: Array of wavelengths
        bin_size: Number of consecutive bands to average (e.g., 2, 3, 5)

    Returns:
        tuple: (binned_cube, binned_wavelengths)
    """
    if bin_size is None or bin_size == 1:
        # No binning needed
        return cube, wavelengths

    if bin_size < 1:
        raise ValueError(f"SPECTRAL_BINNING must be at least 1, got {bin_size}")

    height, width, num_bands = cube.shape

    # Calculate number of bins
    num_bins = num_bands // bin_size
    remainder = num_bands % bin_size

    print(f"\n✓ Spectral binning applied:")
    print(f"  Bin size: {bin_size} bands")
    print(f"  Original bands: {num_bands}")
    print(f"  Binned bands: {num_bins}")
    if remainder > 0:
        print(f"  Note: {remainder} trailing band(s) will be discarded")

    # Initialize binned cube
    binned_cube = np.zeros((height, width, num_bins), dtype=np.float32)
    binned_wavelengths = np.zeros(num_bins)

    # Perform binning
    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size

        # Average bands in this bin
        binned_cube[:, :, i] = np.mean(cube[:, :, start_idx:end_idx], axis=2)

        # Average wavelengths in this bin
        binned_wavelengths[i] = np.mean(wavelengths[start_idx:end_idx])

    # Convert back to uint8
    binned_cube = binned_cube.astype(np.uint8)

    print(f"  Wavelength range: {binned_wavelengths[0]:.2f} - {binned_wavelengths[-1]:.2f} nm")
    print(f"  Average spectral spacing: {np.mean(np.diff(binned_wavelengths)):.2f} nm")

    return binned_cube, binned_wavelengths


def bin_spatial(cube, bin_size):
    """
    Bin (average) spatial pixels to reduce image size and noise

    Args:
        cube: 3D numpy array (height, width, bands)
        bin_size: Size of spatial bins (e.g., 2 for 2x2, 4 for 4x4)

    Returns:
        Spatially binned cube
    """
    if bin_size is None or bin_size == 1:
        # No binning needed
        return cube

    if bin_size < 1:
        raise ValueError(f"SPATIAL_BINNING must be at least 1, got {bin_size}")

    height, width, num_bands = cube.shape

    # Calculate new dimensions
    new_height = height // bin_size
    new_width = width // bin_size

    # Trim to ensure even division
    trimmed_height = new_height * bin_size
    trimmed_width = new_width * bin_size

    print(f"\n✓ Spatial binning applied:")
    print(f"  Bin size: {bin_size}x{bin_size} pixels")
    print(f"  Original size: {width} x {height}")
    print(f"  Binned size: {new_width} x {new_height}")
    if trimmed_height < height or trimmed_width < width:
        print(f"  Note: Trimmed to {trimmed_width} x {trimmed_height} for even division")

    # Initialize binned cube
    binned_cube = np.zeros((new_height, new_width, num_bands), dtype=np.float32)

    # Perform spatial binning for each band
    for band_idx in range(num_bands):
        band = cube[:trimmed_height, :trimmed_width, band_idx]

        # Reshape to create bins
        reshaped = band.reshape(new_height, bin_size, new_width, bin_size)

        # Average over the bin dimensions (axis 1 and 3)
        binned_cube[:, :, band_idx] = reshaped.mean(axis=(1, 3))

    # Convert back to uint8
    binned_cube = binned_cube.astype(np.uint8)

    print(f"  Compression ratio: {(height * width) / (new_height * new_width):.2f}x")

    return binned_cube


def denoise_cube(cube, method='gaussian', strength=1.0):
    """
    Apply denoising filter to each band of the hyperspectral cube

    Args:
        cube: 3D numpy array (height, width, bands)
        method: Denoising method ('gaussian', 'median', 'bilateral')
        strength: Strength parameter (sigma for gaussian, size for median)

    Returns:
        Denoised cube
    """
    print(f"\nApplying denoising filter...")
    print(f"  Method: {method}")
    print(f"  Strength: {strength}")

    denoised_cube = np.zeros_like(cube, dtype=np.uint8)
    num_bands = cube.shape[2]

    if method == 'gaussian':
        # Gaussian blur - good for general noise reduction
        sigma = strength
        for i in range(num_bands):
            denoised_cube[:, :, i] = gaussian_filter(cube[:, :, i], sigma=sigma)
            if (i + 1) % 100 == 0 or (i + 1) == num_bands:
                print(f"  Processed {i+1}/{num_bands} bands ({100*(i+1)/num_bands:.1f}%)")

    elif method == 'median':
        # Median filter - good for salt-and-pepper noise
        size = int(2 * strength + 1)  # Convert to odd integer
        for i in range(num_bands):
            denoised_cube[:, :, i] = median_filter(cube[:, :, i], size=size)
            if (i + 1) % 100 == 0 or (i + 1) == num_bands:
                print(f"  Processed {i+1}/{num_bands} bands ({100*(i+1)/num_bands:.1f}%)")

    elif method == 'bilateral':
        # Bilateral-like filter using gaussian (approximation)
        # True bilateral filter is computationally expensive
        sigma = strength
        print("  Note: Using Gaussian approximation for bilateral filter")
        for i in range(num_bands):
            denoised_cube[:, :, i] = gaussian_filter(cube[:, :, i], sigma=sigma)
            if (i + 1) % 100 == 0 or (i + 1) == num_bands:
                print(f"  Processed {i+1}/{num_bands} bands ({100*(i+1)/num_bands:.1f}%)")

    else:
        raise ValueError(f"Unknown denoising method: {method}. Options: 'gaussian', 'median', 'bilateral'")

    print(f"✓ Denoising complete")

    return denoised_cube


def main():
    """Main function"""
    print("\n" + "="*70)
    print("HYPERSPECTRAL CUBE VIEWER")
    print("="*70)
    print(f"Selected data folder: {DATA_FOLDER}")
    if WAVELENGTH_RANGE is not None:
        print(f"Wavelength filter: {WAVELENGTH_RANGE[0]} - {WAVELENGTH_RANGE[1]} nm")
    else:
        print(f"Wavelength filter: None (viewing all bands)")
    if SELECT_N_BANDS is not None:
        print(f"Band selection: {SELECT_N_BANDS} evenly-spaced bands")
    else:
        print(f"Band selection: All bands")
    if SPECTRAL_BINNING is not None:
        print(f"Spectral binning: {SPECTRAL_BINNING} bands per bin")
    else:
        print(f"Spectral binning: Disabled")
    if SPATIAL_BINNING is not None:
        print(f"Spatial binning: {SPATIAL_BINNING}x{SPATIAL_BINNING} pixels")
    else:
        print(f"Spatial binning: Disabled")
    if DENOISE_ENABLED:
        print(f"Denoising: Enabled ({DENOISE_METHOD}, strength={DENOISE_STRENGTH})")
    else:
        print(f"Denoising: Disabled")

    try:
        # Load data
        cube, wavelengths, num_bands = load_hyperspectral_data(DATA_FOLDER)

        # Apply wavelength filtering if specified
        if WAVELENGTH_RANGE is not None:
            cube, wavelengths, band_indices = filter_wavelength_range(
                cube, wavelengths, WAVELENGTH_RANGE
            )
        else:
            band_indices = np.arange(len(wavelengths))

        # Apply band selection if specified (mutually exclusive with spectral binning)
        if SELECT_N_BANDS is not None and SPECTRAL_BINNING is not None:
            print("\n⚠ Warning: Both SELECT_N_BANDS and SPECTRAL_BINNING are set.")
            print("  Using SELECT_N_BANDS and ignoring SPECTRAL_BINNING.")
            print("  For spectral binning, set SELECT_N_BANDS = None")
            cube, wavelengths, selected_indices = select_n_bands(
                cube, wavelengths, SELECT_N_BANDS
            )
        elif SELECT_N_BANDS is not None:
            cube, wavelengths, selected_indices = select_n_bands(
                cube, wavelengths, SELECT_N_BANDS
            )
        elif SPECTRAL_BINNING is not None:
            cube, wavelengths = bin_spectral_bands(
                cube, wavelengths, SPECTRAL_BINNING
            )

        # Apply spatial binning if specified
        if SPATIAL_BINNING is not None:
            cube = bin_spatial(cube, SPATIAL_BINNING)

        # Apply denoising if enabled
        if DENOISE_ENABLED:
            cube = denoise_cube(cube, method=DENOISE_METHOD, strength=DENOISE_STRENGTH)

        # Create and show viewer
        viewer = HyperspectralViewer(cube, wavelengths, DATA_FOLDER)
        viewer.show()

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check the DATA_FOLDER variable at the top of this script.")
        return 1
    except KeyboardInterrupt:
        print("\n\nViewer closed by user.")
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())

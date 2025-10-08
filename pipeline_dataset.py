"""
Dataset Module
PyTorch dataset for pixel-wise hyperspectral classification
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import json


# Class mapping from labels.json
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


def load_label_image(label_path):
    """
    Load label image and convert RGB to class indices.

    Args:
        label_path: Path to labels.png

    Returns:
        label_array: 2D array (H x W) with class indices
    """
    label_img = Image.open(label_path).convert('RGB')
    label_rgb = np.array(label_img)
    height, width, _ = label_rgb.shape

    label_array = np.zeros((height, width), dtype=np.int64)

    for rgb_tuple, class_id in CLASS_MAPPING.items():
        mask = np.all(label_rgb == rgb_tuple, axis=2)
        label_array[mask] = class_id

    return label_array


class HyperspectralDataset(Dataset):
    """
    PyTorch Dataset for pixel-wise hyperspectral classification.

    Each sample is a 1D spectral signature (num_bands,) with corresponding class label.
    """

    def __init__(self, cube, labels, augment=False):
        """
        Args:
            cube: 3D numpy array (H x W x Bands)
            labels: 2D numpy array (H x W) with class indices
            augment: If True, apply data augmentation
        """
        self.cube = cube
        self.labels = labels
        self.augment = augment

        # Get valid pixel coordinates
        height, width, num_bands = cube.shape
        self.num_bands = num_bands

        # Create list of (row, col) for all pixels (including background)
        valid_mask = np.ones_like(labels, dtype=bool)

        self.pixel_coords = np.argwhere(valid_mask)  # (N x 2)
        self.num_samples = len(self.pixel_coords)

        # Count samples per class
        self.class_counts = np.bincount(labels[valid_mask], minlength=NUM_CLASSES)

        print(f"Dataset initialized:")
        print(f"  Total pixels: {self.num_samples}")
        print(f"  Num bands: {self.num_bands}")
        print(f"  Class distribution:")
        for i, count in enumerate(self.class_counts):
            if count > 0:
                print(f"    Class {i} ({CLASS_NAMES[i]}): {count} pixels ({count/self.num_samples*100:.2f}%)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            spectrum: 1D tensor (num_bands,)
            label: Scalar tensor (class index)
        """
        row, col = self.pixel_coords[idx]

        # Get spectral signature
        spectrum = self.cube[row, col, :].astype(np.float32)

        # Get label
        label = self.labels[row, col]

        # Normalize to [0, 1]
        spectrum = spectrum / 255.0

        # Data augmentation
        if self.augment:
            # Add small random noise
            noise = np.random.normal(0, 0.01, spectrum.shape).astype(np.float32)
            spectrum = spectrum + noise
            spectrum = np.clip(spectrum, 0, 1).astype(np.float32)

        # Ensure float32 for MPS compatibility (Apple Silicon doesn't support float64)
        return torch.from_numpy(spectrum).float(), torch.tensor(label, dtype=torch.long)

    def get_class_weights(self):
        """
        Calculate class weights for handling imbalanced dataset.

        Returns:
            weights: 1D tensor (num_classes,)
        """
        # Inverse frequency weighting
        weights = np.zeros(NUM_CLASSES)
        for i in range(NUM_CLASSES):
            if self.class_counts[i] > 0:
                weights[i] = 1.0 / self.class_counts[i]

        # Normalize
        weights = weights / weights.sum() * NUM_CLASSES

        return torch.from_numpy(weights).float()


def create_train_val_split(dataset, val_ratio=0.2, random_seed=42):
    """
    Split dataset into training and validation sets.

    Args:
        dataset: HyperspectralDataset
        val_ratio: Fraction of data for validation
        random_seed: Random seed for reproducibility

    Returns:
        train_dataset, val_dataset
    """
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size

    torch.manual_seed(random_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"\nDataset split:")
    print(f"  Training: {train_size} samples")
    print(f"  Validation: {val_size} samples")

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, batch_size=512, num_workers=4):
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader
    """
    # Disable pin_memory on MPS (Apple Silicon) as it's not supported
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    print(f"\nDataLoaders created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    return train_loader, val_loader


if __name__ == '__main__':
    # Example usage
    from pipeline_preprocess import load_hyperspectral_cube, preprocess_cube

    # Load data
    print("Loading hyperspectral cube...")
    cube, wavelengths, header = load_hyperspectral_cube('training_dataset_normalized')

    # Preprocess (optional)
    preprocess_config = {
        'wavelength_range': None,
        'select_n_bands': None,
        'spectral_binning': 2,  # Reduce to ~229 bands
        'spatial_binning': None,
        'denoise_enabled': False
    }
    cube, wavelengths = preprocess_cube(cube, wavelengths, preprocess_config)

    # Load labels
    print("\nLoading labels...")
    labels = load_label_image('Ground_Truth/labels.png')

    # Create dataset
    print("\nCreating dataset...")
    dataset = HyperspectralDataset(cube, labels, augment=True)

    # Split dataset
    train_dataset, val_dataset = create_train_val_split(dataset, val_ratio=0.2)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, batch_size=512
    )

    # Test loading a batch
    print("\nTesting batch loading...")
    for batch_spectra, batch_labels in train_loader:
        print(f"Batch spectra shape: {batch_spectra.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        break

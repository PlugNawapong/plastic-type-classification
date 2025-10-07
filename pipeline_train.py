"""
Training Pipeline
Train 1D CNN for hyperspectral plastic classification
"""

import os
# Fix OpenMP library conflict on macOS (must be set before importing torch)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from pipeline_preprocess import load_hyperspectral_cube, preprocess_cube, bin_spatial_labels
from pipeline_dataset import (
    load_label_image, HyperspectralDataset,
    create_train_val_split, create_dataloaders, CLASS_NAMES
)
from pipeline_model import create_model, count_parameters


# Module-level CONFIG that can be overridden by run_pipeline_config.py
CONFIG = None


class Trainer:
    """Training manager for hyperspectral classification."""

    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device (Apple Silicon M4 GPU support via MPS)
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"✓ Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print(f"⚠ Using CPU (GPU not available)")

        self.model.to(self.device)

        # Loss function (with class weights for imbalanced data)
        if config.get('use_class_weights'):
            class_weights = config['class_weights'].to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.best_epoch = 0

        # Output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for spectra, labels in pbar:
            spectra = spectra.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(spectra)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * spectra.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """Validate model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for spectra, labels in tqdm(self.val_loader, desc='Validation'):
                spectra = spectra.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(spectra)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * spectra.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)

    def train(self, num_epochs):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\n{'='*60}")
        print(f"Starting training on {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            print("-" * 40)

            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_acc)

            # Print epoch summary
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth')
                print(f"✓ Best model saved (Val Acc: {val_acc:.2f}%)")

            # Save latest model
            self.save_checkpoint('latest_model.pth')

            print()

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        print(f"{'='*60}\n")

        # Save training history
        self.save_history()
        self.plot_history()

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'config': self.config
        }
        torch.save(checkpoint, self.output_dir / filename)

    def save_history(self):
        """Save training history to JSON."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    def plot_history(self):
        """Plot training history."""
        epochs = range(1, len(self.train_losses) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(epochs, self.train_accs, 'b-', label='Train Acc')
        ax2.plot(epochs, self.val_accs, 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150)
        print(f"Training plots saved to {self.output_dir / 'training_history.png'}")


def main():
    """Main training pipeline."""

    global CONFIG

    # =====================================================================
    # CONFIGURATION
    # =====================================================================

    # Use module-level CONFIG if set (by run_pipeline_config.py), otherwise use defaults
    if CONFIG is None:
        CONFIG = {
            # Data
            'data_folder': 'training_dataset_normalized',
            'label_path': 'Ground_Truth/labels.png',

            # Preprocessing
            'preprocess': {
                'wavelength_range': None,  # e.g., (450, 700)
                'select_n_bands': None,    # e.g., 100
                'spectral_binning': 2,     # Average 2 consecutive bands
                'spatial_binning': None,   # e.g., 2
                'denoise_enabled': False,
                'denoise_method': 'gaussian',
                'denoise_strength': 1.0
            },

            # Dataset
            'ignore_background': True,
            'augment': True,
            'val_ratio': 0.2,
            'batch_size': 512,
            'num_workers': 4,

            # Model
            'model_type': 'resnet',  # 'cnn' or 'resnet'
            'dropout_rate': 0.3,

            # Training
            'num_epochs': 50,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'use_class_weights': True,

            # Output
            'output_dir': 'output/training'
        }

    # =====================================================================
    # LOAD DATA
    # =====================================================================

    print("="*60)
    print("HYPERSPECTRAL PLASTIC CLASSIFICATION - TRAINING PIPELINE")
    print("="*60)

    print("\n[1/6] Loading hyperspectral cube...")
    cube, wavelengths, header = load_hyperspectral_cube(CONFIG['data_folder'])

    print("\n[2/6] Preprocessing cube...")
    cube, wavelengths = preprocess_cube(cube, wavelengths, CONFIG['preprocess'])

    print("\n[3/6] Loading labels...")
    labels = load_label_image(CONFIG['label_path'])

    # Apply spatial binning to labels if it was applied to cube
    if CONFIG['preprocess'].get('spatial_binning'):
        labels = bin_spatial_labels(labels, CONFIG['preprocess']['spatial_binning'])

    print("\n[4/6] Creating dataset...")
    dataset = HyperspectralDataset(
        cube, labels,
        ignore_background=CONFIG['ignore_background'],
        augment=CONFIG['augment']
    )

    # Get class weights
    class_weights = dataset.get_class_weights()
    CONFIG['class_weights'] = class_weights

    # Split dataset
    train_dataset, val_dataset = create_train_val_split(
        dataset, val_ratio=CONFIG['val_ratio']
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers']
    )

    # =====================================================================
    # CREATE MODEL
    # =====================================================================

    print("\n[5/6] Creating model...")
    num_bands = cube.shape[2]
    model = create_model(
        num_bands=num_bands,
        num_classes=11,
        model_type=CONFIG['model_type'],
        dropout_rate=CONFIG['dropout_rate']
    )

    print(f"Model type: {CONFIG['model_type']}")
    print(f"Total parameters: {count_parameters(model):,}")

    # =====================================================================
    # TRAIN MODEL
    # =====================================================================

    print("\n[6/6] Training model...")
    trainer = Trainer(model, train_loader, val_loader, CONFIG)
    trainer.train(num_epochs=CONFIG['num_epochs'])

    print("\nTraining pipeline completed successfully!")
    print(f"Output directory: {CONFIG['output_dir']}")


if __name__ == '__main__':
    main()

"""
Training Pipeline for MacBook Air M4
Optimized for Apple Silicon MPS (Metal Performance Shaders)
"""
import os
# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from pipeline_preprocess import load_hyperspectral_cube, preprocess_cube, bin_spatial_labels
from pipeline_dataset import (
    load_label_image, HyperspectralDataset,
    create_train_val_split, create_dataloaders, CLASS_NAMES
)
from pipeline_model import create_model, count_parameters


# =====================================================================
# CONFIGURATION FOR MACBOOK AIR M4
# =====================================================================
CONFIG = {
    # Data paths - Base directory: DeepLearning_Plastics
    'data_folder': 'training_dataset_normalized',
    'label_path': 'Ground_Truth/labels.png',
    'output_dir': 'outputs/local_m4',
    
    # Preprocessing
    'preprocess': {
        'wavelength_range': (450, 700),  # nm
        'spatial_binning': 2,  # None or int (e.g., 2 for 2x2 binning)
        'spectral_binning': 5,  # None or int
        'smoothing': False,
        'normalize': True,
    },
    
    # Model architecture
    'model_type': 'spectral_cnn',  # Options: 'spectral_cnn', 'hybrid_sn', 'resnet1d', 'attention_net', 'deep_cnn'
    'dropout_rate': 0.5,
    
    # Training parameters (optimized for M4)
    'batch_size': 64,  # M4 can handle moderate batch sizes
    'num_epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'val_ratio': 0.2,
    
    # Data augmentation
    'augment': True,
    
    # Class balancing
    'use_class_weights': True,
    
    # System (optimized for M4)
    'num_workers': 0,  # Use 0 for MPS to avoid multiprocessing issues
    'device': 'mps',  # Force MPS for M4
    
    # Early stopping
    'early_stopping_patience': 15,
    
    # Save options
    'save_best_only': True,
    'save_frequency': 10,  # Save checkpoint every N epochs
}


class Trainer:
    """Training manager optimized for MacBook Air M4."""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Force MPS device for M4
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"✓ Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device('cpu')
            print(f"⚠ MPS not available, using CPU")
        
        self.model.to(self.device)
        
        # Loss function
        if config.get('use_class_weights') and 'class_weights' in config:
            weights = torch.FloatTensor(config['class_weights']).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
            print(f"✓ Using weighted loss with class weights")
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
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # Output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for spectra, labels in pbar:
            spectra = spectra.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(spectra)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for spectra, labels in tqdm(self.val_loader, desc='Validation', leave=False):
                spectra = spectra.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(spectra)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs):
        """Main training loop with early stopping."""
        print(f"\nStarting training on {self.device}...")
        print(f"Model: {self.config['model_type']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                if self.config.get('save_best_only', True):
                    self.save_checkpoint(f'best_model_{self.timestamp}.pth')
                    print(f"  ✓ New best model saved (Val Acc: {val_acc:.2f}%)")
            else:
                self.epochs_without_improvement += 1
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.get('save_frequency', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}_{self.timestamp}.pth')
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.get('early_stopping_patience', 15):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
                break
        
        # Save final results
        self.save_history()
        self.plot_history()
        
        print("\n" + "="*60)
        print(f"Training completed!")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': len(self.train_losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
        }
        path = self.output_dir / filename
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
        }
        np.save(self.output_dir / f'history_{self.timestamp}.npy', history)
    
    def plot_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(self.train_losses, label='Train')
        ax1.plot(self.val_losses, label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.train_accs, label='Train')
        ax2.plot(self.val_accs, label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'training_history_{self.timestamp}.png', dpi=150)
        plt.close()


def main():
    """Main training pipeline for MacBook Air M4."""
    
    print("="*60)
    print("HYPERSPECTRAL CLASSIFICATION - LOCAL TRAINING (M4)")
    print("="*60)
    
    # Load data
    print("\n[1/6] Loading hyperspectral cube...")
    cube, wavelengths, header = load_hyperspectral_cube(CONFIG['data_folder'])
    print(f"  Cube shape: {cube.shape}")
    
    print("\n[2/6] Preprocessing...")
    cube, wavelengths = preprocess_cube(cube, wavelengths, CONFIG['preprocess'])
    print(f"  Processed shape: {cube.shape}")
    print(f"  Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    
    print("\n[3/6] Loading labels...")
    labels = load_label_image(CONFIG['label_path'])
    
    if CONFIG['preprocess'].get('spatial_binning'):
        bin_size = CONFIG['preprocess']['spatial_binning']
        labels = bin_spatial_labels(labels, bin_size)
        print(f"  Applied {bin_size}x{bin_size} spatial binning to labels")
    
    print("\n[4/6] Creating dataset...")
    dataset = HyperspectralDataset(cube, labels, augment=CONFIG['augment'])
    print(f"  Total samples: {len(dataset)}")
    print(f"  Classes: {CLASS_NAMES}")
    
    # Get class weights
    class_weights = dataset.get_class_weights()
    CONFIG['class_weights'] = class_weights
    
    # Split dataset
    train_dataset, val_dataset = create_train_val_split(
        dataset, val_ratio=CONFIG['val_ratio']
    )
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers']
    )
    
    print("\n[5/6] Creating model...")
    num_bands = cube.shape[2]
    model = create_model(
        num_bands=num_bands,
        num_classes=len(CLASS_NAMES),
        model_type=CONFIG['model_type'],
        dropout_rate=CONFIG['dropout_rate']
    )
    print(f"  Model: {CONFIG['model_type']}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    print("\n[6/6] Training...")
    trainer = Trainer(model, train_loader, val_loader, CONFIG)
    trainer.train(num_epochs=CONFIG['num_epochs'])


if __name__ == '__main__':
    main()

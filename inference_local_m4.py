"""
Inference Pipeline for MacBook Air M4
Optimized for Apple Silicon MPS (Metal Performance Shaders)
Performs prediction on new hyperspectral data
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

from pipeline_preprocess import load_hyperspectral_cube, preprocess_cube
from pipeline_dataset import CLASS_NAMES, CLASS_MAPPING
from pipeline_model import create_model


# =====================================================================
# CONFIGURATION FOR MACBOOK AIR M4
# =====================================================================
CONFIG = {
    # Model checkpoint
    'checkpoint_path': '/Users/nawapong/Projects/plastic-type-classification/outputs/local_m4/best_model_20251010_120000.pth',
    
    # Inference data paths
    'data_folder': '/Users/nawapong/Projects/plastic-type-classification/Inference_dataset1_normalized',
    'output_dir': '/Users/nawapong/Projects/plastic-type-classification/outputs/inference_local_m4',
    
    # Preprocessing (should match training)
    'preprocess': {
        'wavelength_range': (450, 1000),
        'spatial_binning': None,
        'spectral_binning': None,
        'smoothing': False,
        'normalize': True,
    },
    
    # Inference settings (optimized for M4)
    'batch_size': 512,  # Larger batches for inference (no gradients)
    'device': 'mps',
    
    # Visualization
    'save_visualization': True,
    'save_confidence_map': True,
}


# Inverse class mapping (class_id -> RGB color)
CLASS_TO_RGB = {v: k for k, v in CLASS_MAPPING.items()}


class InferenceEngine:
    """Inference engine optimized for MacBook Air M4."""
    
    def __init__(self, checkpoint_path, config):
        self.config = config
        
        # Setup device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"✓ Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device('cpu')
            print(f"⚠ MPS not available, using CPU")
        
        # Load checkpoint
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract training config
        self.train_config = checkpoint.get('config', {})
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        print(f"✓ Model trained to {self.best_val_acc:.2f}% validation accuracy")
        print(f"✓ Model type: {self.train_config.get('model_type', 'unknown')}")
        
        # Create model
        self.num_bands = None  # Will be set after loading data
        self.model = None
        self.checkpoint = checkpoint
        
        # Output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _initialize_model(self, num_bands):
        """Initialize model with correct number of bands."""
        if self.model is not None:
            return
        
        self.num_bands = num_bands
        self.model = create_model(
            num_bands=num_bands,
            num_classes=len(CLASS_NAMES),
            model_type=self.train_config.get('model_type', 'spectral_cnn'),
            dropout_rate=self.train_config.get('dropout_rate', 0.5)
        )
        
        # Load weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model initialized with {num_bands} spectral bands")
    
    def predict_cube(self, cube, batch_size=512):
        """
        Perform inference on entire hyperspectral cube.
        
        Args:
            cube: Hyperspectral cube (H, W, Bands)
            batch_size: Batch size for inference
        
        Returns:
            prediction_map: Predicted class for each pixel (H, W)
            confidence_map: Confidence score for each pixel (H, W)
        """
        # Initialize model if needed
        if self.model is None:
            self._initialize_model(cube.shape[2])
        
        h, w, bands = cube.shape
        
        # Reshape cube to (H*W, Bands)
        pixels = cube.reshape(-1, bands)
        num_pixels = pixels.shape[0]
        
        # Prepare outputs
        predictions = np.zeros(num_pixels, dtype=np.int32)
        confidences = np.zeros(num_pixels, dtype=np.float32)
        
        print(f"\nRunning inference on {num_pixels:,} pixels...")
        
        # Process in batches
        with torch.no_grad():
            for i in tqdm(range(0, num_pixels, batch_size), desc='Inference'):
                batch_end = min(i + batch_size, num_pixels)
                batch = pixels[i:batch_end]
                
                # Convert to tensor
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                # Forward pass
                outputs = self.model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
                
                # Get predictions and confidence
                conf, pred = probs.max(dim=1)
                
                predictions[i:batch_end] = pred.cpu().numpy()
                confidences[i:batch_end] = conf.cpu().numpy()
        
        # Reshape back to image
        prediction_map = predictions.reshape(h, w)
        confidence_map = confidences.reshape(h, w)
        
        return prediction_map, confidence_map
    
    def create_rgb_visualization(self, prediction_map):
        """Convert prediction map to RGB image."""
        h, w = prediction_map.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in range(len(CLASS_NAMES)):
            if class_id in CLASS_TO_RGB:
                mask = prediction_map == class_id
                rgb_image[mask] = CLASS_TO_RGB[class_id]
        
        return rgb_image
    
    def save_results(self, prediction_map, confidence_map, rgb_image):
        """Save inference results."""
        # Save prediction map as numpy array
        np.save(self.output_dir / f'prediction_map_{self.timestamp}.npy', prediction_map)
        np.save(self.output_dir / f'confidence_map_{self.timestamp}.npy', confidence_map)
        
        # Save RGB visualization
        rgb_pil = Image.fromarray(rgb_image)
        rgb_pil.save(self.output_dir / f'prediction_rgb_{self.timestamp}.png')
        
        # Save confidence map visualization
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_image)
        plt.title('Predicted Classes')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Confidence')
        plt.title('Prediction Confidence')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'inference_results_{self.timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Results saved to: {self.output_dir}")
    
    def print_statistics(self, prediction_map, confidence_map):
        """Print inference statistics."""
        print("\n" + "="*60)
        print("INFERENCE STATISTICS")
        print("="*60)
        
        # Class distribution
        print("\nPredicted class distribution:")
        unique, counts = np.unique(prediction_map, return_counts=True)
        total_pixels = prediction_map.size
        
        for class_id, count in zip(unique, counts):
            if class_id < len(CLASS_NAMES):
                percentage = 100.0 * count / total_pixels
                print(f"  {CLASS_NAMES[class_id]:20s}: {count:8,} pixels ({percentage:5.2f}%)")
        
        # Confidence statistics
        print(f"\nConfidence statistics:")
        print(f"  Mean confidence: {confidence_map.mean():.4f}")
        print(f"  Median confidence: {np.median(confidence_map):.4f}")
        print(f"  Min confidence: {confidence_map.min():.4f}")
        print(f"  Max confidence: {confidence_map.max():.4f}")
        
        # Low confidence pixels
        low_conf_threshold = 0.5
        low_conf_pixels = (confidence_map < low_conf_threshold).sum()
        low_conf_pct = 100.0 * low_conf_pixels / total_pixels
        print(f"  Pixels with confidence < {low_conf_threshold}: {low_conf_pixels:,} ({low_conf_pct:.2f}%)")
    
    def run_inference(self, cube):
        """Run complete inference pipeline."""
        print("\n" + "="*60)
        print("RUNNING INFERENCE")
        print("="*60)
        
        # Perform prediction
        prediction_map, confidence_map = self.predict_cube(
            cube, 
            batch_size=self.config['batch_size']
        )
        
        # Create visualization
        print("\nCreating visualizations...")
        rgb_image = self.create_rgb_visualization(prediction_map)
        
        # Print statistics
        self.print_statistics(prediction_map, confidence_map)
        
        # Save results
        if self.config.get('save_visualization', True):
            self.save_results(prediction_map, confidence_map, rgb_image)
        
        print("\n" + "="*60)
        print("INFERENCE COMPLETED")
        print("="*60)
        
        return prediction_map, confidence_map, rgb_image


def main():
    """Main inference pipeline for MacBook Air M4."""
    
    print("="*60)
    print("HYPERSPECTRAL INFERENCE - LOCAL (M4)")
    print("="*60)
    
    # Load data
    print("\n[1/3] Loading hyperspectral cube...")
    cube, wavelengths, header = load_hyperspectral_cube(CONFIG['data_folder'])
    print(f"  Cube shape: {cube.shape}")
    
    # Preprocess
    print("\n[2/3] Preprocessing...")
    cube, wavelengths = preprocess_cube(cube, wavelengths, CONFIG['preprocess'])
    print(f"  Processed shape: {cube.shape}")
    print(f"  Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    
    # Run inference
    print("\n[3/3] Performing inference...")
    engine = InferenceEngine(CONFIG['checkpoint_path'], CONFIG)
    prediction_map, confidence_map, rgb_image = engine.run_inference(cube)
    
    print(f"\nDone! Check results in: {CONFIG['output_dir']}")


if __name__ == '__main__':
    main()

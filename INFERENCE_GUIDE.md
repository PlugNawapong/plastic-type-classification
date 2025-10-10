# Hyperspectral Material Classification - Inference Guide

## Overview
This guide covers how to use trained models to perform inference (prediction) on new hyperspectral data using:
1. **Local inference** on MacBook Air M4
2. **Cloud inference** on Google Colab

## Quick Start

### Local Inference (MacBook Air M4)
```bash
python inference_local_m4.py
```

### Cloud Inference (Google Colab)
1. Upload `inference_colab.ipynb` to Google Colab
2. Mount Google Drive
3. Update paths to your trained model and data
4. Run all cells

## Prerequisites

### You Need:
1. ✅ A trained model checkpoint (`.pth` file)
2. ✅ New hyperspectral data for inference (normalized)
3. ✅ Same preprocessing settings as training

### Important Notes:
- **Preprocessing must match training** - Use the same wavelength range, binning, etc.
- **Model architecture must match** - The checkpoint contains model type info
- **Normalized data recommended** - Use the normalized datasets for best results

## Local Inference (MacBook Air M4)

### Configuration

Edit `inference_local_m4.py`:

```python
CONFIG = {
    # Path to trained model
    'checkpoint_path': 'outputs/local_m4/best_model_20251010_120000.pth',
    
    # Path to inference data (normalized)
    'data_folder': 'Inference_dataset1_normalized',
    
    # Output directory
    'output_dir': 'outputs/inference_local_m4',
    
    # Preprocessing (MUST MATCH TRAINING)
    'preprocess': {
        'wavelength_range': (450, 1000),
        'spatial_binning': None,
        'spectral_binning': None,
        'smoothing': False,
        'normalize': True,
    },
    
    # Inference settings
    'batch_size': 512,  # Larger for inference (no gradients)
    'device': 'mps',    # Use MPS for M4
}
```

### Running Inference

```bash
python inference_local_m4.py
```

### Output Files

The script generates:
```
outputs/inference_local_m4/
├── prediction_map_20251010_120000.npy      # Predicted class IDs (H, W)
├── confidence_map_20251010_120000.npy      # Prediction confidence (H, W)
├── prediction_rgb_20251010_120000.png      # RGB visualization
└── inference_results_20251010_120000.png   # Combined visualization
```

### Performance (M4)
- ~512 pixels/batch
- ~1-2 seconds per 10K pixels
- Total time: ~30 seconds for 512x512 image

## Cloud Inference (Google Colab)

### Setup

1. **Upload to Google Drive**
   ```
   MyDrive/plastic-type-classification/
   ├── outputs/colab/
   │   └── best_model_20251010_120000.pth  # Trained model
   ├── Inference_dataset1_normalized/      # Data to predict
   ├── pipeline_preprocess.py
   ├── pipeline_dataset.py
   ├── pipeline_model.py
   └── inference_colab.ipynb
   ```

2. **Update Paths in Notebook**
   ```python
   PROJECT_ROOT = '/content/drive/MyDrive/plastic-type-classification'
   CHECKPOINT_PATH = f'{PROJECT_ROOT}/outputs/colab/best_model_20251010_120000.pth'
   INFERENCE_DATA_FOLDER = f'{PROJECT_ROOT}/Inference_dataset1_normalized'
   ```

3. **Run All Cells**
   - Runtime → Run all (Ctrl/Cmd+F9)

### Batch Inference

To process multiple datasets at once, use the batch inference cell:

```python
INFERENCE_DATASETS = [
    f'{PROJECT_ROOT}/Inference_dataset1_normalized',
    f'{PROJECT_ROOT}/Inference_dataset2_normalized',
    f'{PROJECT_ROOT}/Inference_dataset3_normalized',
]
```

### Performance (A100)
- ~1024 pixels/batch
- ~0.5-1 second per 10K pixels
- Total time: ~10-15 seconds for 512x512 image

## Understanding Results

### Prediction Map
- **Format**: NumPy array (H, W) with integer class IDs
- **Values**: 0-10 (11 classes)
- **Usage**: `prediction_map[y, x]` gives class ID at pixel (x, y)

```python
import numpy as np

# Load prediction map
pred_map = np.load('outputs/inference_local_m4/prediction_map_20251010_120000.npy')

# Check specific pixel
class_id = pred_map[100, 200]
print(f"Pixel at (200, 100): {CLASS_NAMES[class_id]}")

# Get class distribution
unique, counts = np.unique(pred_map, return_counts=True)
for class_id, count in zip(unique, counts):
    print(f"{CLASS_NAMES[class_id]}: {count} pixels")
```

### Confidence Map
- **Format**: NumPy array (H, W) with float values
- **Range**: 0.0 to 1.0 (0% to 100% confidence)
- **Interpretation**: Higher values = more confident predictions

```python
# Load confidence map
conf_map = np.load('outputs/inference_local_m4/confidence_map_20251010_120000.npy')

# Find low confidence areas
low_confidence = conf_map < 0.5
print(f"Low confidence pixels: {low_confidence.sum()}")

# Average confidence per class
for class_id in range(len(CLASS_NAMES)):
    mask = pred_map == class_id
    if mask.any():
        avg_conf = conf_map[mask].mean()
        print(f"{CLASS_NAMES[class_id]}: {avg_conf:.4f}")
```

### RGB Visualization
- **Format**: PNG image with RGB colors
- **Purpose**: Visual inspection of predictions
- **Colors**: Match the ground truth label colors

## Statistics Output

The inference scripts print detailed statistics:

### Class Distribution
```
Predicted class distribution:
  PET                 :   45,231 pixels (17.28%)
  HDPE                :   38,567 pixels (14.74%)
  PVC                 :   29,834 pixels (11.40%)
  ...
```

### Confidence Statistics
```
Confidence statistics:
  Mean confidence: 0.8723
  Median confidence: 0.9156
  Min confidence: 0.3421
  Max confidence: 0.9987
  Pixels with confidence < 0.5: 12,456 (4.76%)
```

## Advanced Usage

### 1. Load and Use Predictions

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load results
pred_map = np.load('outputs/inference_local_m4/prediction_map_20251010_120000.npy')
conf_map = np.load('outputs/inference_local_m4/confidence_map_20251010_120000.npy')
rgb_img = Image.open('outputs/inference_local_m4/prediction_rgb_20251010_120000.png')

# Filter by confidence threshold
threshold = 0.7
high_conf_mask = conf_map >= threshold

# Get high-confidence predictions only
filtered_pred = pred_map.copy()
filtered_pred[~high_conf_mask] = -1  # Mark low confidence as -1

# Visualize
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(rgb_img)
plt.title('All Predictions')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(conf_map, cmap='viridis')
plt.colorbar(label='Confidence')
plt.title('Confidence Map')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(filtered_pred, cmap='tab10')
plt.title(f'High Confidence (>{threshold})')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 2. Export to GeoTIFF

```python
from osgeo import gdal, osr
import numpy as np

def export_geotiff(prediction_map, output_path, geo_transform=None, projection=None):
    """Export prediction map as GeoTIFF."""
    driver = gdal.GetDriverByName('GTiff')
    h, w = prediction_map.shape
    
    dataset = driver.Create(output_path, w, h, 1, gdal.GDT_Byte)
    
    if geo_transform:
        dataset.SetGeoTransform(geo_transform)
    if projection:
        dataset.SetProjection(projection)
    
    dataset.GetRasterBand(1).WriteArray(prediction_map)
    dataset.FlushCache()
    dataset = None
    
    print(f"Exported to: {output_path}")

# Use it
pred_map = np.load('outputs/inference_local_m4/prediction_map_20251010_120000.npy')
export_geotiff(pred_map, 'outputs/prediction.tif')
```

### 3. Calculate Material Percentages

```python
def calculate_material_percentages(prediction_map, pixel_size_mm=1.0):
    """Calculate material coverage."""
    unique, counts = np.unique(prediction_map, return_counts=True)
    total_pixels = prediction_map.size
    pixel_area_mm2 = pixel_size_mm ** 2
    
    results = []
    for class_id, count in zip(unique, counts):
        if class_id < len(CLASS_NAMES):
            percentage = 100.0 * count / total_pixels
            area_mm2 = count * pixel_area_mm2
            
            results.append({
                'material': CLASS_NAMES[class_id],
                'pixels': count,
                'percentage': percentage,
                'area_mm2': area_mm2,
            })
    
    return results

# Use it
pred_map = np.load('outputs/inference_local_m4/prediction_map_20251010_120000.npy')
results = calculate_material_percentages(pred_map, pixel_size_mm=0.5)

for r in sorted(results, key=lambda x: x['percentage'], reverse=True):
    print(f"{r['material']:20s}: {r['percentage']:6.2f}% ({r['area_mm2']:8.1f} mm²)")
```

### 4. Compare Multiple Inference Results

```python
import numpy as np
import matplotlib.pyplot as plt

# Load multiple predictions
pred1 = np.load('outputs/inference_local_m4/dataset1_prediction.npy')
pred2 = np.load('outputs/inference_local_m4/dataset2_prediction.npy')
pred3 = np.load('outputs/inference_local_m4/dataset3_prediction.npy')

# Compare distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
predictions = [pred1, pred2, pred3]
titles = ['Dataset 1', 'Dataset 2', 'Dataset 3']

for ax, pred, title in zip(axes, predictions, titles):
    unique, counts = np.unique(pred, return_counts=True)
    labels = [CLASS_NAMES[c] for c in unique if c < len(CLASS_NAMES)]
    
    ax.bar(labels, counts)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Material Class')
    ax.set_ylabel('Pixel Count')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### 5. Uncertainty Analysis

```python
def analyze_uncertainty(prediction_map, confidence_map, threshold=0.5):
    """Analyze prediction uncertainty."""
    # High vs low confidence areas
    high_conf = confidence_map >= threshold
    low_conf = confidence_map < threshold
    
    print(f"High confidence pixels: {high_conf.sum():,} ({100*high_conf.sum()/confidence_map.size:.2f}%)")
    print(f"Low confidence pixels: {low_conf.sum():,} ({100*low_conf.sum()/confidence_map.size:.2f}%)")
    
    # Class-wise confidence
    print("\nAverage confidence per class:")
    for class_id in range(len(CLASS_NAMES)):
        mask = prediction_map == class_id
        if mask.any():
            avg_conf = confidence_map[mask].mean()
            std_conf = confidence_map[mask].std()
            print(f"  {CLASS_NAMES[class_id]:20s}: {avg_conf:.4f} ± {std_conf:.4f}")
    
    # Spatial uncertainty map
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.imshow(confidence_map, cmap='RdYlGn', vmin=0, vmax=1)
    ax1.set_title('Confidence Map')
    ax1.axis('off')
    
    ax2.imshow(low_conf, cmap='Reds')
    ax2.set_title(f'Low Confidence Areas (<{threshold})')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

# Use it
pred_map = np.load('outputs/inference_local_m4/prediction_map_20251010_120000.npy')
conf_map = np.load('outputs/inference_local_m4/confidence_map_20251010_120000.npy')
analyze_uncertainty(pred_map, conf_map, threshold=0.7)
```

## Troubleshooting

### Issue: Model dimension mismatch

**Error**: `size mismatch for fc1.weight`

**Solution**: Ensure preprocessing matches training
```python
# Check your training config
checkpoint = torch.load('model.pth')
print(checkpoint['config']['preprocess'])

# Use same settings for inference
CONFIG['preprocess'] = checkpoint['config']['preprocess']
```

### Issue: Out of memory during inference

**Solution**: Reduce batch size
```python
# Local (M4)
'batch_size': 256,  # Instead of 512

# Colab
'batch_size': 512,  # Instead of 1024
```

### Issue: Low confidence predictions

**Possible causes**:
1. Different data distribution than training
2. Poor lighting conditions
3. Unknown materials
4. Preprocessing mismatch

**Solutions**:
```python
# Check preprocessing match
print("Training config:", checkpoint['config']['preprocess'])
print("Inference config:", CONFIG['preprocess'])

# Visualize low confidence areas
low_conf_mask = confidence_map < 0.5
plt.imshow(low_conf_mask)
plt.title('Low Confidence Pixels')
plt.show()

# Consider retraining with augmented data
```

### Issue: Checkpoint file not found

**Solution**: Check paths
```python
from pathlib import Path

checkpoint_path = Path('outputs/local_m4/best_model_20251010_120000.pth')
print(f"Exists: {checkpoint_path.exists()}")
print(f"Absolute path: {checkpoint_path.absolute()}")

# List all checkpoints
output_dir = Path('outputs/local_m4')
checkpoints = list(output_dir.glob('*.pth'))
print(f"Available checkpoints: {checkpoints}")
```

## Performance Benchmarks

### MacBook Air M4 (8GB RAM)
| Image Size | Batch Size | Time | Memory |
|------------|------------|------|--------|
| 256x256 | 512 | ~8s | ~2GB |
| 512x512 | 512 | ~30s | ~3GB |
| 1024x1024 | 512 | ~2min | ~4GB |

### Google Colab A100 (40GB)
| Image Size | Batch Size | Time | Memory |
|------------|------------|------|--------|
| 256x256 | 1024 | ~2s | ~5GB |
| 512x512 | 1024 | ~8s | ~6GB |
| 1024x1024 | 1024 | ~30s | ~8GB |

## Best Practices

### 1. Preprocessing Consistency
✅ **Always use the same preprocessing as training**
```python
# Save preprocessing config during training
checkpoint['preprocess_config'] = CONFIG['preprocess']

# Load and use during inference
CONFIG['preprocess'] = checkpoint['preprocess_config']
```

### 2. Confidence Thresholding
✅ **Filter predictions by confidence**
```python
# Only keep high-confidence predictions
threshold = 0.7
reliable_predictions = prediction_map.copy()
reliable_predictions[confidence_map < threshold] = -1  # Unknown
```

### 3. Batch Processing
✅ **Process multiple datasets efficiently**
```python
datasets = ['dataset1', 'dataset2', 'dataset3']
for dataset in datasets:
    # Load, process, save
    pass
```

### 4. Result Validation
✅ **Sanity check results**
```python
# Check class distribution
unique, counts = np.unique(pred_map, return_counts=True)
if len(unique) < 3:  # Too few classes
    print("⚠ Warning: Only predicted few classes")

# Check confidence
if conf_map.mean() < 0.5:
    print("⚠ Warning: Low average confidence")
```

## Next Steps

After inference:
1. ✅ Validate predictions visually
2. ✅ Check confidence statistics
3. ✅ Compare with ground truth (if available)
4. ✅ Analyze misclassifications
5. ✅ Consider ensemble methods for uncertain areas
6. ✅ Export results for further analysis

## References

- Training Guide: `TRAINING_GUIDE.md`
- Model architectures: `pipeline_model.py`
- Preprocessing: `pipeline_preprocess.py`
- Dataset handling: `pipeline_dataset.py`

Happy predicting! 🎯

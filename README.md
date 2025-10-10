# Hyperspectral Plastic Classification - Complete Pipeline

A complete machine learning pipeline for hyperspectral material classification optimized for **MacBook Air M4 (Apple Silicon)** and **Google Colab (A100 GPU)**.

## 🚀 Quick Start

### Training (Local - M4)
```bash
python train_local_m4.py
```

### Training (Colab)
1. Upload `train_colab.ipynb` to Google Colab
2. Mount Google Drive
3. Run all cells

### Inference (Local - M4)
```bash
python inference_local_m4.py
```

Or use the quick script:
```bash
python quick_inference.py \
  --checkpoint outputs/local_m4/best_model.pth \
  --data Inference_dataset1_normalized \
  --output outputs/results
```

### Inference (Colab)
1. Upload `inference_colab.ipynb` to Google Colab
2. Update paths to trained model
3. Run all cells

## 📁 Project Structure

```
plastic-type-classification/
├── Training Files
│   ├── train_local_m4.py          # Training for MacBook Air M4
│   ├── train_colab.ipynb          # Training for Google Colab
│   └── pipeline_train.py          # Original training pipeline
│
├── Inference Files
│   ├── inference_local_m4.py      # Inference for MacBook Air M4
│   ├── inference_colab.ipynb      # Inference for Google Colab
│   ├── quick_inference.py         # Command-line inference tool
│   └── pipeline_inference.py      # Original inference pipeline
│
├── Core Pipeline
│   ├── pipeline_model.py          # Model architectures (5 models)
│   ├── pipeline_dataset.py        # Dataset handling
│   ├── pipeline_preprocess.py     # Data preprocessing
│   └── pipeline_normalize.py      # Normalization
│
├── Documentation
│   ├── README.md                  # This file
│   ├── TRAINING_GUIDE.md          # Detailed training guide
│   └── INFERENCE_GUIDE.md         # Detailed inference guide
│
├── Data
│   ├── training_dataset_normalized/
│   ├── Inference_dataset1_normalized/
│   ├── Inference_dataset2_normalized/
│   ├── Inference_dataset3_normalized/
│   └── Ground_Truth/
│       └── labels.json
│
└── Outputs
    ├── local_m4/                  # Local training results
    ├── colab/                     # Colab training results
    └── inference_local_m4/        # Inference results
```

## 🎯 Available Models

| Model | Speed | Accuracy | Parameters | Best For |
|-------|-------|----------|------------|----------|
| `spectral_cnn` | ⚡⚡⚡ Fast | Good | ~500K | Quick baseline |
| `hybrid_sn` | ⚡⚡ Medium | Very Good | ~800K | Spatial features |
| `resnet1d` | ⚡⚡ Medium | Very Good | ~1M | Deep features |
| `attention_net` ⭐ | ⚡⚡ Medium | Excellent | ~700K | **Recommended** |
| `deep_cnn` | ⚡ Slow | Excellent | ~2M | Maximum accuracy |

## 🔧 Installation

### Local (MacBook M4)
```bash
# Install PyTorch for Apple Silicon
pip install torch torchvision

# Install other dependencies
pip install spectral numpy matplotlib scikit-learn pillow tqdm
```

### Google Colab
```bash
# Colab has PyTorch pre-installed
!pip install spectral tqdm matplotlib scikit-learn pillow
```

## 📊 Training

### Configuration Options

**Model Selection**
```python
'model_type': 'attention_net'  # Change to any model
```

**Training Parameters**
```python
'batch_size': 64,        # M4: 64, Colab: 256
'num_epochs': 100,       # M4: 100, Colab: 200
'learning_rate': 0.001,
'dropout_rate': 0.5,     # 0.5 for local, 0.4 for Colab
```

**Preprocessing**
```python
'preprocess': {
    'wavelength_range': (450, 1000),
    'spatial_binning': None,
    'spectral_binning': None,
    'normalize': True,
}
```

### Performance Benchmarks

#### MacBook Air M4 (8GB RAM)
- Batch size: 64
- Time/epoch: ~2-3 seconds (10K samples)
- Total training: ~5-10 minutes (50 epochs)
- Best accuracy: 95%+

#### Google Colab A100 (40GB)
- Batch size: 256
- Time/epoch: ~0.5-1 second (10K samples)
- Total training: ~2-5 minutes (100 epochs)
- Best accuracy: 96%+

## 🎯 Inference

### Quick Inference
```bash
python quick_inference.py \
  --checkpoint outputs/local_m4/best_model.pth \
  --data Inference_dataset1_normalized \
  --output outputs/results \
  --batch-size 512 \
  --threshold 0.7
```

### Options
```
--checkpoint, -c    Path to trained model (.pth)
--data, -d          Path to inference data
--output, -o        Output directory
--batch-size, -b    Batch size (default: 512)
--device            Device: auto, mps, cuda, cpu
--threshold         Confidence threshold (default: 0.5)
--no-viz            Skip visualization
```

### Output Files
```
outputs/results/
├── prediction_map.npy       # Class predictions (H, W)
├── confidence_map.npy       # Confidence scores (H, W)
├── prediction_rgb.png       # RGB visualization
└── visualization.png        # Combined plots
```

## 📈 Results Interpretation

### Prediction Map
- **Format**: NumPy array with class IDs (0-10)
- **Usage**: `prediction_map[y, x]` → class ID

### Confidence Map
- **Format**: NumPy array with confidence (0.0-1.0)
- **Interpretation**: Higher = more confident

### Statistics
```
Predicted class distribution:
  PET  : 45,231 pixels (17.28%)
  HDPE : 38,567 pixels (14.74%)
  ...

Confidence statistics:
  Mean:   0.8723
  Median: 0.9156
  High confidence (>=0.7): 95.24%
```

## 🔬 Advanced Usage

### Load Predictions
```python
import numpy as np

pred_map = np.load('outputs/results/prediction_map.npy')
conf_map = np.load('outputs/results/confidence_map.npy')

# Filter by confidence
reliable = pred_map[conf_map >= 0.7]
```

### Batch Inference
```python
datasets = [
    'Inference_dataset1_normalized',
    'Inference_dataset2_normalized',
    'Inference_dataset3_normalized',
]

for data in datasets:
    # Run inference
    !python quick_inference.py -c model.pth -d {data} -o outputs/{data}
```

### Custom Analysis
```python
# Calculate material percentages
unique, counts = np.unique(pred_map, return_counts=True)
for class_id, count in zip(unique, counts):
    pct = 100.0 * count / pred_map.size
    print(f"{CLASS_NAMES[class_id]}: {pct:.2f}%")
```

## 🎓 Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training documentation
  - Model descriptions
  - Hyperparameter tuning
  - Platform-specific optimizations
  - Troubleshooting

- **[INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)** - Complete inference documentation
  - Usage examples
  - Result interpretation
  - Advanced analysis
  - Best practices

## 💡 Tips & Best Practices

### Training
✅ Start with `attention_net` model  
✅ Use class weights for imbalanced data  
✅ Enable data augmentation  
✅ Monitor validation accuracy  
✅ Use early stopping  

### Inference
✅ Match preprocessing with training  
✅ Use confidence thresholds  
✅ Validate predictions visually  
✅ Check class distributions  
✅ Analyze uncertainty  

### Performance
✅ Use larger batches for inference  
✅ Enable MPS on M4 for GPU acceleration  
✅ Use pin_memory on CUDA devices  
✅ Process multiple datasets in batch  

## 🐛 Troubleshooting

### MPS Not Available (M4)
```bash
# Update PyTorch
pip install --upgrade torch torchvision
```

### Out of Memory
```python
# Reduce batch size
'batch_size': 32,  # Instead of 64
```

### Model Dimension Mismatch
```python
# Ensure preprocessing matches training
checkpoint = torch.load('model.pth')
CONFIG['preprocess'] = checkpoint['config']['preprocess']
```

### Low Confidence Predictions
- Check preprocessing consistency
- Verify data quality
- Consider retraining with more data
- Use ensemble methods

## 📚 References

### Papers
- **HybridSN**: Roy et al. "HybridSN: Exploring 3D-2D CNN Feature Hierarchy for Hyperspectral Image Classification." IEEE GRSL, 2020.
- **ResNet**: He et al. "Deep Residual Learning for Image Recognition." CVPR, 2016.
- **Attention**: Hu et al. "Squeeze-and-Excitation Networks." CVPR, 2018.

### Technologies
- PyTorch: Deep learning framework
- Spectral Python: Hyperspectral image processing
- MPS: Apple Metal Performance Shaders
- CUDA: NVIDIA GPU acceleration

## 🤝 Contributing

To add new models:
1. Add model class to `pipeline_model.py`
2. Update `create_model()` function
3. Test with training pipeline
4. Update documentation

## 📝 License

This project is for research and educational purposes.

## 📧 Support

For issues or questions:
- Check documentation guides
- Review error messages
- Verify paths and configurations
- Compare with example configurations

## 🎉 Acknowledgments

- PyTorch team for MPS support
- Google Colab for free GPU access
- Spectral Python developers
- Research community for model architectures

---

**Ready to classify?** Start with the [TRAINING_GUIDE.md](TRAINING_GUIDE.md)! 🚀

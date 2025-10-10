# Quick Reference Guide

## 🚀 One-Command Operations

### Training

**Local (MacBook M4)**
```bash
python train_local_m4.py
```

**Colab (Google Colab)**
- Open `train_colab.ipynb` in Colab
- Run all cells

### Inference

**Quick Inference**
```bash
python quick_inference.py \
  --checkpoint outputs/local_m4/best_model.pth \
  --data Inference_dataset1_normalized
```

**Full Inference**
```bash
python inference_local_m4.py
```

**Compare Models**
```bash
python compare_models.py \
  --models outputs/local_m4/model1.pth outputs/local_m4/model2.pth \
  --data Inference_dataset1_normalized
```

## 📝 Common Tasks

### Change Model Type
```python
# In train_local_m4.py or CONFIG section
'model_type': 'attention_net'  # or: spectral_cnn, hybrid_sn, resnet1d, deep_cnn
```

### Adjust Batch Size
```python
# Training
'batch_size': 64,   # M4: 32-128, Colab: 128-512

# Inference
'batch_size': 512,  # M4: 256-1024, Colab: 512-2048
```

### Change Learning Rate
```python
'learning_rate': 0.001,  # Default
'learning_rate': 0.0001, # If unstable
'learning_rate': 0.01,   # If too slow
```

### Enable/Disable Augmentation
```python
'augment': True,   # Enable (recommended)
'augment': False,  # Disable for faster training
```

### Adjust Early Stopping
```python
'early_stopping_patience': 15,  # Stop after 15 epochs without improvement
'early_stopping_patience': 30,  # More patience
```

## 🔍 Check Results

### View Training History
```python
import numpy as np
import matplotlib.pyplot as plt

history = np.load('outputs/local_m4/history_TIMESTAMP.npy', allow_pickle=True).item()

print(f"Best accuracy: {history['best_val_acc']:.2f}%")
print(f"At epoch: {history['best_epoch']}")

plt.plot(history['val_accs'])
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.show()
```

### Load Predictions
```python
import numpy as np

pred_map = np.load('outputs/inference_local_m4/prediction_map.npy')
conf_map = np.load('outputs/inference_local_m4/confidence_map.npy')

print(f"Shape: {pred_map.shape}")
print(f"Mean confidence: {conf_map.mean():.4f}")
```

### Check Model Info
```python
import torch

checkpoint = torch.load('outputs/local_m4/best_model.pth')
config = checkpoint['config']

print(f"Model type: {config['model_type']}")
print(f"Best accuracy: {checkpoint['best_val_acc']:.2f}%")
print(f"Trained at epoch: {checkpoint['epoch']}")
```

## 🎯 Model Selection Guide

**For Quick Experiments**
→ Use `spectral_cnn` (fastest)

**For Best Overall Performance**
→ Use `attention_net` (recommended)

**For Maximum Accuracy**
→ Use `deep_cnn` (slowest but best)

**For Spatial Features**
→ Use `hybrid_sn` (3D-2D hybrid)

**For Deep Features**
→ Use `resnet1d` (residual network)

## 🐛 Quick Fixes

### Out of Memory
```python
'batch_size': 32,  # Reduce batch size
```

### Training Too Slow
```python
'batch_size': 128,     # Increase batch size (if memory allows)
'num_workers': 4,      # More data loading workers (Colab only)
```

### Poor Accuracy
```python
'augment': True,           # Enable augmentation
'learning_rate': 0.0001,   # Lower learning rate
'dropout_rate': 0.3,       # Less dropout
'num_epochs': 200,         # Train longer
```

### Overfitting
```python
'dropout_rate': 0.6,       # More dropout
'weight_decay': 1e-3,      # More regularization
'augment': True,           # More augmentation
```

### MPS Not Working (M4)
```bash
pip install --upgrade torch torchvision
```

## 📊 File Locations

### Training Outputs
```
outputs/local_m4/
├── best_model_TIMESTAMP.pth              # Best model
├── checkpoint_epoch_10_TIMESTAMP.pth     # Periodic saves
├── history_TIMESTAMP.npy                 # Training history
└── training_history_TIMESTAMP.png        # Plots
```

### Inference Outputs
```
outputs/inference_local_m4/
├── prediction_map_TIMESTAMP.npy          # Predictions
├── confidence_map_TIMESTAMP.npy          # Confidence
├── prediction_rgb_TIMESTAMP.png          # RGB visualization
└── inference_results_TIMESTAMP.png       # Combined plot
```

## 🔗 Documentation Links

- **[README.md](README.md)** - Project overview
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training guide
- **[INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)** - Complete inference guide

## ⚡ Performance Tips

### MacBook M4
✅ Use `device='mps'` for GPU  
✅ Set `num_workers=0`  
✅ Batch size 32-128  
✅ Close other apps for more RAM  

### Google Colab
✅ Use A100 GPU if available  
✅ Set `num_workers=4`  
✅ Batch size 128-512  
✅ Use `pin_memory=True`  

## 📞 Getting Help

1. Check error message carefully
2. Review relevant guide (Training/Inference)
3. Verify paths are correct
4. Check preprocessing matches
5. Try reducing batch size
6. Compare with example configs

## 🎓 Learning Path

**Beginner**
1. Run `train_local_m4.py` with defaults
2. Check training plots
3. Run `quick_inference.py`
4. View predictions

**Intermediate**
1. Try different models
2. Tune hyperparameters
3. Enable augmentation
4. Compare models

**Advanced**
1. Create custom models
2. Implement ensemble methods
3. Add new preprocessing
4. Optimize for your data

---

**Happy classifying!** 🎯

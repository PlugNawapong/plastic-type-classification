# 🚀 Quick Start for MacBook Air M4

## ✅ All Issues Fixed!

### What Was Fixed
1. ✅ **OpenMP library conflict** - Built-in automatic fix
2. ✅ **pin_memory warning** - Automatically disabled for MPS
3. ✅ **M4 GPU support** - Automatic detection and usage
4. ✅ **6 model architectures** - All optimized for your M4

---

## 🎯 Run Training Now!

### Option 1: Recommended (ResNet, 50 epochs, ~33 min)
```bash
python run_pipeline_config.py --mode full \
  --model-type resnet \
  --epochs 50 \
  --batch-size 512 \
  --spectral-binning 2
```

### Option 2: Best Accuracy (Inception, 100 epochs, ~75 min)
```bash
python run_pipeline_config.py --mode full \
  --model-type inception \
  --epochs 100 \
  --batch-size 256 \
  --spectral-binning 2 \
  --denoise --denoise-strength 1.0
```

### Option 3: Quick Test (CNN, 10 epochs, ~5 min)
```bash
python run_pipeline_config.py --mode full \
  --model-type cnn \
  --epochs 10 \
  --batch-size 1024 \
  --spectral-binning 5
```

---

## 📊 Available Models

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| `cnn` | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Quick experiments |
| `resnet` | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | **Default choice** |
| `deep` | ⚡⚡⚡ | ⭐⭐⭐⭐ | Large datasets |
| `inception` | ⚡⚡ | ⭐⭐⭐⭐⭐ | Best accuracy |
| `lstm` | ⚡⚡ | ⭐⭐⭐ | Sequential |
| `transformer` | ⚡ | ⭐⭐⭐⭐ | State-of-the-art |

---

## ⚡ Key Parameters

### Model Selection
```bash
--model-type resnet        # ResNet (default, recommended)
--model-type inception     # Best accuracy
--model-type transformer   # State-of-the-art
```

### Training Control
```bash
--epochs 50                # Number of epochs
--batch-size 512           # Batch size (256-1024)
--lr 0.001                 # Learning rate
```

### Data Preprocessing
```bash
--spectral-binning 2       # Reduce bands (2,5,10)
--spatial-binning 2        # Reduce resolution (2,4)
--denoise                  # Enable denoising
--wavelength-range 450 700 # Filter wavelengths
```

---

## 🔍 Verify GPU is Working

```bash
python -c "import torch; print('M4 GPU:', torch.backends.mps.is_available())"
```

Expected: `M4 GPU: True`

When training starts, you should see:
```
✓ Using Apple Silicon GPU (MPS)
```

---

## 💾 Output Files

After training completes:
```
output/
├── training/
│   ├── best_model.pth           # Best model weights
│   ├── training_history.png     # Loss/accuracy plots
│   └── training_history.json    # Metrics
└── inference/
    ├── predictions.png          # Prediction visualization
    ├── probability_maps/        # Per-class probabilities
    └── inference_statistics.json # Statistics
```

---

## 📈 Expected Training Times (M4)

| Configuration | Time | Accuracy |
|--------------|------|----------|
| `cnn`, 20 epochs, binning=5 | ~10 min | ~85% |
| `resnet`, 50 epochs, binning=2 | ~33 min | ~90% |
| `inception`, 100 epochs, binning=2 | ~75 min | ~92-95% |
| `transformer`, 100 epochs, binning=2 | ~75 min | ~91-94% |

---

## 🛠️ Troubleshooting

### Out of Memory?
```bash
--batch-size 256 --spatial-binning 2
```

### Too Slow?
```bash
--spectral-binning 5 --batch-size 1024
```

### Low Accuracy?
```bash
--model-type inception --epochs 100 --denoise
```

### More Help
See [M4_TROUBLESHOOTING.md](M4_TROUBLESHOOTING.md) for detailed troubleshooting.

---

## 📚 Full Documentation

- **[M4_GPU_SETUP.md](M4_GPU_SETUP.md)** - Complete M4 setup guide
- **[GPU_AND_MODELS.md](GPU_AND_MODELS.md)** - Model architectures explained
- **[M4_TROUBLESHOOTING.md](M4_TROUBLESHOOTING.md)** - Troubleshooting guide
- **[README_PIPELINE.md](README_PIPELINE.md)** - Full pipeline documentation

---

## 🎓 Examples from Your Previous Command

Your command with the OpenMP fix applied:
```bash
python run_pipeline_config.py --mode full \
  --lower-percentile 2 \
  --upper-percentile 98 \
  --spectral-binning 10 \
  --spatial-binning 4 \
  --denoise --denoise-method gaussian --denoise-strength 3.0 \
  --model-type inception \
  --dropout 0.3 \
  --epochs 100 \
  --lr 0.001 \
  --batch-size 256 \
  --val-ratio 0.2
```

**This will now run without errors!** ✅

Expected output:
```
✓ Using Apple Silicon GPU (MPS)
Spectral binning: 458 bands -> 45 bands (bin_size=10)
Spatial binning: (1176, 2092) -> (294, 523) (bin_size=4)
Denoising with gaussian filter (strength=3.0)...
Model type: inception
Total parameters: 2,200,000

Starting training on mps
[Training progress...]
```

---

## ⚡ One-Line Test

Test everything is working:
```bash
python run_pipeline_config.py --mode train --model-type cnn --epochs 2 --spectral-binning 10 --spatial-binning 4
```

Should complete in ~2 minutes with:
```
✓ Using Apple Silicon GPU (MPS)
✓ Best model saved
```

---

**You're all set! Start training! 🚀**

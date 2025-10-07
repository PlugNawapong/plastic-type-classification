# GPU Support & Model Selection Guide

## 🚀 Apple Silicon M4 GPU Acceleration

### Automatic GPU Detection

The pipeline now **automatically detects and uses your M4 GPU** via PyTorch's Metal Performance Shaders (MPS) backend!

**When you run training or inference**, you'll see:
```
✓ Using Apple Silicon GPU (MPS)
```

### GPU Performance Benefits

With M4 GPU acceleration:
- **Training speed**: 5-10x faster than CPU
- **Inference speed**: 10-20x faster than CPU
- **Memory efficiency**: Unified memory architecture optimizes data transfer

### Verify GPU is Working

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

Expected output:
```
MPS available: True
```

### GPU Memory Management

If you encounter memory issues on your MacBook Air M4:

```bash
# Reduce batch size
python run_pipeline_config.py --mode train --batch-size 256

# Or reduce spatial resolution
python run_pipeline_config.py --mode train --spatial-binning 4

# Or use spectral binning
python run_pipeline_config.py --mode train --spectral-binning 5
```

---

## 🧠 Model Architecture Selection

The pipeline now includes **6 different model architectures** optimized for hyperspectral classification:

### 1. **ResNet** (Recommended, Default)
```bash
--model-type resnet
```

**Best for:** Most cases, especially with limited data

**Architecture:**
- Residual connections for better gradient flow
- 3 residual blocks with downsampling
- Global average pooling

**Parameters:** ~694K
**Speed:** Fast
**Accuracy:** Excellent

**When to use:**
- Default choice for most applications
- Good balance of speed and accuracy
- Works well with small to medium datasets

---

### 2. **Basic CNN**
```bash
--model-type cnn
```

**Best for:** Quick experiments, baseline comparisons

**Architecture:**
- 4 convolutional blocks
- Batch normalization and dropout
- 3 fully connected layers

**Parameters:** ~500K
**Speed:** Very fast
**Accuracy:** Good

**When to use:**
- Fast prototyping
- Establishing baselines
- When computational resources are limited

---

### 3. **Deep CNN** (New!)
```bash
--model-type deep
```

**Best for:** Complex patterns, large datasets

**Architecture:**
- 5 convolutional layers (32→64→128→256→512 channels)
- Multiple pooling stages
- 3-layer classifier

**Parameters:** ~1.8M
**Speed:** Medium
**Accuracy:** Very good

**When to use:**
- Large training datasets (>1M pixels)
- Complex spectral patterns
- When you need deeper feature extraction

---

### 4. **Inception** (New!)
```bash
--model-type inception
```

**Best for:** Multi-scale feature extraction

**Architecture:**
- Inception modules with parallel convolutions (1x1, 3x3, 5x5)
- Multi-scale feature extraction
- Captures both broad and fine spectral features

**Parameters:** ~2.2M
**Speed:** Medium-slow
**Accuracy:** Excellent

**When to use:**
- Spectral signatures at multiple scales
- Best overall accuracy needed
- Diverse material types

---

### 5. **LSTM** (New!)
```bash
--model-type lstm
```

**Best for:** Sequential spectral patterns

**Architecture:**
- Bidirectional LSTM layers
- Treats spectral bands as time series
- Captures spectral dependencies

**Parameters:** ~600K
**Speed:** Slow
**Accuracy:** Good

**When to use:**
- Spectral bands have strong sequential dependencies
- Absorption/emission features in specific wavelength orders
- Alternative perspective to CNNs

---

### 6. **Transformer** (New!)
```bash
--model-type transformer
```

**Best for:** Global spectral relationships, state-of-the-art

**Architecture:**
- Self-attention mechanism
- Positional encoding for wavelength information
- 4-layer transformer encoder

**Parameters:** ~850K
**Speed:** Slow
**Accuracy:** Very good to excellent

**When to use:**
- Maximum accuracy is priority
- Sufficient computational resources
- Global spectral feature relationships important
- Research and experimentation

---

## 📊 Model Comparison

| Model | Parameters | Speed | Accuracy | Memory | Best Use Case |
|-------|-----------|-------|----------|---------|---------------|
| **cnn** | ~500K | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Low | Quick experiments |
| **resnet** | ~694K | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Low | **Default choice** |
| **deep** | ~1.8M | ⚡⚡⚡ | ⭐⭐⭐⭐ | Medium | Large datasets |
| **inception** | ~2.2M | ⚡⚡ | ⭐⭐⭐⭐⭐ | Medium | Best accuracy |
| **lstm** | ~600K | ⚡⚡ | ⭐⭐⭐ | Medium | Sequential patterns |
| **transformer** | ~850K | ⚡ | ⭐⭐⭐⭐ | High | State-of-the-art |

---

## 🎯 Usage Examples

### Example 1: Fast Training with ResNet (Recommended)
```bash
python run_pipeline_config.py --mode train \
  --model-type resnet \
  --epochs 50 \
  --batch-size 512 \
  --spectral-binning 2
```

### Example 2: Maximum Accuracy with Inception
```bash
python run_pipeline_config.py --mode train \
  --model-type inception \
  --epochs 100 \
  --lr 0.0005 \
  --batch-size 256 \
  --spectral-binning 2 \
  --denoise --denoise-strength 1.0
```

### Example 3: Quick Experiment with Basic CNN
```bash
python run_pipeline_config.py --mode train \
  --model-type cnn \
  --epochs 20 \
  --batch-size 1024 \
  --spectral-binning 5
```

### Example 4: Deep Learning with Transformer
```bash
python run_pipeline_config.py --mode train \
  --model-type transformer \
  --epochs 100 \
  --lr 0.0001 \
  --batch-size 256 \
  --spectral-binning 2 \
  --dropout 0.2
```

### Example 5: Sequential Analysis with LSTM
```bash
python run_pipeline_config.py --mode train \
  --model-type lstm \
  --epochs 75 \
  --lr 0.001 \
  --batch-size 512 \
  --spectral-binning 2
```

### Example 6: Large Dataset with Deep CNN
```bash
python run_pipeline_config.py --mode train \
  --model-type deep \
  --epochs 100 \
  --batch-size 512 \
  --spectral-binning 2 \
  --weight-decay 0.0001
```

---

## 🔬 Model Selection Decision Tree

```
Start here
    │
    ├─ Need fastest results? → Use 'cnn'
    │
    ├─ Standard classification? → Use 'resnet' (default)
    │
    ├─ Large dataset (>1M pixels)?
    │   ├─ Yes → Try 'deep' or 'inception'
    │   └─ No → Use 'resnet'
    │
    ├─ Need best possible accuracy?
    │   ├─ Computational resources available? → Use 'inception' or 'transformer'
    │   └─ Limited resources → Use 'resnet'
    │
    └─ Research/experimentation? → Try 'transformer' or 'lstm'
```

---

## 💡 Performance Tips

### For MacBook Air M4

**1. Optimize Batch Size for GPU**
```bash
# Start with 512 and increase if no memory errors
--batch-size 512

# If memory errors, reduce
--batch-size 256
```

**2. Use Mixed Precision (Coming Soon)**
```bash
# Will be added in future update
--use-mixed-precision
```

**3. Reduce Data Dimensionality**
```bash
# Spectral binning (458 → 45 bands)
--spectral-binning 10

# Spatial binning (reduce resolution 16x)
--spatial-binning 4
```

**4. Efficient Data Loading**
```bash
# Increase workers for faster data loading
--num-workers 8
```

---

## 🐛 Troubleshooting

### GPU Not Detected

**Problem:** Shows "Using CPU" instead of "Using Apple Silicon GPU (MPS)"

**Solution:**
```bash
# Update PyTorch to latest version
pip install --upgrade torch torchvision

# Verify MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Out of Memory Errors

**Problem:** RuntimeError: MPS backend out of memory

**Solutions:**
```bash
# 1. Reduce batch size
--batch-size 128

# 2. Reduce model size
--model-type cnn  # Instead of deep/inception

# 3. Reduce data dimensions
--spectral-binning 5 --spatial-binning 2

# 4. Reduce number of bands
--select-bands 50
```

### Slow Training

**Problem:** Training is slower than expected

**Check:**
1. ✓ GPU is being used (should show "Using Apple Silicon GPU")
2. ✓ Batch size is reasonable (256-512)
3. ✓ Data preprocessing is cached (normalized folder exists)
4. ✓ Number of workers is set (--num-workers 4-8)

---

## 📈 Expected Performance (M4 GPU)

| Model | Bands | Batch Size | Time/Epoch | Total Time (50 epochs) |
|-------|-------|-----------|------------|----------------------|
| cnn | 229 | 512 | ~30s | ~25 min |
| resnet | 229 | 512 | ~40s | ~33 min |
| deep | 229 | 512 | ~60s | ~50 min |
| inception | 229 | 256 | ~90s | ~75 min |
| lstm | 229 | 512 | ~120s | ~100 min |
| transformer | 229 | 256 | ~90s | ~75 min |

*Times are approximate and depend on dataset size and preprocessing*

---

## 🏆 Recommended Configurations

### For Production (Best Accuracy)
```bash
python run_pipeline_config.py --mode full \
  --model-type inception \
  --epochs 100 \
  --lr 0.0005 \
  --batch-size 256 \
  --spectral-binning 2 \
  --denoise --denoise-strength 1.0 \
  --dropout 0.3
```

### For Fast Iteration (Quick Results)
```bash
python run_pipeline_config.py --mode full \
  --model-type resnet \
  --epochs 30 \
  --batch-size 512 \
  --spectral-binning 5
```

### For Research (State-of-the-art)
```bash
python run_pipeline_config.py --mode full \
  --model-type transformer \
  --epochs 100 \
  --lr 0.0001 \
  --batch-size 256 \
  --spectral-binning 2 \
  --dropout 0.2 \
  --weight-decay 0.00001
```

---

## 📚 References

- **Apple Silicon MPS**: https://pytorch.org/docs/stable/notes/mps.html
- **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (2015)
- **Inception**: Szegedy et al., "Going Deeper with Convolutions" (2014)
- **Transformer**: Vaswani et al., "Attention Is All You Need" (2017)
- **Hyperspectral Classification**: Various research papers on spectral signature analysis

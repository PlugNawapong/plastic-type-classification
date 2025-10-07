# 🚀 Google Colab Quick Start

## ⚡ 3-Step Setup

### Step 1: Push to GitHub (5 min)
```bash
cd /Users/nawapong/Projects/plastic-type-classification

# Initialize and push
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/plastic-type-classification.git
git push -u origin main
```

### Step 2: Upload Data to Google Drive (30-60 min)
1. Create folder in Google Drive: `plastic-type-classification`
2. Upload these folders:
   - `training_dataset/` (458 PNG files + header.json)
   - `Ground_Truth/` (labels.png + labels.json)
   - `Inference_dataset1/` (458 PNG files + header.json)

### Step 3: Run in Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Open `colab_pipeline.ipynb` from your GitHub
3. Set Runtime → A100 GPU
4. Run all cells!

---

## 📊 Performance Comparison

| Platform | GPU | Time (50 epochs) | Speedup |
|----------|-----|-----------------|---------|
| MacBook Air M4 | MPS | ~33 min | 1x |
| Colab T4 | CUDA | ~20 min | 1.7x |
| Colab V100 | CUDA | ~12 min | 2.8x |
| **Colab A100** | **CUDA** | **~6 min** | **5.5x** |

**Colab Pro+ A100 is 5-6x faster than your M4!**

---

## 🎯 Complete Workflow

### On Your Mac (One Time Setup)

```bash
# 1. Navigate to project
cd /Users/nawapong/Projects/plastic-type-classification

# 2. Create .gitignore (already done)
# 3. Initialize git
git init

# 4. Add files
git add .

# 5. Commit
git commit -m "Initial commit: Hyperspectral pipeline with 6 models"

# 6. Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/plastic-type-classification.git
git push -u origin main
```

### Upload Data to Google Drive

**Files to upload** (use Google Drive web interface or Desktop app):
```
Google Drive/plastic-type-classification/
├── training_dataset/          # 458 PNG files
├── Ground_Truth/              # 2 files
└── Inference_dataset1/        # 458 PNG files
```

**Code will come from GitHub** (automatically cloned in Colab)

### In Google Colab

**Method 1: Clone from GitHub + Mount Drive for Data**
```python
# Cell 1: Clone code from GitHub
!git clone https://github.com/YOUR_USERNAME/plastic-type-classification.git
%cd plastic-type-classification

# Cell 2: Mount Drive for data
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Create symbolic links to data
!ln -s /content/drive/MyDrive/plastic-type-classification/training_dataset ./
!ln -s /content/drive/MyDrive/plastic-type-classification/Ground_Truth ./
!ln -s /content/drive/MyDrive/plastic-type-classification/Inference_dataset1 ./

# Cell 4: Run pipeline
!python run_pipeline_config.py --mode train --model-type inception --epochs 100
```

**Method 2: Everything in Drive (Simpler)**
```python
# Cell 1: Mount and navigate
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/plastic-type-classification

# Cell 2: Pull latest code from GitHub
!git pull

# Cell 3: Run pipeline
!python run_pipeline_config.py --mode train --model-type inception --epochs 100
```

---

## 🎬 Example Colab Session

```python
# === CELL 1: Setup ===
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/plastic-type-classification

# === CELL 2: Check GPU ===
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
# Expected: CUDA available: True, GPU: Tesla A100-SXM4-40GB

# === CELL 3: Quick Training Test ===
!python run_pipeline_config.py --mode train \
  --model-type cnn \
  --epochs 5 \
  --batch-size 1024 \
  --spectral-binning 10 \
  --spatial-binning 4

# Should complete in ~2-3 minutes

# === CELL 4: Full Training ===
!python run_pipeline_config.py --mode full \
  --model-type inception \
  --epochs 100 \
  --batch-size 512 \
  --spectral-binning 2 \
  --denoise --denoise-strength 1.0

# Expected: ~12-15 minutes on A100

# === CELL 5: View Results ===
import json
with open('output/training/training_history.json') as f:
    history = json.load(f)
print(f"Best Accuracy: {history['best_val_acc']:.2f}%")

# === CELL 6: Download Results ===
!zip -r results.zip output/
from google.colab import files
files.download('results.zip')
```

---

## 💡 Tips for Colab Pro+

### 1. Request A100 GPU
```python
# Check what GPU you got
import torch
print(torch.cuda.get_device_name(0))

# If not A100, disconnect and reconnect
# Runtime → Disconnect and delete runtime
# Runtime → Change runtime type → GPU → A100
```

### 2. Keep Session Alive
- Keep browser tab open
- Run lightweight monitoring:
```python
import time
for i in range(100):
    print(f"Epoch {i}/100 running...")
    time.sleep(60)  # Update every minute
```

### 3. Use Larger Batch Sizes
```python
# Colab A100 has 40GB memory, use it!
BATCH_SIZE = 1024  # or even 2048
```

### 4. Try All Models in One Session
```python
models = ["cnn", "resnet", "deep", "inception", "transformer"]
for model in models:
    !python run_pipeline_config.py --mode train \
      --skip-normalize \
      --model-type {model} \
      --epochs 50
```

---

## 📥 Getting Results Back

### Option 1: Direct Download
```python
!zip -r results.zip output/
from google.colab import files
files.download('results.zip')
```

### Option 2: Already in Google Drive
Results are automatically saved to Drive if you're working there:
```
Google Drive/plastic-type-classification/output/
├── training/
│   ├── best_model.pth
│   ├── training_history.json
│   └── training_history.png
└── inference/
    ├── predictions.png
    └── inference_statistics.json
```

### Option 3: Sync to Mac via Drive
If using Google Drive Desktop app, files automatically sync to Mac!

---

## 🔄 Iterative Development

### Mac → Colab Workflow

**On Mac: Make improvements**
```bash
# Edit code
vim pipeline_model.py

# Commit and push
git add pipeline_model.py
git commit -m "Improved model architecture"
git push
```

**On Colab: Get updates and train**
```python
%cd /content/drive/MyDrive/plastic-type-classification
!git pull origin main

# Train with new code
!python run_pipeline_config.py --mode train --model-type resnet
```

---

## 🎯 Recommended Colab Workflow

### Day 1: Setup and Test
```python
# 1. Quick test (5 min)
!python run_pipeline_config.py --mode train \
  --model-type cnn --epochs 5 --spectral-binning 10 --spatial-binning 4

# 2. If successful, run full (60 min)
!python run_pipeline_config.py --mode full \
  --model-type resnet --epochs 50
```

### Day 2: Model Comparison
```python
# Normalize once
!python run_pipeline_config.py --mode normalize

# Try all models
for model in ["cnn", "resnet", "deep", "inception", "transformer"]:
    !python run_pipeline_config.py --mode train --skip-normalize \
      --model-type {model} --epochs 50
```

### Day 3: Best Model, More Epochs
```python
# Train best model longer
!python run_pipeline_config.py --mode train --skip-normalize \
  --model-type inception --epochs 150
```

---

## 📊 Cost Estimate

**Colab Pro+ ($49.99/month):**
- ~100 compute units/month
- A100 costs ~5 units/hour
- 1 training run (1 hour) = ~5 units
- **Can do ~20 full training runs/month**

**Much cheaper than buying a GPU!**

---

## ✅ Checklist

Before starting:
- [ ] Code pushed to GitHub
- [ ] Data uploaded to Google Drive
- [ ] Colab Pro+ subscription active
- [ ] Runtime set to A100 GPU
- [ ] Tested with quick 5-epoch run

---

## 🚀 You're Ready!

1. ✅ Code on GitHub
2. ✅ Data on Google Drive
3. ✅ Colab notebook ready
4. ✅ A100 GPU selected
5. ⚡ **6x faster training than M4!**

**Go train some models! 🎉**

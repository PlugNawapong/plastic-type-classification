# 🚀 Google Colab Pro+ Setup Guide

## Why Use Colab?

Your MacBook Air M4 is great, but Google Colab Pro+ offers:
- **⚡ More powerful GPUs** (A100, V100, T4)
- **💨 10-20x faster training** than M4
- **☁️ Cloud-based** - no local resource usage
- **💾 Persistent storage** via Google Drive

---

## 📋 Prerequisites

1. ✅ Google Colab Pro+ subscription
2. ✅ Google Drive with ~30 GB free space
3. ✅ Your project files ready

---

## 🗂️ Step 1: Upload Data to Google Drive

### Option A: Via Web Interface (Recommended for first time)

1. **Go to Google Drive** (drive.google.com)

2. **Create project folder:**
   - Create folder: `plastic-type-classification`

3. **Upload your data:**
   Upload these folders/files to `plastic-type-classification/`:
   ```
   plastic-type-classification/
   ├── training_dataset/          # Upload this folder (458 PNG files + header.json)
   ├── Ground_Truth/              # Upload this folder (labels.png + labels.json)
   ├── Inference_dataset1/        # Upload this folder (458 PNG files + header.json)
   ├── pipeline_*.py              # Upload all Python files
   ├── config.py                  # Upload config file
   ├── run_pipeline_config.py     # Upload main script
   └── colab_pipeline.ipynb       # Upload the Colab notebook
   ```

4. **Upload time estimate:**
   - Total data: ~15-20 GB
   - Upload time: 30-60 minutes (depending on your internet speed)

---

### Option B: Via Google Drive Desktop App (Faster)

1. **Install Google Drive Desktop** (if not installed)

2. **Copy files to synced folder:**
   ```bash
   # On your Mac, copy entire project to Google Drive folder
   cp -r /Users/nawapong/Projects/plastic-type-classification/* \
         ~/Google\ Drive/plastic-type-classification/
   ```

3. **Wait for sync to complete** (check Drive icon in menu bar)

---

### Option C: Via rclone (For Advanced Users)

```bash
# Install rclone
brew install rclone

# Configure Google Drive
rclone config

# Sync folder
rclone sync /Users/nawapong/Projects/plastic-type-classification \
            gdrive:plastic-type-classification -P
```

---

## 📂 Step 2: Verify Folder Structure

Your Google Drive should have this structure:

```
Google Drive/
└── plastic-type-classification/
    ├── training_dataset/
    │   ├── ImagesStack001.png
    │   ├── ImagesStack002.png
    │   ├── ... (458 files)
    │   └── header.json
    ├── Ground_Truth/
    │   ├── labels.png
    │   └── labels.json
    ├── Inference_dataset1/
    │   ├── ImagesStack001.png
    │   ├── ... (458 files)
    │   └── header.json
    ├── pipeline_normalize.py
    ├── pipeline_preprocess.py
    ├── pipeline_dataset.py
    ├── pipeline_model.py
    ├── pipeline_train.py
    ├── pipeline_inference.py
    ├── config.py
    ├── run_pipeline_config.py
    └── colab_pipeline.ipynb
```

---

## 🎯 Step 3: Open Colab Notebook

### Method 1: From Google Drive
1. Go to Google Drive
2. Navigate to `plastic-type-classification/`
3. **Double-click** `colab_pipeline.ipynb`
4. It will open in Google Colab

### Method 2: From Colab Directly
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File > Open notebook**
3. Click **Google Drive** tab
4. Navigate to `plastic-type-classification/colab_pipeline.ipynb`
5. Click to open

---

## ⚙️ Step 4: Configure Colab Settings

### A. Change Runtime Type (IMPORTANT!)

1. In Colab, click **Runtime > Change runtime type**
2. Settings:
   - **Hardware accelerator:** GPU
   - **GPU type:** A100 (if available with Pro+) or V100
   - **Runtime shape:** High-RAM (if available)
3. Click **Save**

### B. Increase Timeout (Optional)
- Pro+ gives you longer sessions (24 hours vs 12 hours)
- Keep the browser tab open during training

---

## 🚀 Step 5: Run the Pipeline

### Quick Start (Use Notebook Cells)

1. **Run each cell in order** (Shift+Enter)

2. **Cell 1:** Mount Google Drive
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   - This will ask for authorization
   - Allow access to your Google Drive

3. **Cell 2:** Check GPU
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
   - Should show `True` and GPU name

4. **Cell 3:** Install dependencies (if needed)

5. **Cell 4:** Verify data structure

6. **Cell 5:** Configure parameters
   - Edit the configuration cell:
   ```python
   MODEL_TYPE = "inception"  # Choose model
   EPOCHS = 100
   BATCH_SIZE = 512
   SPECTRAL_BINNING = 2
   ```

7. **Cell 6:** Run pipeline
   - This will run the complete training
   - Expected time: 30-60 minutes (on A100)

8. **Cell 7-8:** View results

---

## 📊 Expected Performance on Colab

| GPU Type | Time/Epoch (229 bands) | 50 Epochs | 100 Epochs |
|----------|----------------------|-----------|------------|
| **A100** | ~15s | ~12 min | ~25 min |
| **V100** | ~25s | ~20 min | ~40 min |
| **T4** | ~40s | ~33 min | ~65 min |
| M4 (Mac) | ~90s | ~75 min | ~150 min |

**Colab A100 is ~6x faster than your M4!**

---

## 💡 Tips for Colab Pro+

### 1. Save Checkpoints Regularly
The notebook automatically saves to Google Drive, but you can also:
```python
# Force save best model
!cp output/training/best_model.pth /content/drive/MyDrive/plastic-type-classification/
```

### 2. Use Multiple Sessions for Experiments
Open multiple Colab tabs to run experiments in parallel:
- Tab 1: Train ResNet
- Tab 2: Train Inception
- Tab 3: Train Transformer

### 3. Monitor Resource Usage
```python
# Check GPU memory
!nvidia-smi

# Check disk usage
!df -h
```

### 4. Download Results
Results are automatically saved to Google Drive, but you can also:
```python
from google.colab import files
!zip -r results.zip output/
files.download('results.zip')
```

---

## 🔬 Advanced: Try Multiple Models Quickly

After normalizing data once, try all models:

```python
models = ["cnn", "resnet", "deep", "inception", "transformer"]

for model in models:
    !python run_pipeline_config.py --mode train \
      --skip-normalize \
      --model-type {model} \
      --epochs 50 \
      --batch-size 512
```

This runs in **one session** and compares all models!

---

## 🐛 Troubleshooting

### Issue 1: "Runtime disconnected"
**Solution:** Colab has usage limits. With Pro+, you get:
- Up to 24 hours continuous
- Higher priority GPU access

### Issue 2: "Out of memory"
**Solution:**
```python
# Reduce batch size
BATCH_SIZE = 256  # instead of 512

# Or reduce dimensions
SPATIAL_BINNING = 4
```

### Issue 3: "Files not found"
**Solution:**
```python
# Check mounted path
!ls /content/drive/MyDrive/plastic-type-classification/

# Verify you're in correct directory
import os
print(os.getcwd())
```

### Issue 4: "Slow training"
**Solution:**
- Check GPU type: Runtime > Change runtime type > Select A100
- Verify GPU is being used:
  ```python
  import torch
  print(torch.cuda.is_available())  # Should be True
  ```

---

## 💾 Saving Your Work

### Auto-saved to Google Drive
All outputs are automatically saved:
- `output/training/` - Training results
- `output/inference/` - Inference results
- `training_dataset_normalized/` - Normalized data (can delete to save space)

### Download to Local Mac
After training completes:
1. Download results from Colab (see notebook cell)
2. Or sync from Google Drive to your Mac

---

## 📈 Comparison: M4 vs Colab

| Task | MacBook Air M4 | Colab A100 | Speedup |
|------|---------------|------------|---------|
| Normalize | ~6 min | ~3 min | 2x |
| Train (50 epochs, ResNet) | ~33 min | ~6 min | **5.5x** |
| Train (100 epochs, Inception) | ~75 min | ~12 min | **6.3x** |
| Full pipeline | ~90 min | ~15 min | **6x** |

**Colab Pro+ A100 is worth it for large experiments!**

---

## 🎓 Next Steps

1. ✅ Upload data to Google Drive
2. ✅ Open `colab_pipeline.ipynb` in Colab
3. ✅ Set runtime to A100 GPU
4. ✅ Run all cells
5. ✅ Download results
6. ✅ Celebrate! 🎉

---

## 📞 Need Help?

- Colab Docs: https://colab.research.google.com/notebooks/intro.ipynb
- GPU Guide: https://colab.research.google.com/notebooks/gpu.ipynb

See also:
- [GITHUB_SETUP.md](GITHUB_SETUP.md) - Push to GitHub
- [COLAB_TIPS.md](COLAB_TIPS.md) - Advanced tips

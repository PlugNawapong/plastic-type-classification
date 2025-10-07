# Normalization Options Guide

## 📋 Overview

The pipeline has **3 ways** to handle data normalization:

1. **Run full pipeline with normalization** (default)
2. **Skip normalization** (use already normalized data)
3. **Training-only mode** (assumes data is already normalized)

---

## Option 1: Full Pipeline with Normalization (Default)

### Command
```bash
python run_pipeline_config.py --mode full \
  --model-type inception \
  --epochs 100
```

### What Happens
1. ✅ **Normalizes** `training_dataset/` → `training_dataset_normalized/`
2. ✅ **Normalizes** `Inference_dataset1/` → `Inference_dataset1_normalized/`
3. ✅ **Trains** model on normalized data
4. ✅ **Runs inference** on normalized data

### When to Use
- ✅ First time running the pipeline
- ✅ Changed normalization parameters
- ✅ Want everything automated

---

## Option 2: Skip Normalization Flag

### Command
```bash
python run_pipeline_config.py --mode full \
  --skip-normalize \
  --model-type inception \
  --epochs 100
```

### What Happens
1. ⏭️ **Skips normalization** (uses existing `training_dataset_normalized/`)
2. ⏭️ **Skips normalization** (uses existing `Inference_dataset1_normalized/`)
3. ✅ **Trains** model on normalized data
4. ✅ **Runs inference** on normalized data

### When to Use
- ✅ You've already normalized data once
- ✅ Experimenting with different models/parameters
- ✅ Want to save 5-10 minutes

### Requirements
- ❗ Folders must exist: `training_dataset_normalized/` and `Inference_dataset1_normalized/`

---

## Option 3: Train-Only Mode (Implicit Skip)

### Command
```bash
python run_pipeline_config.py --mode train \
  --model-type inception \
  --epochs 100
```

### What Happens
1. ⏭️ **Automatically skips normalization** (normalization not in train mode)
2. ✅ **Trains** model using existing `training_dataset_normalized/`

### When to Use
- ✅ Quick training experiments
- ✅ Data already normalized from previous runs
- ✅ Only want to train (no inference)

### Requirements
- ❗ Folder must exist: `training_dataset_normalized/`

---

## Option 4: Normalize-Only Mode

### Command
```bash
python run_pipeline_config.py --mode normalize \
  --lower-percentile 2 \
  --upper-percentile 98
```

### What Happens
1. ✅ **Normalizes** both training and inference data
2. ⏹️ **Stops** (no training or inference)

### When to Use
- ✅ Prepare data once for multiple experiments
- ✅ Want to normalize with specific parameters

---

## Custom Normalization Parameters

### Change Percentiles
```bash
python run_pipeline_config.py --mode normalize \
  --lower-percentile 1 \
  --upper-percentile 99
```

### Use Different Source Data
```bash
python run_pipeline_config.py --mode full \
  --training-data my_raw_data \
  --inference-data my_inference_data
```

---

## Workflow Examples

### Workflow 1: First Time User
```bash
# Run everything (normalize + train + inference)
python run_pipeline_config.py --mode full \
  --model-type resnet \
  --epochs 50
```

---

### Workflow 2: Multiple Model Experiments

```bash
# Step 1: Normalize once
python run_pipeline_config.py --mode normalize

# Step 2: Try different models (fast!)
python run_pipeline_config.py --mode train --model-type cnn --epochs 20
python run_pipeline_config.py --mode train --model-type resnet --epochs 50
python run_pipeline_config.py --mode train --model-type inception --epochs 100
python run_pipeline_config.py --mode train --model-type transformer --epochs 100

# Each training run saves ~5-10 minutes!
```

---

### Workflow 3: Hyperparameter Tuning

```bash
# Normalize once
python run_pipeline_config.py --mode normalize

# Try different hyperparameters
python run_pipeline_config.py --mode train --skip-normalize --lr 0.001 --dropout 0.3
python run_pipeline_config.py --mode train --skip-normalize --lr 0.0005 --dropout 0.2
python run_pipeline_config.py --mode train --skip-normalize --lr 0.0001 --dropout 0.4
```

---

### Workflow 4: Different Preprocessing

```bash
# Experiment 1: Light preprocessing
python run_pipeline_config.py --mode full \
  --spectral-binning 2 \
  --model-type resnet

# Experiment 2: Heavy preprocessing (re-normalize needed)
python run_pipeline_config.py --mode full \
  --spectral-binning 10 \
  --spatial-binning 4 \
  --denoise \
  --model-type inception

# Note: Different preprocessing requires re-normalization
```

---

## Understanding Data Folders

### Input Folders (Raw Data)
```
training_dataset/          # Raw training data (458 bands)
├── ImagesStack001.png
├── ImagesStack002.png
├── ...
└── header.json

Inference_dataset1/        # Raw inference data (458 bands)
├── ImagesStack001.png
├── ...
└── header.json
```

### Output Folders (Normalized Data)
```
training_dataset_normalized/     # Created by normalization
├── ImagesStack001.png           # Normalized (2%-98%)
├── ImagesStack002.png
├── ...
├── header.json
└── normalization_metadata.json

Inference_dataset1_normalized/   # Created by normalization
├── ImagesStack001.png
├── ...
└── normalization_metadata.json
```

---

## Quick Reference

| Scenario | Command | Normalizes? | Time |
|----------|---------|-------------|------|
| First run | `--mode full` | ✅ Yes | Full |
| Already normalized | `--mode full --skip-normalize` | ❌ No | -10min |
| Train only | `--mode train` | ❌ No | -10min |
| Just normalize | `--mode normalize` | ✅ Yes | ~10min |
| Try new model | `--mode train --model-type X` | ❌ No | Fast |

---

## Your Current Situation

Since you mentioned **"the training and inference data have been normalized already"**, you have **3 options**:

### Option A: Use Existing Normalized Data (Recommended)
```bash
python run_pipeline_config.py --mode train \
  --spectral-binning 10 \
  --spatial-binning 4 \
  --denoise --denoise-strength 3.0 \
  --model-type inception \
  --epochs 100 \
  --batch-size 256
```

This will:
- ⏭️ Skip normalization (train mode doesn't normalize)
- ✅ Use existing `training_dataset_normalized/`
- ✅ Train immediately

---

### Option B: Explicitly Skip with Full Pipeline
```bash
python run_pipeline_config.py --mode full \
  --skip-normalize \
  --spectral-binning 10 \
  --spatial-binning 4 \
  --denoise --denoise-strength 3.0 \
  --model-type inception \
  --epochs 100 \
  --batch-size 256
```

This will:
- ⏭️ Skip normalization explicitly
- ✅ Use existing normalized data
- ✅ Train and run inference

---

### Option C: Re-normalize (If You Changed Parameters)
```bash
python run_pipeline_config.py --mode full \
  --lower-percentile 1 \
  --upper-percentile 99 \
  --spectral-binning 10 \
  --spatial-binning 4 \
  --denoise --denoise-strength 3.0 \
  --model-type inception \
  --epochs 100 \
  --batch-size 256
```

This will:
- ✅ Re-normalize with different percentiles
- ✅ Train on newly normalized data
- ✅ Run inference

---

## Summary

### To Skip Normalization:
1. **Use `--mode train`** (implicitly skips normalization)
2. **Use `--skip-normalize`** with `--mode full`
3. **Use `--mode normalize` first**, then `--mode train` for experiments

### Your Best Option (Data Already Normalized):
```bash
python run_pipeline_config.py --mode train \
  --spectral-binning 10 \
  --spatial-binning 4 \
  --model-type inception \
  --epochs 100
```

**This will train immediately without re-normalizing! ⚡**

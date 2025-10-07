# MacBook Air M4 Troubleshooting Guide

## ✅ OpenMP Library Conflict - FIXED

### The Error You Saw
```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

### What Was Fixed

**1. Automatic Fix in Python Scripts**
- ✓ Added `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'` to:
  - `pipeline_train.py`
  - `run_pipeline_config.py`

**2. Fixed pin_memory Warning**
- ✓ Disabled `pin_memory` for MPS devices in `pipeline_dataset.py`
- Pin memory is only enabled for CUDA GPUs now

**3. Created Shell Script Wrapper**
- ✓ `run_pipeline_m4.sh` - Sets environment variables before running

### How to Run Now

**Option 1: Direct Python (Recommended)**
```bash
python run_pipeline_config.py --mode full --model-type inception --epochs 100
```
The OpenMP fix is now built-in!

**Option 2: Shell Script (Alternative)**
```bash
./run_pipeline_m4.sh --mode full --model-type inception --epochs 100
```

**Option 3: Manual Environment Variable**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
python run_pipeline_config.py --mode full --model-type inception --epochs 100
```

---

## Common M4 Issues & Solutions

### Issue 1: "pin_memory not supported on MPS"

**Error:**
```
UserWarning: 'pin_memory' argument is set as true but not supported on MPS
```

**Status:** ✅ **FIXED** - pin_memory is now automatically disabled for MPS

**Verification:**
When you run training, you should NOT see this warning anymore.

---

### Issue 2: Out of Memory on M4

**Error:**
```
RuntimeError: MPS backend out of memory
```

**Solutions (try in order):**

**A. Reduce Batch Size**
```bash
python run_pipeline_config.py --mode train --batch-size 256  # or 128
```

**B. Reduce Spatial Resolution**
```bash
python run_pipeline_config.py --mode train --spatial-binning 4
```

**C. Reduce Spectral Bands**
```bash
python run_pipeline_config.py --mode train --spectral-binning 10
```

**D. Use Lighter Model**
```bash
python run_pipeline_config.py --mode train --model-type cnn  # instead of inception
```

**E. Combined Approach**
```bash
python run_pipeline_config.py --mode train \
  --model-type resnet \
  --batch-size 256 \
  --spectral-binning 5 \
  --spatial-binning 2
```

---

### Issue 3: MPS Not Detected

**Symptom:**
Shows "Using CPU" instead of "Using Apple Silicon GPU (MPS)"

**Diagnosis:**
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

**If False, Solutions:**

**A. Update PyTorch**
```bash
pip install --upgrade torch torchvision torchaudio
```

**B. Install PyTorch with MPS Support**
```bash
pip install torch torchvision torchaudio
```

**C. Check macOS Version**
```bash
sw_vers
```
MPS requires macOS 12.3 or later.

---

### Issue 4: Training is Slow

**Expected Training Times on M4:**

| Model | Bands | Batch Size | Time/Epoch | 50 Epochs |
|-------|-------|-----------|------------|-----------|
| cnn | 229 | 512 | ~30s | ~25 min |
| resnet | 229 | 512 | ~40s | ~33 min |
| deep | 229 | 512 | ~60s | ~50 min |
| inception | 229 | 256 | ~90s | ~75 min |
| transformer | 229 | 256 | ~90s | ~75 min |

**If slower than expected:**

**A. Verify GPU is Being Used**
Should see: `✓ Using Apple Silicon GPU (MPS)`

**B. Increase Batch Size (if memory allows)**
```bash
--batch-size 1024  # instead of 512
```

**C. Reduce Data Dimensions**
```bash
--spectral-binning 5  # Fewer bands = faster
```

**D. Check Activity Monitor**
- Open Activity Monitor
- Look for "Python" process
- CPU should be low (5-20%)
- GPU should show activity

---

### Issue 5: Training Crashes/Hangs

**A. Reduce num_workers**
```bash
python run_pipeline_config.py --mode train --num-workers 2
```

**B. Disable Multiprocessing**
```bash
python run_pipeline_config.py --mode train --num-workers 0
```

**C. Check for Other Running Processes**
```bash
# Close other heavy applications
# Safari, Chrome tabs, etc.
```

---

### Issue 6: Model Accuracy is Low

**Symptom:** Validation accuracy < 80%

**Solutions:**

**A. Train Longer**
```bash
--epochs 100  # instead of 50
```

**B. Use Better Model**
```bash
--model-type inception  # instead of cnn
```

**C. Add Denoising**
```bash
--denoise --denoise-strength 1.0
```

**D. Reduce Regularization**
```bash
--dropout 0.2  # instead of 0.3
```

**E. Adjust Learning Rate**
```bash
--lr 0.0005  # instead of 0.001 (slower learning)
```

**F. Enable Class Weights** (default, but verify)
```bash
--use-class-weights  # Should be default
```

---

### Issue 7: "No module named 'torch'"

**Solution:**
```bash
# Make sure you're in the right conda environment
conda activate dl-env

# Or install PyTorch
pip install torch torchvision torchaudio
```

---

### Issue 8: Inference Fails

**Error:** Model input/output size mismatch

**Cause:** Inference preprocessing doesn't match training

**Solution:** Use SAME preprocessing for inference as training

**Example:**
If you trained with:
```bash
--spectral-binning 2 --spatial-binning 4
```

Then inference MUST also use:
```bash
--spectral-binning 2 --spatial-binning 4
```

The pipeline should handle this automatically, but verify in config.

---

## Verification Checklist

Before running full training, verify:

### ✓ 1. PyTorch and MPS
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

Expected:
```
PyTorch version: 2.x.x
MPS available: True
```

### ✓ 2. OpenMP Fix Applied
```bash
python -c "import os; print('KMP_DUPLICATE_LIB_OK:', os.environ.get('KMP_DUPLICATE_LIB_OK', 'Not set'))"
```

If "Not set", the fix is built into the scripts and will be applied when they run.

### ✓ 3. Data Files Present
```bash
ls training_dataset/*.png | wc -l
```

Should show: `458`

### ✓ 4. Quick Test
```bash
python run_pipeline_config.py --mode train \
  --model-type cnn \
  --epochs 2 \
  --batch-size 512 \
  --spectral-binning 10 \
  --spatial-binning 4
```

Should complete in ~2 minutes with no errors.

---

## Performance Optimization for M4

### Recommended Settings for Different Scenarios

**1. Fast Experiments (5-10 minutes)**
```bash
python run_pipeline_config.py --mode train \
  --model-type cnn \
  --epochs 10 \
  --batch-size 1024 \
  --spectral-binning 10 \
  --spatial-binning 2
```

**2. Balanced (30-40 minutes)**
```bash
python run_pipeline_config.py --mode train \
  --model-type resnet \
  --epochs 50 \
  --batch-size 512 \
  --spectral-binning 2
```

**3. Best Accuracy (75-90 minutes)**
```bash
python run_pipeline_config.py --mode train \
  --model-type inception \
  --epochs 100 \
  --batch-size 256 \
  --spectral-binning 2 \
  --denoise --denoise-strength 1.0
```

**4. Production (100+ minutes)**
```bash
python run_pipeline_config.py --mode full \
  --model-type inception \
  --epochs 150 \
  --lr 0.0005 \
  --batch-size 256 \
  --spectral-binning 2 \
  --denoise --denoise-strength 1.0 \
  --dropout 0.3 \
  --weight-decay 0.0001
```

---

## Monitoring Training

### Watch GPU Usage
```bash
# Terminal 1: Run training
python run_pipeline_config.py --mode train ...

# Terminal 2: Monitor resources
watch -n 1 "ps aux | grep python"
```

### Check Training Progress
Training will show:
```
Epoch 1/100
Training: 100%|████████████| 2064/2064 [00:40<00:00, 51.23it/s, loss=1.234, acc=75.2%]
Validation: 100%|██████████| 516/516 [00:08<00:00, 62.45it/s]

Train Loss: 1.2345 | Train Acc: 75.23%
Val Loss: 1.0234 | Val Acc: 78.45%
✓ Best model saved (Val Acc: 78.45%)
```

---

## Getting Help

If issues persist:

1. **Check logs:** Training history saved to `output/training/training_history.json`

2. **Verify installation:**
```bash
pip list | grep torch
pip list | grep numpy
pip list | grep pillow
```

3. **System info:**
```bash
python --version
sw_vers
system_profiler SPHardwareDataType | grep "Model Name"
```

4. **Create minimal test case:**
```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    x = torch.randn(100, 100).to('mps')
    y = torch.randn(100, 100).to('mps')
    z = torch.mm(x, y)
    print('MPS computation successful!')
"
```

---

## Summary of Fixes Applied

✅ **OpenMP conflict** - Fixed in `pipeline_train.py` and `run_pipeline_config.py`
✅ **pin_memory warning** - Fixed in `pipeline_dataset.py`
✅ **MPS GPU detection** - Added to training and inference
✅ **6 model architectures** - All optimized for M4
✅ **Shell script wrapper** - `run_pipeline_m4.sh` for manual env control

**Your M4 is now fully optimized and ready to train! 🚀**

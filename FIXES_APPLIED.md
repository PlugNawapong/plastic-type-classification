# Fixes Applied to Pipeline

## Issue 1: File Naming Bug ✓ FIXED

**Problem:** The code was looking for `ImagesStack1.png`, `ImagesStack2.png`, etc., but the actual files are named with zero-padding: `ImagesStack001.png`, `ImagesStack002.png`, etc.

**Files Fixed:**
- ✓ `pipeline_normalize.py` (lines 84, 91)
- ✓ `pipeline_preprocess.py` (lines 36, 44)
- ✓ `hyperspectral_viewer.py` (already correct)
- ✓ `normalize_bands.py` (already correct)

**Change Made:**
```python
# Before (incorrect)
f'ImagesStack{i}.png'

# After (correct)
f'ImagesStack{i:03d}.png'
```

---

## Issue 2: Configuration Parameters Not Taking Effect ✓ FIXED

**Problem:** Command-line parameters from `run_pipeline_config.py` were being passed to `pipeline_train.py` and `pipeline_inference.py`, but those scripts were ignoring them because they defined their own local `CONFIG` variables inside the `main()` function.

**Root Cause:**
```python
# In run_pipeline_config.py
pipeline_train.CONFIG = {...}  # Sets module-level variable
pipeline_train.main()

# In pipeline_train.py main() function
def main():
    CONFIG = {...}  # Creates NEW local variable, ignoring module-level one!
```

**Files Fixed:**
- ✓ `pipeline_train.py`
- ✓ `pipeline_inference.py`

**Changes Made:**

1. Added module-level CONFIG variable:
```python
# Module-level CONFIG that can be overridden by run_pipeline_config.py
CONFIG = None
```

2. Modified `main()` function to check module-level CONFIG:
```python
def main():
    global CONFIG

    # Use module-level CONFIG if set, otherwise use defaults
    if CONFIG is None:
        CONFIG = {
            # ... default values ...
        }
```

This allows `run_pipeline_config.py` to override the CONFIG before calling `main()`, and those settings will be used.

---

## Issue 3: Learning Rate Scheduler Argument Error ✓ FIXED

**Problem:** PyTorch's `ReduceLROnPlateau` scheduler no longer accepts `verbose` parameter in newer versions.

**File Fixed:**
- ✓ `pipeline_train.py` (line 64)

**Change Made:**
```python
# Before (causes error in newer PyTorch)
self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer,
    mode='max',
    factor=0.5,
    patience=5,
    verbose=True  # ← Removed this
)

# After (works in all PyTorch versions)
self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer,
    mode='max',
    factor=0.5,
    patience=5
)
```

---

## Verification

Now when you run:
```bash
python run_pipeline_config.py --mode full \
  --spectral-binning 10 \
  --spatial-binning 4 \
  --denoise --denoise-strength 3.0 \
  --epochs 100
```

**Expected behavior:**
1. ✓ Files will load correctly (ImagesStack001.png, etc.)
2. ✓ Spectral binning of 10 will be applied → 458 bands becomes ~45 bands
3. ✓ Spatial binning of 4 will be applied → resolution reduced by 16x
4. ✓ Denoising with strength 3.0 will be applied
5. ✓ Training will run for 100 epochs

**What you should see in output:**
```
[2/6] Preprocessing cube...
Spectral binning: 458 bands -> 45 bands (bin_size=10)
Spatial binning: (1176, 2092) -> (294, 523) (bin_size=4)
Denoising with gaussian filter (strength=3.0)...
Final cube shape: (294, 523, 45)
```

Instead of:
```
Spectral binning: 458 bands -> 229 bands (bin_size=2)  # Wrong!
Final cube shape: (1176, 2092, 229)  # Wrong!
```

---

## Summary

All three issues have been fixed:
1. ✅ File naming now uses zero-padded format
2. ✅ Command-line configuration parameters now properly override defaults
3. ✅ PyTorch scheduler compatibility issue resolved

You can now run the pipeline with custom parameters and they will take effect!

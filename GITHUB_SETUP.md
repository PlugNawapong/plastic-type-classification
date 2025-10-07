# 📦 GitHub Setup Guide

## Why Push to GitHub?

- ✅ **Version control** for your code
- ✅ **Easy sync** between Mac and Colab
- ✅ **Backup** of your pipeline
- ✅ **Share** with collaborators
- ✅ **Clone** directly in Colab

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Create GitHub Repository

1. Go to [github.com](https://github.com)
2. Click **"+"** → **"New repository"**
3. Settings:
   - Repository name: `plastic-type-classification`
   - Description: `Hyperspectral plastic classification using 1D CNN`
   - Visibility: **Private** (recommended) or Public
   - ❌ **Don't** initialize with README (we have one)
4. Click **"Create repository"**

---

### Step 2: Initialize Git (if not already done)

```bash
cd /Users/nawapong/Projects/plastic-type-classification

# Check if git is initialized
git status

# If not initialized, run:
git init
```

---

### Step 3: Add Files to Git

```bash
# Add all code files (data is excluded via .gitignore)
git add .

# Check what will be committed
git status

# Should show:
# - All .py files ✓
# - All .md files ✓
# - .ipynb file ✓
# - config.py ✓
# - .gitignore ✓
#
# Should NOT show:
# - training_dataset/ ✗
# - output/ ✗
# - *.pth files ✗
```

---

### Step 4: Create Initial Commit

```bash
git commit -m "Initial commit: Hyperspectral plastic classification pipeline

- 6 model architectures (CNN, ResNet, Deep, Inception, LSTM, Transformer)
- M4 GPU (MPS) support
- Google Colab notebook
- Complete training and inference pipeline
- Comprehensive documentation"
```

---

### Step 5: Push to GitHub

```bash
# Add your GitHub repository as remote
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/plastic-type-classification.git

# Check remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

**Enter your GitHub credentials when prompted.**

---

## 🔐 GitHub Authentication

### Option A: Personal Access Token (Recommended)

1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Click **"Generate new token (classic)"**
3. Select scopes: **repo** (full control)
4. Copy the token
5. When git asks for password, paste the token

### Option B: GitHub CLI

```bash
# Install GitHub CLI
brew install gh

# Authenticate
gh auth login

# Follow prompts
```

### Option C: SSH Key

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH keys → New SSH key

# Use SSH remote instead
git remote set-url origin git@github.com:YOUR_USERNAME/plastic-type-classification.git
```

---

## 📥 Clone in Google Colab

Once pushed to GitHub, you can clone directly in Colab:

### Method 1: Public Repository

```python
# In Colab cell
!git clone https://github.com/YOUR_USERNAME/plastic-type-classification.git
%cd plastic-type-classification
```

### Method 2: Private Repository

```python
# In Colab cell
from getpass import getpass
import os

# Get your Personal Access Token
token = getpass('Enter your GitHub token: ')

# Clone
!git clone https://{token}@github.com/YOUR_USERNAME/plastic-type-classification.git
%cd plastic-type-classification
```

### Method 3: Mount Drive + Clone

```python
# Best approach: Clone to Google Drive
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/
!git clone https://github.com/YOUR_USERNAME/plastic-type-classification.git
%cd plastic-type-classification
```

---

## 🔄 Sync Between Mac and Colab

### On Mac: Make changes and push

```bash
# Make changes to code
vim pipeline_train.py

# Stage and commit
git add pipeline_train.py
git commit -m "Improved training loop"

# Push to GitHub
git push
```

### On Colab: Pull latest changes

```python
# In Colab cell
%cd /content/drive/MyDrive/plastic-type-classification
!git pull

# Now you have the latest code!
```

---

## 📁 What Gets Pushed to GitHub?

### ✅ Included (Code & Docs)
```
plastic-type-classification/
├── pipeline_normalize.py
├── pipeline_preprocess.py
├── pipeline_dataset.py
├── pipeline_model.py
├── pipeline_train.py
├── pipeline_inference.py
├── run_pipeline_config.py
├── config.py
├── colab_pipeline.ipynb
├── .gitignore
├── README_PIPELINE.md
├── COLAB_SETUP.md
├── GITHUB_SETUP.md
├── M4_GPU_SETUP.md
├── GPU_AND_MODELS.md
├── M4_TROUBLESHOOTING.md
├── NORMALIZATION_OPTIONS.md
├── SPATIAL_BINNING_FIX.md
├── FLOAT64_FIX.md
├── QUICK_START_M4.md
└── FIXES_APPLIED.md
```

### ❌ Excluded (Data & Results)
```
# Too large - store on Google Drive instead
training_dataset/          # ~10 GB
Inference_dataset1/        # ~10 GB
Ground_Truth/              # ~5 MB (could include, but excluded for privacy)
training_dataset_normalized/
Inference_dataset1_normalized/
output/                    # Generated files
*.pth                      # Model weights (100+ MB)
```

---

## 📝 Create a Great README

Create `README.md` in the root:

```bash
cat > README.md << 'EOF'
# Hyperspectral Plastic Classification

Complete pipeline for pixel-wise plastic classification from hyperspectral imagery using deep learning.

## Features

- 🎯 **6 Model Architectures**: CNN, ResNet, Deep CNN, Inception, LSTM, Transformer
- ⚡ **GPU Accelerated**: Supports Apple Silicon M4 (MPS) and CUDA
- 📊 **11 Plastic Classes**: Background + 10 plastic types
- 🔬 **458 Spectral Bands**: 450-850nm wavelength range
- ☁️ **Google Colab Ready**: Notebook included for cloud training

## Quick Start

### Local (Mac/Linux/Windows)
\`\`\`bash
python run_pipeline_config.py --mode full --model-type resnet --epochs 50
\`\`\`

### Google Colab
1. Open \`colab_pipeline.ipynb\` in Google Colab
2. Upload data to Google Drive
3. Run all cells

## Performance

| Platform | GPU | Training Time (50 epochs) |
|----------|-----|-------------------------|
| MacBook Air M4 | MPS | ~33 min |
| Google Colab Pro+ | A100 | ~6 min |

## Documentation

- [Quick Start (M4)](QUICK_START_M4.md)
- [Google Colab Setup](COLAB_SETUP.md)
- [GPU & Models Guide](GPU_AND_MODELS.md)
- [Complete Pipeline Docs](README_PIPELINE.md)

## Requirements

\`\`\`
torch>=2.0.0
torchvision
numpy
pillow
scipy
tqdm
matplotlib
\`\`\`

## Citation

If you use this pipeline, please cite:
\`\`\`
@software{hyperspectral_plastic_classification,
  title = {Hyperspectral Plastic Classification Pipeline},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/plastic-type-classification}
}
\`\`\`

## License

MIT License
EOF
```

Then commit and push:
```bash
git add README.md
git commit -m "Add README"
git push
```

---

## 🏷️ Create requirements.txt

```bash
cat > requirements.txt << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=10.0.0
scipy>=1.10.0
tqdm>=4.65.0
matplotlib>=3.7.0
EOF

git add requirements.txt
git commit -m "Add requirements.txt"
git push
```

---

## 🌿 Branching Strategy

### For experiments:
```bash
# Create experiment branch
git checkout -b experiment/inception-model

# Make changes, commit
git add .
git commit -m "Experiment: Inception model with dropout 0.5"

# Push branch
git push -u origin experiment/inception-model

# Later, merge if successful
git checkout main
git merge experiment/inception-model
git push
```

---

## 📊 Example Workflow

### Day 1: Setup
```bash
# Mac
git init
git add .
git commit -m "Initial commit"
git push -u origin main
```

### Day 2: Train on Colab
```python
# Colab
!git clone https://github.com/YOUR_USERNAME/plastic-type-classification.git
# Upload data to Drive
# Run training
```

### Day 3: Improve on Mac
```bash
# Mac
git pull  # Get any Colab changes
# Make improvements
git add .
git commit -m "Improved preprocessing"
git push
```

### Day 4: Final training on Colab
```python
# Colab
!git pull  # Get Mac improvements
# Run final training
# Download results
```

---

## 🎯 Best Practices

### 1. Commit Often
```bash
# After each significant change
git add .
git commit -m "Descriptive message"
git push
```

### 2. Use Meaningful Commit Messages
```bash
# ✅ Good
git commit -m "Fix spatial binning index error for MPS"

# ❌ Bad
git commit -m "fix bug"
```

### 3. Don't Commit Large Files
- Data files → Google Drive
- Model weights → Release or Drive
- Results → Download locally

### 4. Keep .gitignore Updated
```bash
# Add new patterns as needed
echo "new_folder/" >> .gitignore
git add .gitignore
git commit -m "Update gitignore"
```

---

## 🚀 Ready to Push!

```bash
# Final checklist
git status                    # Verify files
git add .                     # Stage all
git commit -m "Initial commit"  # Commit
git push -u origin main       # Push

# Done! Your code is on GitHub! 🎉
```

---

## 📖 Next Steps

1. ✅ Push code to GitHub
2. ✅ Upload data to Google Drive (see [COLAB_SETUP.md](COLAB_SETUP.md))
3. ✅ Clone in Colab
4. ✅ Run training on Colab Pro+ A100
5. ✅ Enjoy 6x faster training! 🚀

---

## 🆘 Troubleshooting

### "Permission denied (publickey)"
→ Use HTTPS URL or setup SSH keys (see above)

### "Large files detected"
→ Check .gitignore, don't commit data folders

### "Authentication failed"
→ Use Personal Access Token instead of password

### "Nothing to commit"
→ Make sure files aren't in .gitignore

---

Your code is now version controlled, backed up, and ready to use on Google Colab! 🎉

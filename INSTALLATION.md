# Installation Guide for KD-HGRL

## Quick Installation

### Option 1: CPU-only Installation
```bash
pip install -r requirements.txt
```

### Option 2: GPU Installation (Recommended)

#### Step 1: Check your CUDA version
```bash
nvidia-smi
# or
nvcc --version
```

#### Step 2: Install PyTorch with CUDA support

**For CUDA 11.6:**
```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

**For CUDA 11.7:**
```bash
pip install torch==1.12.1+cu117 torchvision==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

**For CUDA 11.8:**
```bash
pip install torch==1.13.1+cu118 torchvision==0.14.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

**For newer CUDA versions (12.x):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Step 3: Install PyTorch Sparse
```bash
# For CUDA 11.6
pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.0+cu116.html

# For CUDA 11.7  
pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.0+cu117.html

# For CUDA 11.8
pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.0+cu118.html

# For CUDA 12.1
pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
```

#### Step 4: Install remaining dependencies
```bash
pip install numpy scipy scikit-learn tqdm jupyter notebook ipykernel matplotlib
```

## Alternative: Conda Installation

```bash
# Create new environment
conda create -n kd-hgrl python=3.8
conda activate kd-hgrl

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install other dependencies
conda install numpy scipy scikit-learn tqdm matplotlib jupyter notebook
pip install torch-sparse
```

## Verification

Test your installation:
```python
import torch
import numpy as np
import scipy.sparse as sp
import torch_sparse
import sklearn
import tqdm

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

## Common Issues

### Issue 1: torch-sparse installation fails
**Solution**: Make sure your PyTorch version matches the torch-sparse version. Check [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for the correct wheel URL.

### Issue 2: CUDA version mismatch
**Solution**: 
1. Check your CUDA version with `nvidia-smi`
2. Install the matching PyTorch version
3. Ensure torch-sparse is compatible

### Issue 3: Import errors in Jupyter
**Solution**:
```bash
python -m ipykernel install --user --name=kd-hgrl
# Then select "kd-hgrl" kernel in Jupyter
```

## System Requirements

- **Python**: 3.7-3.10
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for datasets and models

## Notes

- The project works on both CPU and GPU, but GPU is recommended for faster training
- If you encounter version conflicts, try creating a fresh virtual environment
- For Windows users: Make sure you have Visual C++ build tools installed
# SAM3 Environment Setup for Windows

Complete guide to set up SAM3 (Segment Anything Model 3) on a new Windows PC.

## Prerequisites

- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA support (minimum 8GB VRAM recommended)
- Python 3.12 installed ([Download](https://www.python.org/downloads/))
- Git installed ([Download](https://git-scm.com/downloads))
- NVIDIA Driver 525+ with CUDA 12.6+ support
- HuggingFace account with access to SAM3 model (gated model)

## Step 1: Verify System Requirements

```cmd
:: Check Python 3.12 is available
py -0

:: Check NVIDIA GPU and CUDA
nvidia-smi

:: Check Git
git --version
```

Expected output for `py -0`:
```
-V:3.12          Python 3.12 (64-bit)
```

## Step 2: Create Project Directory

```cmd
mkdir C:\path\to\your\project
cd C:\path\to\your\project
```

## Step 3: Create Python 3.12 Virtual Environment

```cmd
py -3.12 -m venv venv_sam3
```

## Step 4: Activate Environment and Upgrade pip

```cmd
venv_sam3\Scripts\activate
python -m pip install --upgrade pip
```

## Step 5: Install PyTorch with CUDA 12.6

```cmd
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

This downloads ~2.5GB. Wait for completion.

## Step 6: Clone and Install SAM3

```cmd
git clone https://github.com/facebookresearch/sam3.git sam3_repo
pip install -e sam3_repo
```

## Step 7: Install Additional Dependencies

```cmd
pip install opencv-python decord pycocotools psutil pip-system-certs truststore
```

## Step 8: Apply Windows Triton Workaround

SAM3 uses Triton for GPU acceleration, but Triton is not available on Windows. Replace the EDT file with a version that falls back to OpenCV.

Create/replace `sam3_repo/sam3/model/edt.py` with:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Euclidean distance transform (EDT) with Windows fallback"""

import torch
import sys

# Try to import Triton (not available on Windows)
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass


if TRITON_AVAILABLE:
    @triton.jit
    def edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr):
        batch_id = tl.program_id(axis=0)
        if horizontal:
            row_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + row_id * width
            length = width
            stride = 1
        else:
            col_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + col_id
            length = height
            stride = width

        k = 0
        for q in range(1, length):
            cur_input = tl.load(inputs_ptr + block_start + (q * stride))
            r = tl.load(v + block_start + (k * stride))
            z_k = tl.load(z + block_start + (k * stride))
            previous_input = tl.load(inputs_ptr + block_start + (r * stride))
            s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            while s <= z_k and k - 1 >= 0:
                k = k - 1
                r = tl.load(v + block_start + (k * stride))
                z_k = tl.load(z + block_start + (k * stride))
                previous_input = tl.load(inputs_ptr + block_start + (r * stride))
                s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            k = k + 1
            tl.store(v + block_start + (k * stride), q)
            tl.store(z + block_start + (k * stride), s)
            if k + 1 < length:
                tl.store(z + block_start + ((k + 1) * stride), 1e9)

        k = 0
        for q in range(length):
            while (
                k + 1 < length
                and tl.load(
                    z + block_start + ((k + 1) * stride), mask=(k + 1) < length, other=q
                )
                < q
            ):
                k += 1
            r = tl.load(v + block_start + (k * stride))
            d = q - r
            old_value = tl.load(inputs_ptr + block_start + (r * stride))
            tl.store(outputs_ptr + block_start + (q * stride), old_value + d * d)


def edt_triton(data: torch.Tensor):
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images.
    """
    assert data.dim() == 3

    # Use OpenCV fallback on Windows or when Triton is not available
    if not TRITON_AVAILABLE or sys.platform == 'win32':
        return edt_opencv_fallback(data)

    assert data.is_cuda
    B, H, W = data.shape
    data = data.contiguous()

    output = torch.where(data, 1e18, 0.0)
    assert output.is_contiguous()

    parabola_loc = torch.zeros(B, H, W, dtype=torch.uint32, device=data.device)
    parabola_inter = torch.empty(B, H, W, dtype=torch.float, device=data.device)
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    grid = (B, H)
    edt_kernel[grid](
        output.clone(), output, parabola_loc, parabola_inter, H, W, horizontal=True,
    )

    parabola_loc.zero_()
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    grid = (B, W)
    edt_kernel[grid](
        output.clone(), output, parabola_loc, parabola_inter, H, W, horizontal=False,
    )
    return output.sqrt()


def edt_opencv_fallback(data: torch.Tensor) -> torch.Tensor:
    """
    OpenCV-based fallback for EDT when Triton is not available (e.g., on Windows).
    """
    import cv2
    import numpy as np

    device = data.device
    B, H, W = data.shape

    data_np = data.cpu().numpy()

    results = []
    for i in range(B):
        mask = (data_np[i] > 0).astype(np.uint8)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 0)
        results.append(dist)

    output = np.stack(results, axis=0)
    return torch.from_numpy(output).float().to(device)
```

## Step 9: Download SAM3 Model from HuggingFace

SAM3 is a gated model. You need:
1. HuggingFace account
2. Request access at https://huggingface.co/facebook/sam3
3. Generate access token at https://huggingface.co/settings/tokens

### For Corporate Networks (SSL Issues)

If you're behind a corporate proxy (Zscaler, etc.), use this script:

```python
# save as download_sam3.py and run with: venv_sam3\Scripts\python.exe download_sam3.py

import os
import ssl
import warnings
warnings.filterwarnings('ignore')

# Disable SSL verification for corporate proxy
ssl._create_default_https_context = ssl._create_unverified_context

# Set your HuggingFace token here
os.environ['HF_TOKEN'] = 'hf_YOUR_TOKEN_HERE'
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Patch httpx to disable SSL verification
import httpx
class InsecureClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        kwargs['verify'] = False
        super().__init__(*args, **kwargs)
httpx.Client = InsecureClient

from huggingface_hub import hf_hub_download

print('Downloading SAM3 config...')
hf_hub_download(repo_id='facebook/sam3', filename='config.json', token=os.environ['HF_TOKEN'])

print('Downloading SAM3 model (~2GB, please wait)...')
model_path = hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt', token=os.environ['HF_TOKEN'])

print(f'Model downloaded to: {model_path}')
print('SUCCESS!')
```

### For Normal Networks

```cmd
set HF_TOKEN=hf_YOUR_TOKEN_HERE
venv_sam3\Scripts\python.exe -c "from sam3.model_builder import build_sam3_image_model; build_sam3_image_model()"
```

## Step 10: Verify Installation

Create `test_sam3_env.py`:

```python
import sys
import os

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    print("=" * 60)
    print("SAM3 Environment Verification")
    print("=" * 60)

    # Check Python
    print(f"\n1. Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Check PyTorch/CUDA
    print("\n2. PyTorch/CUDA:")
    import torch
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Check SAM3
    print("\n3. SAM3:")
    import sam3
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.edt import TRITON_AVAILABLE
    print(f"   SAM3 imported: OK")
    print(f"   Triton available: {TRITON_AVAILABLE}")
    print(f"   Using OpenCV fallback: {not TRITON_AVAILABLE or sys.platform == 'win32'}")

    # Check dependencies
    print("\n4. Dependencies:")
    for mod in ['numpy', 'cv2', 'PIL', 'huggingface_hub', 'timm', 'decord', 'pycocotools']:
        try:
            __import__(mod)
            print(f"   {mod}: OK")
        except ImportError:
            print(f"   {mod}: MISSING")

    # Test model loading
    print("\n5. Model loading:")
    import warnings
    warnings.filterwarnings('ignore')
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

    model = build_sam3_image_model(device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Model loaded: OK")
    print(f"   Device: {next(model.parameters()).device}")

    print("\n" + "=" * 60)
    print("SAM3 is ready for use!")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

Run verification:
```cmd
venv_sam3\Scripts\python.exe test_sam3_env.py
```

Expected output:
```
============================================================
SAM3 Environment Verification
============================================================

1. Python: 3.12.x

2. PyTorch/CUDA:
   PyTorch: 2.7.0+cu126
   CUDA available: True
   GPU: NVIDIA RTX xxxx
   Memory: xx.x GB

3. SAM3:
   SAM3 imported: OK
   Triton available: False
   Using OpenCV fallback: True

4. Dependencies:
   numpy: OK
   cv2: OK
   PIL: OK
   huggingface_hub: OK
   timm: OK
   decord: OK
   pycocotools: OK

5. Model loading:
   Model loaded: OK
   Device: cuda:0

============================================================
SAM3 is ready for use!
============================================================
```

## Quick Reference

### Activate Environment
```cmd
cd C:\path\to\your\project
venv_sam3\Scripts\activate
```

### Run SAM3 Annotator
```cmd
python sam3_annotator.py <input_folder> --prompt "mouse" --device cuda
```

### Package Versions Summary

| Package | Version |
|---------|---------|
| Python | 3.12.x |
| PyTorch | 2.7.0+cu126 |
| CUDA | 12.6 |
| SAM3 | 0.1.0 |
| numpy | 1.26.0 |
| opencv-python | 4.12.x |

## Troubleshooting

### "No module named 'triton'"
This is expected on Windows. The EDT fallback using OpenCV handles this automatically.

### SSL Certificate Errors
Use the download script in Step 9 with SSL verification disabled.

### CUDA Out of Memory
- Model uses ~3.2GB VRAM
- Reduce batch size or image resolution
- Use `torch.cuda.empty_cache()` between batches

### Model not found
Ensure the model is downloaded to:
```
%USERPROFILE%\.cache\huggingface\hub\models--facebook--sam3\
```

## File Structure After Setup

```
your_project/
├── venv_sam3/                    # Python virtual environment
│   ├── Scripts/
│   │   ├── python.exe           # Use this Python
│   │   └── activate             # Activation script
│   └── Lib/
├── sam3_repo/                    # SAM3 source code
│   └── sam3/
│       └── model/
│           └── edt.py           # Modified for Windows
├── test_sam3_env.py             # Verification script
├── download_sam3.py             # Model download script
└── SAM3_SETUP_README.md         # This file
```

## One-Line Setup (Copy-Paste)

For quick setup on a new PC, run these commands in order:

```cmd
:: Step 1-4: Create environment
py -3.12 -m venv venv_sam3 && venv_sam3\Scripts\python.exe -m pip install --upgrade pip

:: Step 5: Install PyTorch
venv_sam3\Scripts\python.exe -m pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

:: Step 6: Clone and install SAM3
git clone https://github.com/facebookresearch/sam3.git sam3_repo && venv_sam3\Scripts\python.exe -m pip install -e sam3_repo

:: Step 7: Install dependencies
venv_sam3\Scripts\python.exe -m pip install opencv-python decord pycocotools psutil pip-system-certs truststore
```

Then manually:
1. Replace `sam3_repo/sam3/model/edt.py` with Windows version (Step 8)
2. Download model with your HF token (Step 9)
3. Verify installation (Step 10)

---
*Last updated: December 2025*
*SAM3 GitHub: https://github.com/facebookresearch/sam3*

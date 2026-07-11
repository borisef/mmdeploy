# Minimum System Requirements — `my_wrap_deploy1.py` (Python 3.8)

Requirements for running `tools/boris/my_wrap_deploy1.py` and reproducing its
ONNX-runtime inference performance on GPU.

---

## Hardware

| Component | Minimum / Tested |
|-----------|-----------------|
| GPU | NVIDIA GPU with CUDA compute capability ≥ 6.1 (Pascal+) |
| GPU VRAM | ≥ 4 GB (tested on Quadro T1000 — 4 GB) |
| RAM | ≥ 8 GB system RAM |
| Disk | ≥ 5 GB free (model checkpoints + ONNX outputs) |

> Tested GPU: **NVIDIA Quadro T1000 (4 GB)**

---

## NVIDIA Driver & CUDA

| Component | Version |
|-----------|---------|
| NVIDIA driver (Linux) | ≥ 510.39.01 (required for CUDA 11.6 runtime) |
| Tested driver | 470.256.02 (via forward-compat layer) |
| CUDA runtime | 11.6 — bundled inside the PyTorch wheel (`+cu116`), no separate toolkit install needed |
| cuDNN | 8.x — bundled inside the PyTorch wheel at `torch/lib/libcudnn.so.8` |

> **Important:** the script explicitly pre-loads `libcudnn.so.8` from inside the
> PyTorch package directory (`torch/__file__/lib/libcudnn.so.8`) before starting
> the ONNX Runtime CUDA provider. This means cuDNN performance is governed by
> the version that ships with the PyTorch wheel, not any system-installed cuDNN.
> A mismatched system cuDNN on `LD_LIBRARY_PATH` can silently override this and
> cause different FPS or runtime errors.

---

## Python

| Component | Version |
|-----------|---------|
| Python | **3.8.x** (tested: 3.8.10) |

Python 3.9+ is **not** validated — several mmdeploy / mmcv CUDA extension
builds are sensitive to the CPython ABI.

---

## Python Packages (pinned)

These versions are required for numerical reproducibility and FPS parity.
Changing any of the first four will likely shift inference speed.

| Package | Version | Notes |
|---------|---------|-------|
| `torch` | **1.13.1+cu116** | Bundles CUDA 11.6 runtime and cuDNN 8; must match `+cu116` suffix |
| `torchvision` | **0.14.1+cu116** | Must match torch exactly |
| `onnxruntime-gpu` | **1.16.3** | Graph optimizations differ between releases; pin for FPS parity |
| `mmcv` | **2.0.0** | CUDA ops are compiled against specific torch+CUDA versions |
| `mmengine` | **0.9.1** | Config and pipeline management |
| `onnx` | **1.17.0** | Export format; mismatches can fail IR validation |
| `onnxconverter-common` | **1.16.0** | Required if `USE_FP16 = True` |
| `numpy` | **1.24.3** | Input array construction; 2.x series is ABI-incompatible |
| `opencv-python` | **4.7.0.72** | Image decode for preprocessing and visualization overlay |

Install example:

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116

pip install \
    onnxruntime-gpu==1.16.3 \
    onnx==1.17.0 \
    onnxconverter-common==1.16.0 \
    mmengine==0.9.1 \
    numpy==1.24.3 \
    "opencv-python==4.7.0.72"

# mmcv must be built for the exact torch+CUDA version:
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
```

---

## mmpose

`mmpose` must be installed (or on `PYTHONPATH`) at a version compatible with
`mmengine 0.9.1` and the model config used (HRNet-W32 UDP).  
The model config path is hardcoded in the script — verify it exists before
running.

---

## Environment Variables

| Variable | Value | When needed |
|----------|-------|-------------|
| `MMDEPLOY_ORT_OPTIMIZE_ALL` | `1` | Only when `ORT_OPTIMIZE_ALL = True` in script |

---

## What Affects FPS

Ranked by impact:

1. **GPU model** — different GPU generations (Turing vs Ampere etc.) produce
   very different numbers regardless of software stack.
2. **`onnxruntime-gpu` version** — graph optimization passes change between
   releases; 1.16.3 is the validated version.
3. **cuDNN version** — bundled with PyTorch; do not override with system cuDNN.
4. **`torch` version** — CUDA kernel selection differs between releases.
5. **`mmcv` version** — CUDA-compiled ops affect preprocessing speed.
6. **System load / power mode** — GPU power management can throttle clocks;
   run `nvidia-smi -pm 1` to enable persistence mode on a dedicated machine.

---

## Capturing a Reproducible Snapshot

Run on the target machine before archiving a deployment:

```bash
pip freeze > requirements_frozen.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python3 -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)"
python3 -c "import torch; print('cuDNN:', torch.backends.cudnn.version())"
python3 -c "import onnxruntime; print('ORT:', onnxruntime.__version__, '| providers:', onnxruntime.get_available_providers())"
```

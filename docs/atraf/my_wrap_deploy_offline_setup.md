# Offline setup guide for `my_wrap_deploy1.py`

- [Offline setup guide for `my_wrap_deploy1.py`](#offline-setup-guide-for-my_wrap_deploy1py)
  - [Purpose](#purpose)
  - [What this guide assumes](#what-this-guide-assumes)
  - [Verified working runtime combination](#verified-working-runtime-combination)
  - [Important file paths in this project](#important-file-paths-in-this-project)
  - [System requirements](#system-requirements)
  - [Python environment requirements](#python-environment-requirements)
  - [Required Python packages](#required-python-packages)
  - [Required GPU runtime pieces for ONNX Runtime](#required-gpu-runtime-pieces-for-onnx-runtime)
  - [Recommended offline package cache](#recommended-offline-package-cache)
  - [How to prepare an offline machine](#how-to-prepare-an-offline-machine)
    - [1. Create the Python environment](#1-create-the-python-environment)
    - [2. Install PyTorch with CUDA support](#2-install-pytorch-with-cuda-support)
    - [3. Install MMDeploy and codebase dependencies](#3-install-mmdeploy-and-codebase-dependencies)
    - [4. Install ONNX Runtime GPU](#4-install-onnx-runtime-gpu)
    - [5. Install cuDNN 9 for CUDA 12](#5-install-cudnn-9-for-cuda-12)
    - [6. Verify CUDA libraries are visible](#6-verify-cuda-libraries-are-visible)
    - [7. Verify ONNX Runtime CUDA provider](#7-verify-onnx-runtime-cuda-provider)
    - [8. Run `my_wrap_deploy1.py`](#8-run-my_wrap_deploy1py)
  - [Expected successful output](#expected-successful-output)
  - [Why ONNX previously ran on CPU](#why-onnx-previously-ran-on-cpu)
  - [Workarounds already implemented in `my_wrap_deploy1.py`](#workarounds-already-implemented-in-my_wrap_deploy1py)
  - [Offline troubleshooting cookbook](#offline-troubleshooting-cookbook)
    - [Problem: MMPose metainfo file not found](#problem-mmpose-metainfo-file-not-found)
    - [Problem: ONNX Runtime says `libcublasLt.so.11` is missing](#problem-onnx-runtime-says-libcublasltso11-is-missing)
    - [Problem: ONNX Runtime says `libcudnn.so.9` or `libcudnn_adv.so.9` is missing](#problem-onnx-runtime-says-libcudnnso9-or-libcudnn_advso9-is-missing)
    - [Problem: ONNX reports `CPUExecutionProvider` only](#problem-onnx-reports-cpuexecutionprovider-only)
    - [Problem: PyTorch uses GPU but ONNX uses CPU](#problem-pytorch-uses-gpu-but-onnx-uses-cpu)
    - [Problem: `pip install nvidia-cudnn-cu12` is too large or too slow](#problem-pip-install-nvidia-cudnn-cu12-is-too-large-or-too-slow)
    - [Problem: no internet on target machine](#problem-no-internet-on-target-machine)
  - [Offline validation commands](#offline-validation-commands)
  - [Minimal wheel list to carry to an offline machine](#minimal-wheel-list-to-carry-to-an-offline-machine)
  - [Notes for future maintenance](#notes-for-future-maintenance)

______________________________________________________________________

## Purpose

This document explains how to prepare a Linux machine so that
`tools/boris/my_wrap_deploy1.py` can:

1. export the model to ONNX,
2. run the ONNX model with ONNX Runtime,
3. use **GPU** for the ONNX FPS benchmark,
4. work even on a machine that is not connected to the web.

This guide is based on the issues already observed and fixed in this workspace.
It is intentionally practical and machine-oriented.

## What this guide assumes

- OS: Linux
- Python environment style: `venv` or similar isolated Python env
- GPU vendor: NVIDIA
- You want ONNX Runtime GPU inference, not CPU-only inference
- You are using the project-local script:
  - `tools/boris/my_wrap_deploy1.py`

## Verified working runtime combination

The following combination was verified in this workspace:

- Python executable from environment similar to:
  - `/home/borisef/Envs/mmpose/bin/python`
- PyTorch:
  - `torch 2.1.0+cu121`
- CUDA runtime visible on system:
  - CUDA 12.1 / 12.2 class libraries present
- NVIDIA driver:
  - any driver compatible with CUDA 12.x runtime on the machine
- ONNX Runtime GPU:
  - `onnxruntime-gpu==1.20.1`
- cuDNN runtime package:
  - `nvidia-cudnn-cu12==9.10.2.21`

Important conclusion:

- `onnxruntime-gpu==1.20.1` expects **CUDA 12 + cuDNN 9**.
- Older ONNX Runtime GPU builds may expect **CUDA 11** libraries such as
  `libcublasLt.so.11` and will silently fall back to CPU if those are missing.

## Important file paths in this project

Main script:

- `tools/boris/my_wrap_deploy1.py`

MMDeploy config:

- `configs/mmpose/pose-detection_onnxruntime_static.py`

Example image:

- `demo/resources/human-pose.jpg`

Experimental model config used by the script:

- `/home/borisef/projects/mm/mmpose/tools/atraf/borisef/work_dirs/hrnet_UDP_w32_try_changes/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288_try_changes.py`

Checkpoint used by the script:

- `/home/borisef/projects/mm/mmpose/tools/atraf/borisef/work_dirs/hrnet_UDP_w32_try_changes/epoch_11.pth`

Default output directory:

- `/home/borisef/temp/out_mmdeploy_try`

## System requirements

For ONNX Runtime GPU benchmarking, the machine must have:

- NVIDIA GPU
- NVIDIA driver installed and working
- CUDA 12 runtime libraries available on the system, or otherwise accessible
- enough disk space for:
  - Python environment
  - model checkpoint
  - exported ONNX model
  - large wheel files for offline install

Quick validation:

```bash
nvidia-smi
```

You should see your GPU and driver information.

## Python environment requirements

Use one dedicated Python environment for MMDeploy + MMPose + ONNX Runtime.
Do not mix multiple Python environments during debugging.

Quick validation:

```bash
python -u - <<'PY'
import sys
print(sys.executable)
PY
```

The path printed here must be the same interpreter used to run
`my_wrap_deploy1.py`.

## Required Python packages

At minimum, the environment must include:

- `torch`
- `torchvision` if required by your stack
- `mmengine`
- `mmcv`
- `mmpose`
- `mmdeploy`
- `onnxruntime-gpu`
- `nvidia-cudnn-cu12`

Depending on your local setup, additional packages may also be needed by
MMPose/MMDeploy.

## Required GPU runtime pieces for ONNX Runtime

For the verified working path, ONNX Runtime GPU needed all of these classes of
libraries:

- CUDA 12 runtime
  - `libcudart.so.12`
  - `libcublas.so.12`
  - `libcublasLt.so.12`
  - `libcufft.so.11`
  - `libcurand.so.10`
  - `libnvrtc.so.12`
- cuDNN 9 runtime
  - `libcudnn.so.9`
  - `libcudnn_adv.so.9`
  - `libcudnn_ops.so.9`
  - `libcudnn_cnn.so.9`
  - `libcudnn_graph.so.9`
  - `libcudnn_engines_runtime_compiled.so.9`
  - `libcudnn_engines_precompiled.so.9`
  - `libcudnn_heuristic.so.9`

## Recommended offline package cache

For a machine without internet, prepare a directory such as:

```text
offline_wheels/
```

and download all required wheels on a machine that has internet.

Recommended contents include at least:

- `onnxruntime_gpu-1.20.1-...whl`
- `nvidia_cudnn_cu12-9.10.2.21-...whl`
- wheels for `torch`, `torchvision`, `mmengine`, `mmcv`, `mmpose`, and local dependencies

For source checkouts, also carry:

- the `mmdeploy` repo
- the `mmpose` repo or the exact installed package source snapshot
- your checkpoint files
- your custom config files

## How to prepare an offline machine

### 1. Create the Python environment

Example:

```bash
python3 -m venv /opt/venvs/mmpose
source /opt/venvs/mmpose/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install PyTorch with CUDA support

Install from local wheel files prepared in advance.

Example:

```bash
python -m pip install /path/to/offline_wheels/torch-*.whl
python -m pip install /path/to/offline_wheels/torchvision-*.whl
```

Validate:

```bash
python -u - <<'PY'
import torch
print('torch =', torch.__version__)
print('torch cuda =', torch.version.cuda)
print('cuda available =', torch.cuda.is_available())
PY
```

Expected:

- `cuda available = True`

### 3. Install MMDeploy and codebase dependencies

If installing from local repositories:

```bash
cd /path/to/mmdeploy
python -m pip install -e .

cd /path/to/mmpose
python -m pip install -e .
```

Install other required wheels offline as needed:

```bash
python -m pip install /path/to/offline_wheels/mmengine-*.whl
python -m pip install /path/to/offline_wheels/mmcv-*.whl
```

### 4. Install ONNX Runtime GPU

Use the verified build:

```bash
python -m pip install /path/to/offline_wheels/onnxruntime_gpu-1.20.1-*.whl
```

Validate:

```bash
python -u - <<'PY'
import onnxruntime as ort
print('onnxruntime =', ort.__version__)
print('available providers =', ort.get_available_providers())
PY
```

Expected output should contain:

- `CUDAExecutionProvider`

Note:

- this only means the Python package includes CUDA support,
- it does **not** yet guarantee that all runtime libraries are available.

### 5. Install cuDNN 9 for CUDA 12

Use the verified package:

```bash
python -m pip install /path/to/offline_wheels/nvidia_cudnn_cu12-9.10.2.21-*.whl
```

Validate that the files exist:

```bash
python -u - <<'PY'
import glob, os, site
found = []
for base in site.getsitepackages():
    found.extend(sorted(glob.glob(os.path.join(base, 'nvidia', 'cudnn', 'lib', 'libcudnn*.so*'))))
print('\n'.join(found) if found else '<none>')
PY
```

Expected output should list files like:

- `libcudnn.so.9`
- `libcudnn_adv.so.9`
- `libcudnn_ops.so.9`
- `libcudnn_cnn.so.9`

### 6. Verify CUDA libraries are visible

System check:

```bash
ldconfig -p | grep -E 'libcublasLt\.so\.12|libcublas\.so\.12|libcudart\.so\.12|libcufft\.so\.11|libcurand\.so\.10' | cat
```

If your system libraries are not globally visible, you may need:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

### 7. Verify ONNX Runtime CUDA provider

Use this direct validation before running the full script:

```bash
python -u - <<'PY'
import os.path as osp
import importlib.util

module_path = '/home/borisef/projects/mm/mmdeploy/tools/boris/my_wrap_deploy1.py'
spec = importlib.util.spec_from_file_location('my_wrap_deploy1', module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

sess = mod._create_ort_session(
    osp.join(mod.work_dir, 'end2end.onnx'),
    require_cuda=True)

print(sess.get_providers())
PY
```

Expected:

```text
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 8. Run `my_wrap_deploy1.py`

```bash
python -u /home/borisef/projects/mm/mmdeploy/tools/boris/my_wrap_deploy1.py
```

## Expected successful output

For a good GPU ONNX run, logs should include something like:

```text
ONNX Runtime 1.20.1 active providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

And later in the FPS summary:

```text
PyTorch : GPU (CUDA) ...
ONNX    : GPU (CUDA) ...
```

## Why ONNX previously ran on CPU

Two failures were seen during bring-up:

1. old ORT build wanted CUDA 11 libraries
   - error included `libcublasLt.so.11`
2. newer ORT build was correct for CUDA 12, but lacked cuDNN 9
   - error included `libcudnn.so.9`
   - later `libcudnn_adv.so.9`

In both cases, ONNX Runtime silently fell back to `CPUExecutionProvider` unless
explicitly checked.

## Workarounds already implemented in `my_wrap_deploy1.py`

The script now includes practical workarounds:

1. **MMPose metainfo auto-fix**
   - unwraps `ConcatDataset` / related wrappers
   - injects absolute metainfo file paths when needed

2. **Explicit ONNX GPU enforcement**
   - `REQUIRE_ONNX_GPU_FOR_FPS = True`
   - if CUDA provider is not active, the script raises an error instead of
     silently benchmarking CPU

3. **CUDA/cuDNN preloading**
   - preloads CUDA and cuDNN shared libraries from:
     - `torch/lib`
     - pip-installed `nvidia/*/lib`
     - common system CUDA library directories

4. **Full cuDNN 9 component preloading**
   - not only `libcudnn.so.9`
   - also `libcudnn_adv.so.9`, `libcudnn_ops.so.9`, etc.

## Offline troubleshooting cookbook

### Problem: MMPose metainfo file not found

Symptom:

```text
FileNotFoundError: The metainfo config file "configs/_base_/datasets/coco.py" does not exist.
```

Meaning:

- your model config references dataset metainfo indirectly,
- mmdeploy cannot resolve it using the raw relative path.

Fix:

- use the patched `my_wrap_deploy1.py`
- make sure the local `mmpose` checkout exists and contains:

```text
configs/_base_/datasets/coco.py
```

### Problem: ONNX Runtime says `libcublasLt.so.11` is missing

Symptom:

```text
libcublasLt.so.11: cannot open shared object file
```

Meaning:

- your ORT GPU wheel is a CUDA 11 era build,
- but your machine/runtime stack is CUDA 12.

Fix:

- replace the ORT build with:

```bash
python -m pip install --upgrade onnxruntime-gpu==1.20.1
```

Offline:

- carry the `onnxruntime_gpu-1.20.1-*.whl` file to the target machine.

### Problem: ONNX Runtime says `libcudnn.so.9` or `libcudnn_adv.so.9` is missing

Meaning:

- ORT is now the correct CUDA 12 build,
- but the cuDNN 9 runtime is not installed or not visible.

Fix:

```bash
python -m pip install nvidia-cudnn-cu12==9.10.2.21
```

Offline:

- carry the `nvidia_cudnn_cu12-9.10.2.21-*.whl` file.

### Problem: ONNX reports `CPUExecutionProvider` only

Check:

```bash
python -u - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

If `CUDAExecutionProvider` is not listed:

- wrong ORT wheel installed,
- CPU-only ONNX Runtime installed,
- wrong Python environment active.

### Problem: PyTorch uses GPU but ONNX uses CPU

Meaning:

- GPU driver and PyTorch are fine,
- ONNX Runtime still lacks one of its own runtime dependencies.

Typical causes:

- wrong ORT GPU wheel version
- missing cuDNN 9 runtime
- running from a different Python environment than expected

### Problem: `pip install nvidia-cudnn-cu12` is too large or too slow

Notes:

- the wheel is very large,
- download it once on a connected machine,
- copy the wheel file by USB or local network,
- then install from disk on the offline target.

Recommended offline pattern:

```bash
python -m pip install /path/to/nvidia_cudnn_cu12-9.10.2.21-*.whl
```

### Problem: no internet on target machine

Prepare everything in advance on an online machine:

- Python wheel cache directory
- source repos
- model checkpoint
- custom configs
- this documentation file

Then copy to the offline machine.

## Offline validation commands

### Check GPU driver

```bash
nvidia-smi
```

### Check Python environment

```bash
python -u - <<'PY'
import sys
print(sys.executable)
PY
```

### Check PyTorch GPU

```bash
python -u - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

### Check ONNX Runtime package

```bash
python -u - <<'PY'
import onnxruntime as ort
print(ort.__version__)
print(ort.get_available_providers())
PY
```

### Check cuDNN 9 files from pip package

```bash
python -u - <<'PY'
import glob, os, site
for base in site.getsitepackages():
    for p in sorted(glob.glob(os.path.join(base, 'nvidia', 'cudnn', 'lib', 'libcudnn*.so*'))):
        print(p)
PY
```

### Check ORT provider dependencies

```bash
ORT_DIR=$(python -u - <<'PY'
import inspect, os, onnxruntime as ort
print(os.path.dirname(inspect.getfile(ort)))
PY
)
ldd "$ORT_DIR/capi/libonnxruntime_providers_cuda.so" | cat
```

### Check actual ONNX GPU binding

```bash
python -u - <<'PY'
import os.path as osp
import importlib.util
module_path = '/home/borisef/projects/mm/mmdeploy/tools/boris/my_wrap_deploy1.py'
spec = importlib.util.spec_from_file_location('my_wrap_deploy1', module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sess = mod._create_ort_session(osp.join(mod.work_dir, 'end2end.onnx'), require_cuda=True)
print(sess.get_providers())
PY
```

## Minimal wheel list to carry to an offline machine

At minimum, carry these exact wheels or locally equivalent versions:

- `onnxruntime_gpu-1.20.1-...whl`
- `nvidia_cudnn_cu12-9.10.2.21-...whl`
- your exact `torch` wheel
- your exact `torchvision` wheel if used
- wheels for:
  - `mmengine`
  - `mmcv`
  - any other Python dependency not already baked into the environment

Also carry:

- a text file with all versions
- the exported wheel filenames
- this setup guide

## Notes for future maintenance

If ONNX GPU stops working after package upgrades, check these first:

1. `python -c "import onnxruntime as ort; print(ort.__version__)"`
2. `ldd libonnxruntime_providers_cuda.so`
3. whether `libcudnn*.so.9` still exists in the environment
4. whether the script still preloads the full cuDNN 9 library family

If PyTorch is upgraded to a different CUDA generation, re-check that the ORT
build and cuDNN package still match it.


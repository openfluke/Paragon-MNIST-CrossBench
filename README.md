# Paragon AI Framework Showcase

This repository demonstrates the **Paragon AI Framework**, a high-performance, Go-based library for training and benchmarking neural networks, with a focus on the MNIST dataset. It leverages WebGPU for GPU acceleration and includes ADHD (Accuracy Deviation Heatmap Distribution) performance metrics.

## Overview

The Paragon AI Framework enables flexible neural network configuration and efficient training on both CPU and GPU. This example showcases:

- Loading and preprocessing the MNIST dataset (56,000 train, 14,000 test samples).
- Training a neural network with customizable layers and activations.
- Evaluating performance using ADHD metrics (e.g., 99.13% train, 97.62% test accuracy).
- Saving/loading models in JSON format.
- GPU acceleration via WebGPU on NVIDIA GPUs (e.g., RTX 3050 Mobile).

## Prerequisites

- **Go**: Version 1.18 or later.
- **MNIST Dataset**: Files (`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`) in `./data/mnist`.
- **Hardware**: WebGPU-compatible GPU (e.g., NVIDIA RTX 3050 Mobile) for acceleration, or CPU fallback.
- **OS**: Tested on Fedora 41 (Workstation Edition) with NVIDIA driver 575.57.08.
- **Dependencies**: `github.com/openfluke/paragon/v3` and `github.com/openfluke/pilot`.

## Setup

1. **Clone the Repository**:

   ```bash
   https://github.com/openfluke/Paragon-MNIST-CrossBench
   ```

2. **Initialize Go Module**:

   ```bash
   go mod init main
   ```

3. **Install Dependencies**:

   ```bash
   go get github.com/openfluke/paragon/v3@v3.1.0
   go get github.com/openfluke/pilot@v0.0.2
   go get github.com/openfluke/webgpu@ea0f165
   ```

4. **Downloads MNIST Dataset**:

5. **Install NVIDIA Drivers** (for GPU acceleration or whatever gpu drivers needed):

## Running the Example

1. **Ensure MNIST Dataset** is in `./data/mnist`.
2. **Run with GPU Acceleration** (NVIDIA RTX 3050):

   ```bash
   __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./paragon-linux
   ```

   - Uses RTX 3050 Mobile (55 MiB VRAM, 75% utilization at 38W).
   - Disable `__NV_PRIME_RENDER_OFFLOAD` to use Intel Iris Xe iGPU (slower, ~57.5s/epoch).

3. **Run Directly** (CPU fallback if WebGPU fails):
   ```bash
   go run engine.go mnist.go
   ```

## Expected Output

The program:

1. Loads MNIST dataset (56,000 train, 14,000 test samples).
2. Trains a neural network (784-1024-10 MLP, linear/relu/softmax activations) for 10 epochs.
3. Evaluates with ADHD metrics (e.g., 98.68% train samples in 0–10% deviation).
4. Saves the model to `./models/mnist_model.json`.

Sample output (RTX 3050 Mobile, 10 epochs):

```
🚀 Running experiment: MNIST
⚙ Stage: MNIST Dataset Prep
🔧 Running MNIST dataset stage...
✅ All stages completed successfully.
📊 Dataset sizes: Train=56000, Test=14000
⏱ Data Prep Time: 295.481013ms
⏱ Network Init Time: 12.777015ms
[wgpu] [Warn] No config found!
[wgpu] [Warn] EGL says it can present to the window but not natively
✅ WebGPU initialized successfully
⏱ WebGPU Init Time: 211.233114ms
🧠 Training the network...
COPY WEIGHTS FROM GPU TO CPU
Epoch 0, Loss: 0.9763
...
Epoch 9, Loss: 0.0580
⏱ Total Training Time: 3m19.591917042s
📈 ADHD Performance (Train Set):
- 0-10%: 55263 samples (98.68%)
- 10-20%: 108 samples (0.19%)
- 20-30%: 65 samples (0.12%)
- 30-40%: 50 samples (0.09%)
- 40-50%: 43 samples (0.08%)
- 50-100%: 224 samples (0.40%)
- 100%+: 247 samples (0.44%)
- Total Samples: 56000
- Failures (100%+): 247 (0.44%)
- Score: 99.1250%
⏱ Evaluate Time (Train): 10.208314166s
📈 ADHD Performance (Test Set):
- 0-10%: 13503 samples (96.45%)
- 10-20%: 53 samples (0.38%)
- 20-30%: 32 samples (0.23%)
- 30-40%: 50 samples (0.36%)
- 40-50%: 45 samples (0.32%)
- 50-100%: 169 samples (1.21%)
- 100%+: 148 samples (1.06%)
- Total Samples: 14000
- Failures (100%+): 148 (1.06%)
- Score: 97.6225%
⏱ Evaluate Time (Test): 2.589343557s
📊 Train Score: 99.1250%
📊 Test Score: 97.6225%
⏱ Evaluation Time: 12.79766733s
💾 Saved model to models/mnist_model.json
⏱ Model Save Time: 472.617888ms
⏱ Total Experiment Time: 3m33.383668609s
```

## Features of Paragon

- **Flexible Architecture**: Configurable layers (e.g., 784-1024-10), activations (linear, relu, softmax), and connectivity (fully connected or local).
- **WebGPU Acceleration**: ~19.96s/epoch on RTX 3050 Mobile (55 MiB VRAM, 75% utilization), ~3x faster than Intel Iris Xe iGPU (~57.5s/epoch).
- **ADHD Benchmarking**: Detailed performance metrics (e.g., 98.68% train samples in 0–10% deviation, 0.44% failures).
- **Model Persistence**: JSON-based model saving/loading (`mnist_model.json`).
- **Cross-Platform**: Supports Linux (Fedora 41), Windows, and macOS via WebGPU.

## Notes

- **Performance**: Single-sample processing achieves ~19.96s/epoch. Batching could reduce this to ~5–10s/epoch, but the current speed outperforms TensorFlow/PyTorch without batching (~80–150s/epoch).
- **Hardware**: Tested on Fedora 41 with NVIDIA RTX 3050 Mobile (4GB VRAM, driver 575.57.08) and Intel i5-12500H (16 cores, 4.5 GHz, 48GB RAM).
- **WebGPU Warnings**: `[wgpu] [Warn] No config found!` and `EGL` messages are benign, common on Fedora/Wayland with NVIDIA drivers.
- **Scalability**: Best for small-to-medium models (~200K–10M parameters). Large models (e.g., transformers) require shader extensions and batching.

## Building Executables

1. **Single Platform**:

   ```bash
   go build -o paragon-linux engine.go mnist.go
   ```

   Compile for windows from linux

   ```
   env GOOS=windows GOARCH=amd64 CGO_ENABLED=1 CC=x86_64-w64-mingw32-gcc CXX=x86_64-w64-mingw32-g++ go build -o paragon-windows.exe engine.go mnist.go
   ```

2. **Multiple Platforms**:
   Use `build-all.sh` to compile for Linux, Windows, and macOS (amd64):
   ```bash
   chmod +x build-all.sh
   ./build-all.sh
   ```
   - Outputs: `build/mnist_benchmark_linux_amd64`, `build/mnist_benchmark_windows_amd64.exe`, `build/mnist_benchmark_darwin_amd64`.
   - ARM platforms are disabled to avoid dependency issues.

## Setting Up Cross-Compilation for Windows

On Fedora 41, install the MinGW-w64 toolchain for Windows builds:

```bash
sudo dnf install -y mingw64-gcc mingw64-gcc-c++
chmod +x setup-cross-compile.sh
./setup-cross-compile.sh
```

## Troubleshooting

- **WebGPU Errors**: Ensure NVIDIA drivers are installed and use `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia` for RTX 3050.
- **Dependency Issues**: Update `go-webgpu`:
  ```bash
  go get github.com/openfluke/webgpu@ea0f165
  ```
- **Dataset Missing**: Verify `./data/mnist` contains MNIST files.

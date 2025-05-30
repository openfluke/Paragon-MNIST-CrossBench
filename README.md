# Paragon AI Framework Showcase

This README demonstrates how to use the **Paragon AI Framework** to train and benchmark a neural network on the MNIST dataset, leveraging both CPU and GPU acceleration with WebGPU support.

## Overview

The Paragon AI Framework is a high-performance library for building and training neural networks. This example showcases:

- Loading and preparing the MNIST dataset.
- Training a neural network with customizable layers and activations.
- Saving/loading models in JSON format.
- Benchmarking inference performance on CPU and GPU.

## Prerequisites

- Go 1.18 or later
- Access to the MNIST dataset files
- WebGPU-compatible hardware for GPU acceleration (optional)

## Setup

1. **Initialize the Go module**:

   ```bash
   go mod init main
   ```

2. **Install Paragon and Pilot dependencies**:

   ```bash
   go get github.com/openfluke/paragon@v1.0.0
   go get github.com/openfluke/pilot@v0.0.2
   ```

3. **Download MNIST dataset**:
   Place the MNIST dataset files (`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, etc.) in a `./data/mnist` directory.

## Running the Example

1. Ensure the MNIST dataset is in `./data/mnist`.
2. Run the program (assuming you have the example code in `main.go`):
   ```bash
   go run main.go
   ```

## Expected Output

The program will:

1. Prepare and load the MNIST dataset.
2. Train a neural network (or load a pre-trained model).
3. Save the trained model to `mnist_model_float32.json`.
4. Benchmark inference on CPU and GPU.

Sample output:

```
🚀 Preparing MNIST Dataset...
🔧 Running MNIST dataset stage...
📦 Loading MNIST data into memory...
🧠 Training float32 model...
Epoch 0, Loss: 20.7233
Epoch 1, Loss: 2.3026
Epoch 2, Loss: 2.3026
💾 Saving trained model...

⏱️ Benchmarking inference...
[wgpu] [Warn] Detected skylake derivative running on mesa i915. Clears to srgb textures will use manual shader clears.

✅ CPU: 5.132569294s
✅ GPU: 855.346851ms
⚡ Speedup: 6.00x
```

## Features of Paragon

- **Flexible Network Configuration**: Define layers, activations, and connectivity.
- **WebGPU Acceleration**: Achieve significant speedups (e.g., 6x in this example).
- **Model Persistence**: Save/load models in JSON format.
- **MNIST Dataset Support**: Seamless integration with the Pilot library for dataset handling.

## Notes

- The example limits the number of training samples for faster execution. Adjust the `trainLimit` constant in the code to train on the full dataset.
- WebGPU support requires compatible hardware and drivers.
- The `[wgpu] [Warn]` message may appear on some systems but does not affect functionality.

## Building Executables

To build the application for multiple platforms, use the provided `build-all.sh` script:

```bash
sh build-all.sh
```

## Setting Up Cross-Compilation for Windows

To cross-compile the Go project for Windows on Fedora or Ubuntu, you need to install the MinGW-w64 toolchain. Below are the setup instructions for each distribution, followed by building the executables.

### Setup Script

Save the following script as `setup-cross-compile.sh` to install the necessary tools:

```bash
#!/bin/bash

echo "🚀 Setting up cross-compilation environment for Windows..."

# Detect the Linux distribution
if [ -f /etc/fedora-release ]; then
    echo "🔧 Detected Fedora, installing mingw64-gcc..."
    sudo dnf install -y mingw64-gcc mingw64-gcc-c++
elif [ -f /etc/lsb-release ] || [ -f /etc/debian_version ]; then
    echo "🔧 Detected Ubuntu, installing gcc-mingw-w64..."
    sudo apt-get update
    sudo apt-get install -y gcc-mingw-w64 g++-mingw-w64
else
    echo "❌ Unsupported distribution. This script supports Fedora and Ubuntu."
    exit 1
fi

echo "✅ Setup complete! You can now cross-compile for Windows."
```

### Explanation

- **Fedora**: Installs `mingw64-gcc` and `mingw64-gcc-c++` for C and C++ cross-compilation support, as required for Go's CGO if needed (e.g., for dependencies like `go-webgpu`).[](https://stackoverflow.com/questions/41566495/golang-how-to-cross-compile-on-linux-for-windows)
- **Ubuntu**: Installs `gcc-mingw-w64` and `g++-mingw-w64` to provide the Windows cross-compiler toolchain.[](https://stackoverflow.com/questions/41566495/golang-how-to-cross-compile-on-linux-for-windows)
- **Script Logic**: Detects the distribution by checking for `/etc/fedora-release` (Fedora) or `/etc/lsb-release`/`/etc/debian_version` (Ubuntu), then installs the appropriate packages.
- **Integration**: This setup ensures compatibility with your `build-all.sh` script, which builds for Windows (amd64/arm64) among other platforms.

### Addressing Your Error

The error you encountered (`undefined: Limits` in `go-webgpu/wgpu`) suggests a dependency issue with the `go-webgpu` package, likely due to an outdated or incompatible version. To resolve:

1. Update the dependency in your `go.mod`:
   ```bash
   go get github.com/rajveermalviya/go-webgpu/wgpu@latest
   ```

### To build from linux to windows you will need to install

```
go get github.com/rajveermalviya/go-webgpu/wgpu@latest
```

### Also this could help if on fedora

```
sudo dnf install -y mingw64-gcc mingw64-gcc-c++
```

### Disabled Platforms

Support for ARM (`windows/arm64`, `linux/arm64`) and macOS (`darwin/amd64`, `darwin/arm64`) targets is disabled in the `build-all.sh` script for Linux x86_64 environments to avoid dependency conflicts and ensure build stability.

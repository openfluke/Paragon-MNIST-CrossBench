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

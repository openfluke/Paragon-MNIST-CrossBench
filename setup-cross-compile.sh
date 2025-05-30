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
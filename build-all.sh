#!/bin/bash

   APP_NAME="mnist_benchmark"
   OUTPUT_DIR="build"

   mkdir -p $OUTPUT_DIR

   platforms=(
     "linux/amd64"
#     "linux/arm64"
     "windows/amd64"
#     "windows/arm64"
     "darwin/amd64"
#     "darwin/arm64"
   )

   echo "🚀 Building $APP_NAME for multiple platforms..."

   for platform in "${platforms[@]}"
   do
     IFS="/" read -r GOOS GOARCH <<< "$platform"
     output_name="${APP_NAME}_${GOOS}_${GOARCH}"

     # Add .exe extension for Windows
     if [ "$GOOS" = "windows" ]; then
       output_name="${output_name}.exe"
     fi

     echo "🔧 $GOOS/$GOARCH → $output_name"

     # Set CGO environment for Windows builds
     if [ "$GOOS" = "windows" ]; then
       if [ "$GOARCH" = "amd64" ]; then
         env GOOS=$GOOS GOARCH=$GOARCH CGO_ENABLED=1 CC=x86_64-w64-mingw32-gcc CXX=x86_64-w64-mingw32-g++ go build -o "$OUTPUT_DIR/$output_name" .
       elif [ "$GOARCH" = "arm64" ]; then
         env GOOS=$GOOS GOARCH=$GOARCH CGO_ENABLED=1 CC=aarch64-w64-mingw32-gcc CXX=aarch64-w64-mingw32-g++ go build -o "$OUTPUT_DIR/$output_name" .
       fi
     else
       env GOOS=$GOOS GOARCH=$GOARCH go build -o "$OUTPUT_DIR/$output_name" .
     fi

     if [ $? -ne 0 ]; then
       echo "❌ Failed to build for $GOOS/$GOARCH"
       exit 1
     fi
   done
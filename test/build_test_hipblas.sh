#!/bin/bash

# Build script for hipBLAS transpose unit test

echo "Building hipBLAS transpose unit test..."

/opt/rocm/bin/hipcc \
    -o test_hipblas_transpose \
    test_hipblas_transpose.cpp \
    -I/opt/rocm/include \
    -I/opt/rocm/include/hipblas \
    -L/opt/rocm/lib \
    -lhipblas \
    -std=c++17

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo "Run with: ./test_hipblas_transpose"
else
    echo "❌ Build failed!"
    exit 1
fi

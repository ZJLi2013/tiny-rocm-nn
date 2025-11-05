#!/bin/bash

# Build script for rocWMMA basic test

echo "Building rocWMMA basic test..."

/opt/rocm/bin/hipcc \
    -o test_rocwmma_basic \
    test_rocwmma_basic.cpp \
    -I/opt/rocm/include \
    -I/opt/rocm/include/rocwmma \
    -L/opt/rocm/lib \
    -lamdhip64 \
    -std=c++17 \
    -O2

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo "Run with: ./test_rocwmma_basic"
else
    echo "✗ Build failed!"
    exit 1
fi

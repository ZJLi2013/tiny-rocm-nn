#!/bin/bash

# Build script for rocWMMA FP32×FP32 test (v31 validation)

echo "=== Building rocWMMA FP32×FP32 Test ==="

# Compiler settings
CXX=/opt/rocm/bin/hipcc
TARGET=test_rocwmma_fp32_input
SOURCE=test_rocwmma_fp32_input.cpp

# Compile
$CXX $SOURCE -o $TARGET \
    -std=c++17 \
    -O3 \
    -I/opt/rocm/include \
    -L/opt/rocm/lib \
    --offload-arch=gfx90a \
    -D__HIP_PLATFORM_AMD__

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "Running test..."
    ./$TARGET
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ FP32×FP32 → FP32 is supported!"
        echo "v31 can proceed with full FP32 pipeline implementation."
    else
        echo ""
        echo "❌ FP32×FP32 → FP32 test failed!"
        echo "v31 may not be feasible on this hardware."
    fi
else
    echo "❌ Build failed!"
    exit 1
fi

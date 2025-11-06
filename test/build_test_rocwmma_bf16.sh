#!/bin/bash

# Build script for rocWMMA BF16×BF16 → FP32 test (v32 validation)

echo "=== Building rocWMMA BF16×BF16 → FP32 Test ==="

# Compiler settings
CXX=/opt/rocm/bin/hipcc
TARGET=test_rocwmma_bf16
SOURCE=test_rocwmma_bf16.cpp

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
        echo "========================================="
        echo "✅ BF16×BF16 → FP32 is supported!"
        echo "========================================="
        echo ""
        echo "v32 can proceed with BF16 implementation."
        echo ""
        echo "Next steps:"
        echo "1. Modify fragment types: __half → hip_bfloat16"
        echo "2. Modify shared memory: __half* → hip_bfloat16*"
        echo "3. Update conversion functions: __float2half → __float2bfloat16"
        echo "4. Test training stability"
    else
        echo ""
        echo "❌ BF16×BF16 → FP32 test failed!"
        echo "v32 may not be feasible. Consider alternative solutions."
    fi
else
    echo "❌ Build failed!"
    exit 1
fi

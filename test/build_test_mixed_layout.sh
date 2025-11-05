#!/bin/bash

# Build script for mixed layout exact replication test

echo "Building test_mixed_layout_exact..."

/opt/rocm/bin/hipcc \
    -o test_mixed_layout_exact \
    test_mixed_layout_exact.cpp \
    -I/opt/rocm/include \
    -L/opt/rocm/lib \
    -lhipblas \
    -std=c++17

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo ""
    echo "Running test..."
    echo "============================================"
    ./test_mixed_layout_exact
    echo "============================================"
    echo ""
    echo "Test complete."
else
    echo "✗ Build failed!"
    exit 1
fi

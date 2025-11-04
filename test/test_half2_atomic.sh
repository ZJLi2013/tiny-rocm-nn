#!/bin/bash

# Test script to verify if atomicAdd for __half2 is supported in ROCm 7.1
# Run this on your AMD GPU node

echo "=========================================="
echo "Testing __half2 atomicAdd Support"
echo "=========================================="
echo ""

# Check ROCm version
echo "ROCm Version:"
hipcc --version
echo ""

# Build the test
echo "Building test_half2_atomic..."
hipcc test_half2_atomic.cpp -o test_half2_atomic_bin 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Compilation SUCCESSFUL!"
    echo "This means atomicAdd(__half2*, __half2) IS supported in your ROCm version."
    echo ""
    
    # Run the test
    echo "Running test..."
    ./test_half2_atomic_bin
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Test execution SUCCESSFUL!"
        echo ""
        echo "CONCLUSION: Your ROCm 7.1 has native __half2 atomicAdd support."
        echo "You can use atomicAdd directly without custom implementation."
    else
        echo ""
        echo "✗ Test execution FAILED (but compilation succeeded)"
        echo "This might indicate a runtime issue, not a missing API."
    fi
else
    echo ""
    echo "✗ Compilation FAILED!"
    echo "This means atomicAdd(__half2*, __half2) is NOT supported."
    echo ""
    echo "CONCLUSION: You need a custom atomicAdd implementation for __half2."
    echo "Use atomicCAS-based workaround in vec.h"
fi

echo ""
echo "=========================================="

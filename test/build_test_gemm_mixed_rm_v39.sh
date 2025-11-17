#!/bin/bash

# Build script for test_gemm_mixed_rm_v39.cpp
# Tests the MIXED_RM GEMM path that shows anomalies in v39.log

set -e

echo "=== Building test_gemm_mixed_rm_v39 ==="

# Compiler and flags
CXX=hipcc
CXXFLAGS="-std=c++14 -O2 -g"
INCLUDES="-I/opt/rocm/include"
LIBS="-L/opt/rocm/lib -lhipblas -lamdhip64"

# Source and output
SRC="test_gemm_mixed_rm_v39.cpp"
OUT="test_gemm_mixed_rm_v39"

echo "Compiling $SRC..."
$CXX $CXXFLAGS $INCLUDES $SRC $LIBS -o $OUT

if [ $? -eq 0 ]; then
    echo "✓ Build successful: $OUT"
    echo ""
    echo "Run with: ./$OUT"
else
    echo "✗ Build failed"
    exit 1
fi

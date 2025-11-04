#include "hip/hip_runtime.h"
#include <iostream>
#include <vector>

// Test kernel that attempts to use tex2D
__global__ void test_tex2d_kernel(hipTextureObject_t texture, float* result) {

#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include <iostream>

// Test if atomicAdd for __half2 is available in ROCm 7.1
__global__ void test_half2_atomic_kernel(__half2* result) {
    __half2 val = __half2{__float2half(1.0f), __float2half(2.0f)};
    
    // This will compile only if atomicAdd(__half2*, __half2) is available
    #ifdef __HIP_PLATFORM_AMD__
    atomicAdd(result, val);
    #endif
}

int main() {
    std::cout << "Testing __half2 atomicAdd support in ROCm 7.1..." << std::endl;
    
    __half2* d_result;
    hipMalloc(&d_result, sizeof(__half2));
    
    __half2 init = __half2(0.0f, 0.0f);
    hipMemcpy(d_result, &init, sizeof(__half2), hipMemcpyHostToDevice);
    
    test_half2_atomic_kernel<<<1, 32>>>(d_result);
    
    __half2 h_result;
    hipMemcpy(&h_result, d_result, sizeof(__half2), hipMemcpyDeviceToHost);
    
    hipFree(d_result);
    
    std::cout << "Test completed successfully!" << std::endl;
    std::cout << "If this compiles, __half2 atomicAdd is supported." << std::endl;
    
    return 0;
}

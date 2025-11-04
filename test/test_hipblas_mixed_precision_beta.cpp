/*
 * Unit test to demonstrate hipBLAS mixed precision beta accumulation bug
 * 
 * This test shows that hipblasGemmEx with:
 * - HIPBLAS_R_16F (FP16 data)
 * - HIPBLAS_COMPUTE_32F (FP32 compute)
 * - beta != 0 (accumulation mode)
 * 
 * Produces NaN outputs even with valid inputs.
 */

#include <hip/hip_runtime.h>
#include <hipblas.h>
#include "hip/hip_fp16.h"
#include <iostream>
#include <vector>
#include <cmath>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define HIPBLAS_CHECK(call) \
    do { \
        hipblasStatus_t status = call; \
        if (status != HIPBLAS_STATUS_SUCCESS) { \
            std::cerr << "hipBLAS error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << status << std::endl; \
            exit(1); \
        } \
    } while(0)

void print_matrix(const char* name, const std::vector<__half>& mat, int m, int n) {
    std::cout << name << " [" << m << "x" << n << "]:" << std::endl;
    for (int i = 0; i < std::min(5, m); ++i) {
        std::cout << "  ";
        for (int j = 0; j < std::min(5, n); ++j) {
            float val = __half2float(mat[i * n + j]);
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "=== hipBLAS Mixed Precision Beta Accumulation Bug Test ===" << std::endl;
    std::cout << std::endl;

    // Test case from v12.log GEMM #194
    const int m = 64;
    const int n = 64;
    const int k = 4096;

    // Initialize hipBLAS
    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));

    // Allocate host memory
    std::vector<__half> h_A(m * k);
    std::vector<__half> h_B(k * n);
    std::vector<__half> h_C(m * n);

    // Initialize A with small values (mostly zeros, like in v12.log GEMM #194)
    for (int i = 0; i < m * k; ++i) {
        h_A[i] = __float2half(0.0001f * (i % 10 - 5));  // Small values around 0
    }

    // Initialize B with reasonable values
    for (int i = 0; i < k * n; ++i) {
        h_B[i] = __float2half(0.1f * (i % 10));  // Values 0.0 to 0.9
    }

    // Initialize C with values similar to v12.log GEMM #194 C_before
    for (int i = 0; i < m * n; ++i) {
        h_C[i] = __float2half(1.5f + 0.1f * (i % 20));  // Values 1.5 to 3.4
    }

    std::cout << "Input matrices initialized:" << std::endl;
    print_matrix("A (first 5x5)", h_A, m, k);
    print_matrix("B (first 5x5)", h_B, k, n);
    print_matrix("C_before (first 5x5)", h_C, m, n);
    std::cout << std::endl;

    // Allocate device memory
    __half *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, m * k * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_B, k * n * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_C, m * n * sizeof(__half)));

    // Copy to device
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), m * k * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), k * n * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C.data(), m * n * sizeof(__half), hipMemcpyHostToDevice));

    // Test parameters
    float alpha = 1.0f;
    float beta = 1.0f;  // Accumulation mode (this triggers the bug!)

    std::cout << "=== Test 1: hipblasGemmEx with FP16 data + FP32 compute + beta=1.0 ===" << std::endl;
    std::cout << "Parameters: alpha=" << alpha << ", beta=" << beta << std::endl;
    std::cout << "Data type: HIPBLAS_R_16F, Compute type: HIPBLAS_COMPUTE_32F" << std::endl;
    std::cout << std::endl;

    // Call hipblasGemmEx with mixed precision
    HIPBLAS_CHECK(hipblasGemmEx(
        handle,
        HIPBLAS_OP_N, HIPBLAS_OP_N,
        m, n, k,
        &alpha,
        d_A, HIPBLAS_R_16F, m,
        d_B, HIPBLAS_R_16F, k,
        &beta,
        d_C, HIPBLAS_R_16F, m,
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT
    ));

    // Copy result back
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, m * n * sizeof(__half), hipMemcpyDeviceToHost));

    std::cout << "Result C_output (first 5x5):" << std::endl;
    print_matrix("C_output", h_C, m, n);
    
    // Check for NaN
    bool has_nan = false;
    for (int i = 0; i < m * n; ++i) {
        if (std::isnan(__half2float(h_C[i]))) {
            has_nan = true;
            break;
        }
    }
    
    if (has_nan) {
        std::cout << "❌ FAILED: NaN detected in output!" << std::endl;
        std::cout << "This confirms the hipBLAS mixed precision beta accumulation bug." << std::endl;
    } else {
        std::cout << "✓ PASSED: No NaN in output" << std::endl;
    }
    std::cout << std::endl;

    // Test 2: Same operation but with beta=0.0 (should work)
    std::cout << "=== Test 2: Same setup but with beta=0.0 ===" << std::endl;
    
    // Reset C
    for (int i = 0; i < m * n; ++i) {
        h_C[i] = __float2half(1.5f + 0.1f * (i % 20));
    }
    HIP_CHECK(hipMemcpy(d_C, h_C.data(), m * n * sizeof(__half), hipMemcpyHostToDevice));
    
    beta = 0.0f;
    std::cout << "Parameters: alpha=" << alpha << ", beta=" << beta << std::endl;
    std::cout << std::endl;

    HIPBLAS_CHECK(hipblasGemmEx(
        handle,
        HIPBLAS_OP_N, HIPBLAS_OP_N,
        m, n, k,
        &alpha,
        d_A, HIPBLAS_R_16F, m,
        d_B, HIPBLAS_R_16F, k,
        &beta,
        d_C, HIPBLAS_R_16F, m,
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT
    ));

    HIP_CHECK(hipMemcpy(h_C.data(), d_C, m * n * sizeof(__half), hipMemcpyDeviceToHost));

    std::cout << "Result C_output (first 5x5):" << std::endl;
    print_matrix("C_output", h_C, m, n);
    
    has_nan = false;
    for (int i = 0; i < m * n; ++i) {
        if (std::isnan(__half2float(h_C[i]))) {
            has_nan = true;
            break;
        }
    }
    
    if (has_nan) {
        std::cout << "❌ FAILED: NaN detected even with beta=0!" << std::endl;
    } else {
        std::cout << "✓ PASSED: No NaN with beta=0.0" << std::endl;
    }
    std::cout << std::endl;

    // Test 3: Use hipblasGemmEx with FP16 compute (HIPBLAS_COMPUTE_16F) + beta=1.0
    std::cout << "=== Test 3: hipblasGemmEx with FP16 data + FP16 compute + beta=1.0 ===" << std::endl;
    
    // Reset C
    for (int i = 0; i < m * n; ++i) {
        h_C[i] = __float2half(1.5f + 0.1f * (i % 20));
    }
    HIP_CHECK(hipMemcpy(d_C, h_C.data(), m * n * sizeof(__half), hipMemcpyHostToDevice));
    
    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(1.0f);
    
    std::cout << "Using hipblasGemmEx with FP16 compute (matching data type)" << std::endl;
    std::cout << "Parameters: alpha_h=1.0, beta_h=1.0" << std::endl;
    std::cout << "Data type: HIPBLAS_R_16F, Compute type: HIPBLAS_COMPUTE_16F" << std::endl;
    std::cout << std::endl;

    HIPBLAS_CHECK(hipblasGemmEx(
        handle,
        HIPBLAS_OP_N, HIPBLAS_OP_N,
        m, n, k,
        &alpha_h,
        d_A, HIPBLAS_R_16F, m,
        d_B, HIPBLAS_R_16F, k,
        &beta_h,
        d_C, HIPBLAS_R_16F, m,
        HIPBLAS_COMPUTE_16F,  // Match compute type with data type
        HIPBLAS_GEMM_DEFAULT
    ));

    HIP_CHECK(hipMemcpy(h_C.data(), d_C, m * n * sizeof(__half), hipMemcpyDeviceToHost));

    std::cout << "Result C_output (first 5x5):" << std::endl;
    print_matrix("C_output", h_C, m, n);
    
    has_nan = false;
    for (int i = 0; i < m * n; ++i) {
        if (std::isnan(__half2float(h_C[i]))) {
            has_nan = true;
            break;
        }
    }
    
    if (has_nan) {
        std::cout << "❌ FAILED: NaN detected with FP16 compute!" << std::endl;
        std::cout << "FP16 compute may cause overflow in accumulation." << std::endl;
    } else {
        std::cout << "✓ PASSED: FP16 compute works with beta=1.0" << std::endl;
        std::cout << "But may still overflow with many accumulations (see v11.log)" << std::endl;
    }
    std::cout << std::endl;

    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIPBLAS_CHECK(hipblasDestroy(handle));

    std::cout << "=== Test Complete ===" << std::endl;
    std::cout << "Conclusion: Use hipblasHgemm for FP16 to avoid mixed precision bugs" << std::endl;

    return 0;
}

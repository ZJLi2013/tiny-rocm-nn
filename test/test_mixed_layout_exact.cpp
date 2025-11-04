/*
 * Unit test to replicate EXACT scenario from v12.log GEMM #194
 * 
 * This test replicates the MIXED LAYOUT: CM×RM→RM case that produces NaN
 * in the actual training code.
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
            std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define HIPBLAS_CHECK(call) \
    do { \
        hipblasStatus_t status = call; \
        if (status != HIPBLAS_STATUS_SUCCESS) { \
            std::cerr << "hipBLAS error: " << status << std::endl; \
            exit(1); \
        } \
    } while(0)

void print_samples(const char* name, const std::vector<__half>& mat, int count) {
    std::cout << "  " << name << " samples: ";
    for (int i = 0; i < std::min(count, 5); ++i) {
        std::cout << __half2float(mat[i]) << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== Exact Replication of v12.log GEMM #194 ===" << std::endl;
    std::cout << std::endl;

    // Exact dimensions from v12.log GEMM #194
    const int m = 64;   // A rows, C rows
    const int k = 4096; // A cols, B rows
    const int n = 64;   // B cols, C cols

    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));

    // Allocate matrices
    // A is CM (64×4096), stored column-by-column
    std::vector<__half> h_A_cm(m * k);
    // B is RM (4096×64), stored row-by-row
    std::vector<__half> h_B_rm(k * n);
    // C is RM (64×64), stored row-by-row
    std::vector<__half> h_C_rm(m * n);

    // Initialize A (CM layout) with small values
    for (int col = 0; col < k; ++col) {
        for (int row = 0; row < m; ++row) {
            h_A_cm[col * m + row] = __float2half(0.0001f * ((row + col) % 10 - 5));
        }
    }

    // Initialize B (RM layout) with reasonable values
    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            h_B_rm[row * n + col] = __float2half(0.1f * ((row + col) % 10));
        }
    }

    // Initialize C (RM layout) with values like v12.log
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            h_C_rm[row * n + col] = __float2half(1.5f + 0.1f * ((row + col) % 20));
        }
    }

    std::cout << "Matrix layouts:" << std::endl;
    std::cout << "  A: CM (64×4096), stride=64" << std::endl;
    std::cout << "  B: RM (4096×64), stride=64" << std::endl;
    std::cout << "  C: RM (64×64), stride=64" << std::endl;
    std::cout << std::endl;

    print_samples("A_cm", h_A_cm, m * k);
    print_samples("B_rm", h_B_rm, k * n);
    print_samples("C_rm (before)", h_C_rm, m * n);
    std::cout << std::endl;

    // Allocate device memory
    __half *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, m * k * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_B, k * n * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_C, m * n * sizeof(__half)));

    HIP_CHECK(hipMemcpy(d_A, h_A_cm.data(), m * k * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B_rm.data(), k * n * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C_rm.data(), m * n * sizeof(__half), hipMemcpyHostToDevice));

    // Replicate EXACT call from training code for CM×RM→RM
    std::cout << "=== Replicating MIXED LAYOUT call: CM×RM→RM ===" << std::endl;
    std::cout << "hipblasGemmEx(handle, op_b=N, op_a=T, n=64, m=64, k=4096," << std::endl;
    std::cout << "              alpha=1.0, B, ldb=64, A, lda=64," << std::endl;
    std::cout << "              beta=1.0, C, ldc=64," << std::endl;
    std::cout << "              HIPBLAS_COMPUTE_32F)" << std::endl;
    std::cout << std::endl;

    float alpha = 1.0f;
    float beta = 1.0f;

    // This is the EXACT call from the training code
    HIPBLAS_CHECK(hipblasGemmEx(
        handle,
        HIPBLAS_OP_N,        // op_b (for B)
        HIPBLAS_OP_T,        // op_a (for A)
        n, m, k,             // Swapped: n=64, m=64, k=4096
        &alpha,
        d_B, HIPBLAS_R_16F, n,  // B first, ldb = n = 64 (B.stride() for RM)
        d_A, HIPBLAS_R_16F, m,  // A second, lda = m = 64 (A.stride() for CM)
        &beta,
        d_C, HIPBLAS_R_16F, n,  // C, ldc = n = 64 (C.stride() for RM)
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT
    ));

    HIP_CHECK(hipMemcpy(h_C_rm.data(), d_C, m * n * sizeof(__half), hipMemcpyDeviceToHost));

    std::cout << "Result:" << std::endl;
    print_samples("C_rm (after)", h_C_rm, m * n);
    std::cout << std::endl;

    // Check for NaN
    bool has_nan = false;
    for (int i = 0; i < m * n; ++i) {
        if (std::isnan(__half2float(h_C_rm[i]))) {
            has_nan = true;
            break;
        }
    }

    if (has_nan) {
        std::cout << "❌ REPRODUCED THE BUG: NaN detected!" << std::endl;
        std::cout << "The issue is in the MIXED LAYOUT leading dimension calculation!" << std::endl;
    } else {
        std::cout << "✓ NO BUG: Output is valid" << std::endl;
        std::cout << "The issue must be in the actual training data, not the hipBLAS call!" << std::endl;
    }
    std::cout << std::endl;

    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIPBLAS_CHECK(hipblasDestroy(handle));

    return 0;
}

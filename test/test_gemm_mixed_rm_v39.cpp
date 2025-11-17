/*
 * Unit test for v39 MIXED_RM GEMM anomaly
 * 
 * This test replicates the EXACT scenario from v39.log where:
 * - Step≈3500: C_stats_MIXED_RM[16x64] Inf=1
 * - Dimensions: m=16, n=64, k=4096
 * - Input A/B nearly zero, but output C has anomalous values (3.8574, 16.3125, Inf)
 * - Compute type: FP32 (HIPBLAS_COMPUTE_32F)
 * 
 * Goal: Verify if the MIXED_RM path (LC==RM) with op_b=N, op_a=T mapping is correct
 */

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include "hip/hip_fp16.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

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

// Helper to print matrix stats
void print_matrix_stats(const char* name, const std::vector<__half>& mat, int m, int n) {
    size_t nan_count = 0, inf_count = 0;
    float max_abs = 0.0f, min_val = 1e10f, max_val = -1e10f;
    
    for (size_t i = 0; i < mat.size(); ++i) {
        float v = __half2float(mat[i]);
        if (std::isnan(v)) {
            ++nan_count;
        } else if (std::isinf(v)) {
            ++inf_count;
        } else {
            max_abs = std::max(max_abs, std::fabs(v));
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
        }
    }
    
    std::cout << "  " << name << "[" << m << "x" << n << "] stats: "
              << "NaN=" << nan_count << " Inf=" << inf_count 
              << " range=[" << std::fixed << std::setprecision(4) << min_val << ", " << max_val << "]"
              << " max_abs=" << max_abs << std::endl;
    
    // Print first 8 values
    std::cout << "    First 8 values: ";
    for (int i = 0; i < std::min(8, (int)mat.size()); ++i) {
        std::cout << std::fixed << std::setprecision(4) << __half2float(mat[i]) << " ";
    }
    std::cout << std::endl;
}

// CPU reference implementation for C = A * B (all FP32)
void cpu_gemm_fp32(
    const std::vector<float>& A, int m, int k, bool A_is_RM,
    const std::vector<float>& B, int k2, int n, bool B_is_RM,
    std::vector<float>& C, float alpha, float beta
) {
    if (k != k2) {
        std::cerr << "CPU GEMM: k mismatch" << std::endl;
        exit(1);
    }
    
    // C = alpha * A * B + beta * C
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                // A[i,p]
                int a_idx = A_is_RM ? (i * k + p) : (p * m + i);
                // B[p,j]
                int b_idx = B_is_RM ? (p * n + j) : (j * k + p);
                sum += A[a_idx] * B[b_idx];
            }
            int c_idx = i * n + j; // C is always RM in our reference
            C[c_idx] = alpha * sum + beta * C[c_idx];
        }
    }
}

// Convert half to float vector
std::vector<float> half_to_float(const std::vector<__half>& h) {
    std::vector<float> f(h.size());
    for (size_t i = 0; i < h.size(); ++i) {
        f[i] = __half2float(h[i]);
    }
    return f;
}

// Convert float to half vector
std::vector<__half> float_to_half(const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); ++i) {
        h[i] = __float2half(f[i]);
    }
    return h;
}

// Compare GPU result with CPU reference
void compare_results(const std::vector<__half>& gpu, const std::vector<float>& cpu, 
                     int m, int n, const char* test_name) {
    float max_diff = 0.0f;
    size_t diff_count = 0;
    const float tolerance = 1e-2f; // Relaxed tolerance for FP16
    
    for (int i = 0; i < m * n; ++i) {
        float g = __half2float(gpu[i]);
        float c = cpu[i];
        float diff = std::fabs(g - c);
        
        if (diff > tolerance) {
            ++diff_count;
            max_diff = std::max(max_diff, diff);
        }
    }
    
    std::cout << "  " << test_name << " comparison: max_diff=" << max_diff 
              << " diff_count=" << diff_count << "/" << (m*n);
    
    if (diff_count == 0) {
        std::cout << " ✓ PASS" << std::endl;
    } else {
        std::cout << " ✗ FAIL" << std::endl;
    }
}

int main() {
    std::cout << "=== v39 MIXED_RM GEMM Unit Test ===" << std::endl;
    std::cout << std::endl;

    // Exact dimensions from v39.log
    const int m = 16;
    const int k = 4096;
    const int n = 64;
    
    std::cout << "Test configuration:" << std::endl;
    std::cout << "  Dimensions: m=" << m << " k=" << k << " n=" << n << std::endl;
    std::cout << "  Compute type: FP32 (HIPBLAS_COMPUTE_32F)" << std::endl;
    std::cout << "  Data type: FP16 (HIPBLAS_R_16F)" << std::endl;
    std::cout << std::endl;

    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));

    // Random number generator
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f); // Small values like v39.log

    // Test Case 1: MIXED_RM path (A=CM, B=RM, C=RM) - the problematic case
    std::cout << "=== Test Case 1: MIXED_RM (A=CM, B=RM, C=RM) ===" << std::endl;
    {
        // A is CM (16×4096), stored column-by-column
        std::vector<__half> h_A_cm(m * k);
        for (int col = 0; col < k; ++col) {
            for (int row = 0; row < m; ++row) {
                h_A_cm[col * m + row] = __float2half(dist(rng));
            }
        }
        
        // B is RM (4096×64), stored row-by-row
        std::vector<__half> h_B_rm(k * n);
        for (int i = 0; i < k * n; ++i) {
            h_B_rm[i] = __float2half(dist(rng));
        }
        
        // C is RM (16×64), stored row-by-row
        std::vector<__half> h_C_rm(m * n);
        for (int i = 0; i < m * n; ++i) {
            h_C_rm[i] = __float2half(0.002f); // Small initial value
        }

        print_matrix_stats("A_cm", h_A_cm, m, k);
        print_matrix_stats("B_rm", h_B_rm, k, n);
        print_matrix_stats("C_rm (before)", h_C_rm, m, n);

        // Allocate device memory
        __half *d_A, *d_B, *d_C;
        HIP_CHECK(hipMalloc(&d_A, m * k * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_B, k * n * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_C, m * n * sizeof(__half)));

        HIP_CHECK(hipMemcpy(d_A, h_A_cm.data(), m * k * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B_rm.data(), k * n * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_C, h_C_rm.data(), m * n * sizeof(__half), hipMemcpyHostToDevice));

        float alpha = 1.0f;
        float beta = 1.0f;

        std::cout << "\n  Calling hipblasGemmEx with MIXED_RM mapping:" << std::endl;
        std::cout << "    op_b=N, op_a=T (for LC==RM branch)" << std::endl;
        std::cout << "    hipblasGemmEx(handle, N, T, n=" << n << ", m=" << m << ", k=" << k << "," << std::endl;
        std::cout << "                  alpha=1.0, B, ldb=" << n << ", A, lda=" << m << "," << std::endl;
        std::cout << "                  beta=1.0, C, ldc=" << n << ", FP32_COMPUTE)" << std::endl;

        // This is the EXACT call from cublas_matmul.h MIXED_RM branch
        HIPBLAS_CHECK(hipblasGemmEx(
            handle,
            HIPBLAS_OP_N,        // op_b (for B, which is RM)
            HIPBLAS_OP_T,        // op_a (for A, which is CM)
            n, m, k,             // Swapped dimensions
            &alpha,
            d_B, HIPBLAS_R_16F, n,  // B first, ldb = B.stride() = n (for RM)
            d_A, HIPBLAS_R_16F, m,  // A second, lda = A.stride() = m (for CM)
            &beta,
            d_C, HIPBLAS_R_16F, n,  // C, ldc = C.stride() = n (for RM)
            HIPBLAS_COMPUTE_32F,
            HIPBLAS_GEMM_DEFAULT
        ));

        HIP_CHECK(hipMemcpy(h_C_rm.data(), d_C, m * n * sizeof(__half), hipMemcpyDeviceToHost));

        std::cout << "\n  GPU Result:" << std::endl;
        print_matrix_stats("C_rm (after GPU)", h_C_rm, m, n);

        // CPU reference
        std::vector<float> A_fp32 = half_to_float(h_A_cm);
        std::vector<float> B_fp32 = half_to_float(h_B_rm);
        std::vector<float> C_fp32 = half_to_float(h_C_rm);
        
        // Reset C for CPU computation
        for (int i = 0; i < m * n; ++i) {
            C_fp32[i] = 0.002f;
        }
        
        cpu_gemm_fp32(A_fp32, m, k, false, B_fp32, k, n, true, C_fp32, alpha, beta);
        
        std::cout << "\n  CPU Reference:" << std::endl;
        std::cout << "    First 8 values: ";
        for (int i = 0; i < 8; ++i) {
            std::cout << std::fixed << std::setprecision(4) << C_fp32[i] << " ";
        }
        std::cout << std::endl;

        compare_results(h_C_rm, C_fp32, m, n, "MIXED_RM");

        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
    }

    std::cout << "\n=== Test Case 2: Pure RM (A=RM, B=RM, C=RM) - baseline ===" << std::endl;
    {
        // A is RM (16×4096)
        std::vector<__half> h_A_rm(m * k);
        for (int i = 0; i < m * k; ++i) {
            h_A_rm[i] = __float2half(dist(rng));
        }
        
        // B is RM (4096×64)
        std::vector<__half> h_B_rm(k * n);
        for (int i = 0; i < k * n; ++i) {
            h_B_rm[i] = __float2half(dist(rng));
        }
        
        // C is RM (16×64)
        std::vector<__half> h_C_rm(m * n);
        for (int i = 0; i < m * n; ++i) {
            h_C_rm[i] = __float2half(0.002f);
        }

        print_matrix_stats("A_rm", h_A_rm, m, k);
        print_matrix_stats("B_rm", h_B_rm, k, n);

        __half *d_A, *d_B, *d_C;
        HIP_CHECK(hipMalloc(&d_A, m * k * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_B, k * n * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_C, m * n * sizeof(__half)));

        HIP_CHECK(hipMemcpy(d_A, h_A_rm.data(), m * k * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B_rm.data(), k * n * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_C, h_C_rm.data(), m * n * sizeof(__half), hipMemcpyHostToDevice));

        float alpha = 1.0f;
        float beta = 1.0f;

        std::cout << "\n  Calling hipblasGemmEx with pure RM mapping:" << std::endl;
        std::cout << "    op_b=N, op_a=N (standard RM×RM→RM)" << std::endl;

        // Standard RM×RM→RM call
        HIPBLAS_CHECK(hipblasGemmEx(
            handle,
            HIPBLAS_OP_N, HIPBLAS_OP_N,
            n, m, k,
            &alpha,
            d_B, HIPBLAS_R_16F, n,
            d_A, HIPBLAS_R_16F, k,
            &beta,
            d_C, HIPBLAS_R_16F, n,
            HIPBLAS_COMPUTE_32F,
            HIPBLAS_GEMM_DEFAULT
        ));

        HIP_CHECK(hipMemcpy(h_C_rm.data(), d_C, m * n * sizeof(__half), hipMemcpyDeviceToHost));

        std::cout << "\n  GPU Result:" << std::endl;
        print_matrix_stats("C_rm (after GPU)", h_C_rm, m, n);

        // CPU reference
        std::vector<float> A_fp32 = half_to_float(h_A_rm);
        std::vector<float> B_fp32 = half_to_float(h_B_rm);
        std::vector<float> C_fp32 = half_to_float(h_C_rm);
        
        for (int i = 0; i < m * n; ++i) {
            C_fp32[i] = 0.002f;
        }
        
        cpu_gemm_fp32(A_fp32, m, k, true, B_fp32, k, n, true, C_fp32, alpha, beta);

        compare_results(h_C_rm, C_fp32, m, n, "Pure RM");

        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
    }

    std::cout << "\n=== Test Case 3: Pure CM (A=CM, B=CM, C=CM) - baseline ===" << std::endl;
    {
        // A is CM (16×4096)
        std::vector<__half> h_A_cm(m * k);
        for (int col = 0; col < k; ++col) {
            for (int row = 0; row < m; ++row) {
                h_A_cm[col * m + row] = __float2half(dist(rng));
            }
        }
        
        // B is CM (4096×64)
        std::vector<__half> h_B_cm(k * n);
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < k; ++row) {
                h_B_cm[col * k + row] = __float2half(dist(rng));
            }
        }
        
        // C is CM (16×64)
        std::vector<__half> h_C_cm(m * n);
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                h_C_cm[col * m + row] = __float2half(0.002f);
            }
        }

        print_matrix_stats("A_cm", h_A_cm, m, k);
        print_matrix_stats("B_cm", h_B_cm, k, n);

        __half *d_A, *d_B, *d_C;
        HIP_CHECK(hipMalloc(&d_A, m * k * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_B, k * n * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_C, m * n * sizeof(__half)));

        HIP_CHECK(hipMemcpy(d_A, h_A_cm.data(), m * k * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B_cm.data(), k * n * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_C, h_C_cm.data(), m * n * sizeof(__half), hipMemcpyHostToDevice));

        float alpha = 1.0f;
        float beta = 1.0f;

        std::cout << "\n  Calling hipblasGemmEx with pure CM mapping:" << std::endl;

        // Standard CM×CM→CM call
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

        HIP_CHECK(hipMemcpy(h_C_cm.data(), d_C, m * n * sizeof(__half), hipMemcpyDeviceToHost));

        std::cout << "\n  GPU Result:" << std::endl;
        print_matrix_stats("C_cm (after GPU)", h_C_cm, m, n);

        // CPU reference (convert CM to RM for easier comparison)
        std::vector<float> A_fp32 = half_to_float(h_A_cm);
        std::vector<float> B_fp32 = half_to_float(h_B_cm);
        std::vector<float> C_fp32(m * n, 0.002f);
        
        cpu_gemm_fp32(A_fp32, m, k, false, B_fp32, k, n, false, C_fp32, alpha, beta);

        // Convert GPU CM result to RM for comparison
        std::vector<__half> h_C_rm(m * n);
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                h_C_rm[row * n + col] = h_C_cm[col * m + row];
            }
        }

        compare_results(h_C_rm, C_fp32, m, n, "Pure CM");

        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
    }

    HIPBLAS_CHECK(hipblasDestroy(handle));

    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}

#include "hip/hip_runtime.h"
#include <hipblas.h>
#include <iostream>
#include <vector>
#include <cmath>
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

void print_matrix(const char* name, const std::vector<float>& mat, int rows, int cols, bool row_major = true) {
    std::cout << name << " [" << rows << "x" << cols << "]";
    std::cout << (row_major ? " (Row-Major)" : " (Column-Major)") << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = row_major ? (i * cols + j) : (j * rows + i);
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << mat[idx] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Test Case 1: Simple CM×RM→RM multiplication with transpose
void test_cm_rm_to_rm_transpose() {
    std::cout << "=== Test 1: CM×RM→RM with Transpose ===" << std::endl;
    
    // Setup: C_rm[4,4] = A_cm[4,4] × B_rm[4,4]
    const int M = 4, K = 4, N = 4;
    
    // A is Column-Major [4,4]
    std::vector<float> A_cm = {
        1, 2, 3, 4,     // column 0
        5, 6, 7, 8,     // column 1
        9, 10, 11, 12,  // column 2
        13, 14, 15, 16  // column 3
    };
    
    // B is Row-Major [4,4]
    std::vector<float> B_rm = {
        1, 0, 0, 0,  // row 0
        0, 1, 0, 0,  // row 1
        0, 0, 1, 0,  // row 2
        0, 0, 0, 1   // row 3
    };
    
    // C is Row-Major [4,4] - output
    std::vector<float> C_rm(M * N, 0.0f);
    
    print_matrix("A (CM)", A_cm, M, K, false);
    print_matrix("B (RM)", B_rm, K, N, true);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));
    
    HIP_CHECK(hipMemcpy(d_A, A_cm.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B_rm.data(), K * N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, C_rm.data(), M * N * sizeof(float), hipMemcpyHostToDevice));
    
    // Create hipBLAS handle
    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Method 1: Using our current approach (swapped with transpose)
    // We want: C_rm = A_cm × B_rm
    // We compute: C^T = B^T × A^T
    hipblasOperation_t op_a = HIPBLAS_OP_T;  // A is CM, need transpose
    hipblasOperation_t op_b = HIPBLAS_OP_N;  // B is RM, no transpose needed
    
    int lda = M;  // Leading dim of CM[M,K] = M
    int ldb = N;  // Leading dim of RM[K,N] = N
    int ldc = N;  // Leading dim of RM[M,N] = N
    
    std::cout << "Calling hipblasGemmEx with:" << std::endl;
    std::cout << "  op_b=" << (op_b == HIPBLAS_OP_N ? "N" : "T") 
              << ", op_a=" << (op_a == HIPBLAS_OP_N ? "N" : "T") << std::endl;
    std::cout << "  Dimensions: n=" << N << ", m=" << M << ", k=" << K << std::endl;
    std::cout << "  Leading dims: lda=" << lda << ", ldb=" << ldb << ", ldc=" << ldc << std::endl;
    
    HIPBLAS_CHECK(hipblasGemmEx(
        handle,
        op_b, op_a,
        N, M, K,
        &alpha,
        d_B, HIPBLAS_R_32F, ldb,
        d_A, HIPBLAS_R_32F, lda,
        &beta,
        d_C, HIPBLAS_R_32F, ldc,
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT
    ));
    
    HIP_CHECK(hipMemcpy(C_rm.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
    
    print_matrix("C (RM) - Result", C_rm, M, N, true);
    
    // Expected result: A × I = A (but in row-major layout)
    std::cout << "Expected: A matrix in row-major layout" << std::endl;
    std::cout << "  Row 0: 1, 5, 9, 13" << std::endl;
    std::cout << "  Row 1: 2, 6, 10, 14" << std::endl;
    std::cout << "  Row 2: 3, 7, 11, 15" << std::endl;
    std::cout << "  Row 3: 4, 8, 12, 16" << std::endl;
    
    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIPBLAS_CHECK(hipblasDestroy(handle));
    
    std::cout << "\n";
}

// Test Case 2: Test with beta=1.0 accumulation
void test_accumulation_with_transpose() {
    std::cout << "=== Test 2: CM×RM→RM with beta=1.0 Accumulation ===" << std::endl;
    
    const int M = 4, K = 4, N = 4;
    
    // A is Column-Major [4,4] - very small values (like GEMM #194)
    std::vector<float> A_cm(M * K, 0.0001f);  // Almost zero
    
    // B is Row-Major [4,4]
    std::vector<float> B_rm = {
        0.1f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.1f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.1f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.1f
    };
    
    // C is Row-Major [4,4] - pre-filled with values (like C_before)
    std::vector<float> C_rm = {
        1.5f, 2.0f, 2.5f, 3.0f,
        1.0f, 1.5f, 2.0f, 2.5f,
        0.5f, 1.0f, 1.5f, 2.0f,
        0.0f, 0.5f, 1.0f, 1.5f
    };
    
    print_matrix("A (CM) - almost zero", A_cm, M, K, false);
    print_matrix("B (RM)", B_rm, K, N, true);
    print_matrix("C_before (RM)", C_rm, M, N, true);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));
    
    HIP_CHECK(hipMemcpy(d_A, A_cm.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B_rm.data(), K * N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, C_rm.data(), M * N * sizeof(float), hipMemcpyHostToDevice));
    
    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));
    
    float alpha = 1.0f, beta = 1.0f;  // Accumulation mode
    
    hipblasOperation_t op_a = HIPBLAS_OP_T;
    hipblasOperation_t op_b = HIPBLAS_OP_N;
    
    int lda = M;
    int ldb = N;
    int ldc = N;
    
    std::cout << "Calling hipblasGemmEx with beta=1.0 (accumulation):" << std::endl;
    std::cout << "  op_b=N, op_a=T" << std::endl;
    std::cout << "  Leading dims: lda=" << lda << ", ldb=" << ldb << ", ldc=" << ldc << std::endl;
    
    HIPBLAS_CHECK(hipblasGemmEx(
        handle,
        op_b, op_a,
        N, M, K,
        &alpha,
        d_B, HIPBLAS_R_32F, ldb,
        d_A, HIPBLAS_R_32F, lda,
        &beta,
        d_C, HIPBLAS_R_32F, ldc,
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT
    ));
    
    HIP_CHECK(hipMemcpy(C_rm.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
    
    print_matrix("C_after (RM) - Result", C_rm, M, N, true);
    
    std::cout << "Expected: C_after ≈ C_before (since A≈0)" << std::endl;
    std::cout << "  Should be: 1.5, 2.0, 2.5, 3.0 (first row)" << std::endl;
    
    // Check for NaN
    bool has_nan = false;
    for (float val : C_rm) {
        if (std::isnan(val)) {
            has_nan = true;
            break;
        }
    }
    
    if (has_nan) {
        std::cout << "❌ FAILED: NaN detected in output!" << std::endl;
    } else {
        std::cout << "✓ PASSED: No NaN in output" << std::endl;
    }
    
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIPBLAS_CHECK(hipblasDestroy(handle));
    
    std::cout << "\n";
}

// Test Case 3: Test different leading dimension values
void test_leading_dim_variations() {
    std::cout << "=== Test 3: Leading Dimension Variations ===" << std::endl;
    
    const int M = 4, K = 8, N = 4;  // Non-square to test stride effects
    
    // A is Column-Major [4,8]
    std::vector<float> A_cm(M * K);
    for (int i = 0; i < M * K; ++i) A_cm[i] = (i % 10) * 0.1f;
    
    // B is Row-Major [8,4]  
    std::vector<float> B_rm(K * N);
    for (int i = 0; i < K * N; ++i) B_rm[i] = (i % 10) * 0.1f;
    
    // C is Row-Major [4,4]
    std::vector<float> C_rm(M * N, 0.0f);
    
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));
    
    HIP_CHECK(hipMemcpy(d_A, A_cm.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B_rm.data(), K * N * sizeof(float), hipMemcpyHostToDevice));
    
    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Test different lda values
    std::vector<int> lda_values = {M, K};  // Try both possibilities
    
    for (int lda : lda_values) {
        std::cout << "\nTesting with lda=" << lda << ":" << std::endl;
        
        // Reset C
        HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));
        
        HIPBLAS_CHECK(hipblasGemmEx(
            handle,
            HIPBLAS_OP_N, HIPBLAS_OP_T,
            N, M, K,
            &alpha,
            d_B, HIPBLAS_R_32F, N,   // ldb = N for RM[K,N]
            d_A, HIPBLAS_R_32F, lda, // Testing different lda
            &beta,
            d_C, HIPBLAS_R_32F, N,   // ldc = N for RM[M,N]
            HIPBLAS_COMPUTE_32F,
            HIPBLAS_GEMM_DEFAULT
        ));
        
        HIP_CHECK(hipMemcpy(C_rm.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
        
        // Check for NaN
        bool has_nan = false;
        for (float val : C_rm) {
            if (std::isnan(val)) {
                has_nan = true;
                break;
            }
        }
        
        std::cout << "  Result: ";
        if (has_nan) {
            std::cout << "❌ NaN detected!" << std::endl;
        } else {
            std::cout << "✓ Valid output - First 4 values: ";
            for (int i = 0; i < 4; ++i) {
                std::cout << C_rm[i] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIPBLAS_CHECK(hipblasDestroy(handle));
    
    std::cout << "\n";
}

// Test Case 4: Reproduce exact GEMM #194 scenario
void test_gemm_194_scenario() {
    std::cout << "=== Test 4: Reproduce GEMM #194 Scenario ===" << std::endl;
    
    const int M = 64, K = 4096, N = 64;
    
    // A is Column-Major [64,4096] - essentially zero (like GEMM #194)
    std::vector<float> A_cm(M * K, 0.0f);
    A_cm[0] = 0.0001f;  // Tiny non-zero value
    
    // B is Row-Major [4096,64]
    std::vector<float> B_rm(K * N, 0.0f);
    B_rm[0] = 0.1514f;
    B_rm[N * 2] = 0.0714f;
    
    // C is Row-Major [64,64] - pre-filled (like C_before)
    std::vector<float> C_rm(M * N);
    C_rm[0] = 1.6377f;
    C_rm[1] = 2.1035f;
    C_rm[2] = 1.9834f;
    C_rm[3] = 1.0703f;
    C_rm[4] = 2.9375f;
    
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));
    
    HIP_CHECK(hipMemcpy(d_A, A_cm.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B_rm.data(), K * N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, C_rm.data(), M * N * sizeof(float), hipMemcpyHostToDevice));
    
    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));
    
    float alpha = 1.0f, beta = 1.0f;
    
    hipblasOperation_t op_a = HIPBLAS_OP_T;
    hipblasOperation_t op_b = HIPBLAS_OP_N;
    
    int lda = M;  // Leading dim of CM[M,K]
    int ldb = N;  // Leading dim of RM[K,N]
    int ldc = N;  // Leading dim of RM[M,N]
    
    std::cout << "GEMM #194 reproduction:" << std::endl;
    std::cout << "  A≈0, B has small values, C_before has valid values" << std::endl;
    std::cout << "  op_b=N, op_a=T, beta=1.0" << std::endl;
    std::cout << "  lda=" << lda << ", ldb=" << ldb << ", ldc=" << ldc << std::endl;
    
    HIPBLAS_CHECK(hipblasGemmEx(
        handle,
        op_b, op_a,
        N, M, K,
        &alpha,
        d_B, HIPBLAS_R_32F, ldb,
        d_A, HIPBLAS_R_32F, lda,
        &beta,
        d_C, HIPBLAS_R_32F, ldc,
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT
    ));
    
    HIP_CHECK(hipMemcpy(C_rm.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
    
    std::cout << "Result - First 5 values: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << C_rm[i] << " ";
    }
    std::cout << std::endl;
    
    // Check for NaN
    bool has_nan = false;
    for (int i = 0; i < 10; ++i) {  // Check first 10 values
        if (std::isnan(C_rm[i])) {
            has_nan = true;
            break;
        }
    }
    
    if (has_nan) {
        std::cout << "❌ FAILED: NaN detected! This reproduces the bug." << std::endl;
    } else {
        std::cout << "✓ PASSED: No NaN. Expected ≈ C_before values." << std::endl;
    }
    
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIPBLAS_CHECK(hipblasDestroy(handle));
    
    std::cout << "\n";
}

int main() {
    std::cout << "hipBLAS Transpose Operation Unit Tests\n";
    std::cout << "======================================\n\n";
    
    test_cm_rm_to_rm_transpose();
    test_accumulation_with_transpose();
    test_leading_dim_variations();
    test_gemm_194_scenario();
    
    std::cout << "All tests completed.\n";
    return 0;
}

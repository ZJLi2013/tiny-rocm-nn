#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Test FP32×FP32 → FP32 matrix multiplication using rocWMMA
// This is critical for v31 implementation
__global__ void test_fp32_input_mma_kernel(
    const float* A,  // FP32 input matrix A (row-major)
    const float* B,  // FP32 input matrix B (col-major)
    float* C         // FP32 output matrix C (row-major)
) {
    using namespace rocwmma;
    
    // v31: Test FP32 fragments for all matrices
    fragment<matrix_a, 16, 16, 16, float, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, float, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    fill_fragment(c_frag, 0.0f);
    
    load_matrix_sync(a_frag, A, 16);
    load_matrix_sync(b_frag, B, 16);
    
    // FP32×FP32 → FP32 MMA
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    store_matrix_sync(C, c_frag, 16, mem_row_major);
}

// CPU reference implementation
void cpu_matmul_fp32(const std::vector<float>& A, const std::vector<float>& B, 
                     std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // A is row-major: A[i][k] at index i*K + k
                // B is col-major: B[k][j] at index j*K + k
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    std::cout << "=== rocWMMA FP32×FP32 → FP32 Test (v31 Validation) ===\n\n";
    
    const int M = 16, N = 16, K = 16;
    const int size = M * N;
    
    // Allocate host memory
    std::vector<float> h_A(size);
    std::vector<float> h_B(size);
    std::vector<float> h_C(size);
    std::vector<float> h_C_ref(size);
    
    // Initialize input matrices
    std::cout << "Initializing FP32 matrices...\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = (i + j) * 0.1f;
        }
    }
    
    // B matrix in col-major format
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            h_B[n * K + k] = (k - n) * 0.1f;
        }
    }
    
    // Compute CPU reference
    std::cout << "Computing CPU reference...\n";
    cpu_matmul_fp32(h_A, h_B, h_C_ref, M, N, K);
    
    std::cout << "CPU Reference C (first 4x4):\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) 
                      << h_C_ref[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size * sizeof(float));
    hipMalloc(&d_B, size * sizeof(float));
    hipMalloc(&d_C, size * sizeof(float));
    
    // Copy data to device
    hipMemcpy(d_A, h_A.data(), size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), size * sizeof(float), hipMemcpyHostToDevice);
    
    // Launch kernel
    std::cout << "Testing FP32×FP32 → FP32 MMA...\n";
    dim3 block(64, 1, 1);  // 64 threads = 1 wave on AMD
    dim3 grid(1, 1, 1);
    
    hipLaunchKernelGGL(test_fp32_input_mma_kernel, grid, block, 0, 0, d_A, d_B, d_C);
    hipDeviceSynchronize();
    
    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "❌ Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        std::cerr << "rocWMMA may not support FP32×FP32 → FP32 on this hardware!\n";
        return 1;
    }
    
    // Copy result back
    hipMemcpy(h_C.data(), d_C, size * sizeof(float), hipMemcpyDeviceToHost);
    
    // Check for NaN
    bool has_nan = false;
    for (int i = 0; i < size; i++) {
        if (std::isnan(h_C[i])) {
            has_nan = true;
            break;
        }
    }
    
    if (has_nan) {
        std::cout << "❌ ERROR: NaN detected in output!\n";
        return 1;
    }
    
    // Compare with CPU reference
    float max_error = 0.0f;
    for (int i = 0; i < size; i++) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "GPU Output C (first 4x4):\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) 
                      << h_C[i * 16 + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
    std::cout << "Max error (FP32×FP32 → FP32): " << max_error << "\n";
    
    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    if (max_error < 1e-5f) {
        std::cout << "\n✅ Test PASSED! rocWMMA supports FP32×FP32 → FP32\n";
        std::cout << "v31 implementation can proceed.\n";
        return 0;
    } else {
        std::cout << "\n❌ Test FAILED! Error too large: " << max_error << "\n";
        return 1;
    }
}

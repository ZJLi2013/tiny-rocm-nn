#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Simple 16x16x16 matrix multiplication using rocWMMA
// C = A * B where A is 16x16, B is 16x16, C is 16x16
__global__ void test_rocwmma_matmul_kernel(
    const __half* A,  // 16x16 input matrix A (row-major)
    const __half* B,  // 16x16 input matrix B (col-major)
    __half* C         // 16x16 output matrix C (row-major)
) {
    using namespace rocwmma;
    
    // Define fragments for 16x16x16 matrix multiplication
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, __half> c_frag;
    
    // Initialize accumulator to zero
    fill_fragment(c_frag, __float2half(0.0f));
    
    // Load matrices from global memory
    // For row-major A: leading dimension is K (number of columns)
    load_matrix_sync(a_frag, A, 16);
    
    // For col-major B: leading dimension is K (number of rows in the column-major layout)
    // In col-major, the leading dimension is the stride between columns
    load_matrix_sync(b_frag, B, 16);
    
    // Perform matrix multiplication: C = A * B + C
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // Store result back to global memory
    store_matrix_sync(C, c_frag, 16, mem_row_major);
}

// Test with FP32 accumulator
__global__ void test_rocwmma_matmul_fp32_accum_kernel(
    const __half* A,
    const __half* B,
    float* C  // FP32 output for FP32 accumulator
) {
    using namespace rocwmma;
    
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;  // FP32 accumulator
    
    fill_fragment(c_frag, 0.0f);
    
    load_matrix_sync(a_frag, A, 16);
    load_matrix_sync(b_frag, B, 16);
    
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    store_matrix_sync(C, c_frag, 16, mem_row_major);
}

// Helper function to check for NaN values
bool has_nan(const std::vector<__half>& data) {
    for (size_t i = 0; i < data.size(); i++) {
        if (std::isnan(__half2float(data[i]))) {
            return true;
        }
    }
    return false;
}

// Helper function to print matrix
void print_matrix(const char* name, const std::vector<__half>& mat, int rows, int cols) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) 
                      << __half2float(mat[i * cols + j]) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// CPU reference implementation for verification
// A is row-major: A[i][k] = A[i*K + k]
// B is col-major: B[k][j] = B[j*K + k]
void cpu_matmul(const std::vector<__half>& A, const std::vector<__half>& B, 
                std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // A is row-major: A[i][k] at index i*K + k
                // B is col-major: B[k][j] at index j*K + k
                sum += __half2float(A[i * K + k]) * __half2float(B[j * K + k]);
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    std::cout << "=== ROCm rocWMMA Basic Test ===\n\n";
    
    const int M = 16, N = 16, K = 16;
    const int size = M * N;
    
    // Allocate host memory
    std::vector<__half> h_A(size);
    std::vector<__half> h_B(size);
    std::vector<__half> h_C(size);
    std::vector<float> h_C_ref(size);
    
    // Initialize input matrices with simple values
    std::cout << "Initializing matrices...\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            // A: row-major, simple pattern
            h_A[i * K + j] = __float2half((i + j) * 0.1f);
        }
    }
    
    // B matrix: We need to store it in col-major format for rocWMMA
    // B[k][n] in mathematical notation, stored as B[n*K + k] in memory
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            // Store in col-major: column n, row k
            h_B[n * K + k] = __float2half((k - n) * 0.1f);
        }
    }
    
    // Print first 4x4 of inputs for debugging
    std::cout << "Input A (first 4x4, row-major):\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) 
                      << __half2float(h_A[i * K + j]) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
    std::cout << "Input B (first 4x4, stored col-major):\n";
    for (int k = 0; k < 4; k++) {
        for (int n = 0; n < 4; n++) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) 
                      << __half2float(h_B[n * K + k]) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
    // Compute CPU reference
    std::cout << "Computing CPU reference...\n";
    cpu_matmul(h_A, h_B, h_C_ref, M, N, K);
    
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
    __half *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size * sizeof(__half));
    hipMalloc(&d_B, size * sizeof(__half));
    hipMalloc(&d_C, size * sizeof(__half));
    
    // Copy data to device
    hipMemcpy(d_A, h_A.data(), size * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), size * sizeof(__half), hipMemcpyHostToDevice);
    
    // Test 1: FP16 Accumulator
    std::cout << "\n--- Test 1: FP16 Accumulator ---\n";
    dim3 block(64, 1, 1);  // 64 threads = 1 wave on AMD (rocWMMA requires 64 threads)
    dim3 grid(1, 1, 1);
    
    hipLaunchKernelGGL(test_rocwmma_matmul_kernel, grid, block, 0, 0, d_A, d_B, d_C);
    hipDeviceSynchronize();
    
    // Check for errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Copy result back
    hipMemcpy(h_C.data(), d_C, size * sizeof(__half), hipMemcpyDeviceToHost);
    
    // Check for NaN
    if (has_nan(h_C)) {
        std::cout << "ERROR: NaN detected in output!\n";
        print_matrix("Output C (with NaN)", h_C, 4, 4);  // Print first 4x4
        return 1;
    }
    
    // Compare with CPU reference
    float max_error = 0.0f;
    for (int i = 0; i < size; i++) {
        float error = std::abs(__half2float(h_C[i]) - h_C_ref[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Max error (FP16 accum): " << max_error << "\n";
    print_matrix("GPU Output C (first 4x4)", h_C, 4, 4);
    
    // Test 2: FP32 accumulator
    std::cout << "\n--- Test 2: FP32 Accumulator ---\n";
    
    // Allocate FP32 device memory for FP32 accumulator output
    float *d_C_fp32;
    hipMalloc(&d_C_fp32, size * sizeof(float));
    hipMemset(d_C_fp32, 0, size * sizeof(float));
    
    hipLaunchKernelGGL(test_rocwmma_matmul_fp32_accum_kernel, grid, block, 0, 0, d_A, d_B, d_C_fp32);
    hipDeviceSynchronize();
    
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        hipFree(d_C_fp32);
        return 1;
    }
    
    // Copy FP32 result back and convert to FP16 for comparison
    std::vector<float> h_C_fp32(size);
    hipMemcpy(h_C_fp32.data(), d_C_fp32, size * sizeof(float), hipMemcpyDeviceToHost);
    
    // Check for NaN in FP32 output
    bool has_nan_fp32 = false;
    for (int i = 0; i < size; i++) {
        if (std::isnan(h_C_fp32[i])) {
            has_nan_fp32 = true;
            break;
        }
    }
    
    if (has_nan_fp32) {
        std::cout << "ERROR: NaN detected in FP32 output!\n";
        return 1;
    }
    
    max_error = 0.0f;
    for (int i = 0; i < size; i++) {
        float error = std::abs(h_C_fp32[i] - h_C_ref[i]);
        max_error = std::max(max_error, error);
    }
    
    std::cout << "Max error (FP32 accum): " << max_error << "\n";
    std::cout << "GPU Output C (first 4x4, FP32):\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) 
                      << h_C_fp32[i * 16 + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
    hipFree(d_C_fp32);
    
    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    if (max_error < 0.01f) {
        std::cout << "\n✓ Test PASSED!\n";
        return 0;
    } else {
        std::cout << "\n✗ Test FAILED! Error too large.\n";
        return 1;
    }
}

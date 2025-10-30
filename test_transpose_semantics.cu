#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK_THROW(x) \
    do { \
        cudaError_t result = x; \
        if (result != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(result)); \
        } \
    } while(0)

enum MatrixLayout {
    RowMajor = 0,
    ColumnMajor = 1,
    RM = RowMajor,
    CM = ColumnMajor
};

// Simplified GPU matrix for testing
template<typename T>
struct SimpleGPUMatrix {
    T* data;
    int m, n;
    int stride;
    MatrixLayout layout;
    bool owns_data;
    
    SimpleGPUMatrix(int m_, int n_, MatrixLayout layout_) 
        : m(m_), n(n_), layout(layout_), owns_data(true) {
        stride = (layout == CM) ? m : n;
        CUDA_CHECK_THROW(cudaMalloc(&data, m * n * sizeof(T)));
    }
    
    SimpleGPUMatrix(T* data_, int m_, int n_, MatrixLayout layout_, int stride_)
        : data(data_), m(m_), n(n_), layout(layout_), stride(stride_), owns_data(false) {}
    
    ~SimpleGPUMatrix() {
        if (owns_data && data) {
            cudaFree(data);
        }
    }
    
    SimpleGPUMatrix<T> transposed() const {
        MatrixLayout new_layout = (layout == CM) ? RM : CM;
        return SimpleGPUMatrix<T>(data, n, m, new_layout, stride);
    }
    
    void copy_from_host(const std::vector<T>& h_data) {
        CUDA_CHECK_THROW(cudaMemcpy(data, h_data.data(), m * n * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    void copy_to_host(std::vector<T>& h_data) const {
        h_data.resize(m * n);
        CUDA_CHECK_THROW(cudaMemcpy(h_data.data(), data, m * n * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

// Helper function to print matrix
template<typename T>
void print_matrix_cpu(const std::vector<T>& data, int m, int n, MatrixLayout layout, const char* name) {
    std::cout << "\n" << name << " (" << m << "x" << n << ", " 
              << (layout == CM ? "CM" : "RM") << "):\n";
    
    for (int i = 0; i < std::min(m, 8); ++i) {
        for (int j = 0; j < std::min(n, 8); ++j) {
            float val;
            if (layout == CM) {
                val = (float)data[i + j * m];  // Column-major: column index * rows + row index
            } else {
                val = (float)data[i * n + j];  // Row-major: row index * cols + col index
            }
            printf("%6.2f ", val);
        }
        if (n > 8) std::cout << "...";
        std::cout << "\n";
    }
    if (m > 8) std::cout << "...\n";
}

int main() {
    std::cout << "=== Testing Matrix Transpose Semantics ===\n";
    
    // Test case: Create a simple matrix and verify transpose behavior
    const int M = 4;
    const int N = 6;
    
    // Test 1: Row-Major Matrix
    {
        std::cout << "\n--- Test 1: Row-Major Matrix ---\n";
        
        SimpleGPUMatrix<__half> A(M, N, RM);
        
        // Initialize with simple pattern: A[i,j] = i*10 + j
        std::vector<__half> h_data(M * N);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                h_data[i * N + j] = __float2half(i * 10.0f + j);
            }
        }
        
        A.copy_from_host(h_data);
        
        std::cout << "Original matrix A: " << A.m << "x" << A.n 
                  << " layout=" << (A.layout == CM ? "CM" : "RM")
                  << " stride=" << A.stride << "\n";
        
        // Get transposed view
        auto A_T = A.transposed();
        
        std::cout << "Transposed A_T: " << A_T.m << "x" << A_T.n
                  << " layout=" << (A_T.layout == CM ? "CM" : "RM")
                  << " stride=" << A_T.stride << "\n";
        
        // Copy back and verify
        std::vector<__half> h_A, h_A_T;
        A.copy_to_host(h_A);
        A_T.copy_to_host(h_A_T);
        
        print_matrix_cpu(h_A, M, N, RM, "A (original)");
        print_matrix_cpu(h_A_T, N, M, CM, "A_T (transposed view)");
        
        // Verify: A_T[j,i] should equal A[i,j]
        bool correct = true;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float a_ij = __half2float(h_A[i * N + j]);
                float at_ji = __half2float(h_A_T[j + i * N]);  // A_T is CM: j + i*stride
                
                if (std::abs(a_ij - at_ji) > 1e-5) {
                    std::cout << "ERROR: A[" << i << "," << j << "]=" << a_ij
                              << " != A_T[" << j << "," << i << "]=" << at_ji << "\n";
                    correct = false;
                }
            }
        }
        
        if (correct) {
            std::cout << "✓ Transpose semantics CORRECT for RM matrix\n";
        } else {
            std::cout << "✗ Transpose semantics INCORRECT for RM matrix\n";
        }
    }
    
    // Test 2: Column-Major Matrix
    {
        std::cout << "\n--- Test 2: Column-Major Matrix ---\n";
        
        SimpleGPUMatrix<__half> B(M, N, CM);
        
        // Initialize with simple pattern: B[i,j] = i*10 + j
        std::vector<__half> h_data(M * N);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                h_data[i + j * M] = __float2half(i * 10.0f + j);  // CM storage
            }
        }
        
        B.copy_from_host(h_data);
        
        std::cout << "Original matrix B: " << B.m << "x" << B.n
                  << " layout=" << (B.layout == CM ? "CM" : "RM")
                  << " stride=" << B.stride << "\n";
        
        // Get transposed view
        auto B_T = B.transposed();
        
        std::cout << "Transposed B_T: " << B_T.m << "x" << B_T.n
                  << " layout=" << (B_T.layout == CM ? "CM" : "RM")
                  << " stride=" << B_T.stride << "\n";
        
        // Copy back and verify
        std::vector<__half> h_B, h_B_T;
        B.copy_to_host(h_B);
        B_T.copy_to_host(h_B_T);
        
        print_matrix_cpu(h_B, M, N, CM, "B (original)");
        print_matrix_cpu(h_B_T, N, M, RM, "B_T (transposed view)");
        
        // Verify: B_T[j,i] should equal B[i,j]
        bool correct = true;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float b_ij = __half2float(h_B[i + j * M]);
                float bt_ji = __half2float(h_B_T[j * M + i]);  // B_T is RM: j*stride + i
                
                if (std::abs(b_ij - bt_ji) > 1e-5) {
                    std::cout << "ERROR: B[" << i << "," << j << "]=" << b_ij
                              << " != B_T[" << j << "," << i << "]=" << bt_ji << "\n";
                    correct = false;
                }
            }
        }
        
        if (correct) {
            std::cout << "✓ Transpose semantics CORRECT for CM matrix\n";
        } else {
            std::cout << "✗ Transpose semantics INCORRECT for CM matrix\n";
        }
    }
    
    std::cout << "\n=== Test Complete ===\n";
    
    return 0;
}

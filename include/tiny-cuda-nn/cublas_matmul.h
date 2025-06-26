#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>

#include <cublas_v2.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <iostream> 
#include <type_traits>

#include <tiny-cuda-nn/debug_config.h> // for cublas matrix print 

namespace tcnn {

#define Cublas_CHECK_THROW(status) \
    do { \
        cublasStatus_t _status = (status); \
        if (_status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error: " + std::to_string(_status)); \
        } \
    } while (0)

using TypeAccumulator = std::conditional_t<std::is_same<network_precision_t, float>::value, float, __half>;
using TypeCompute = std::conditional_t<std::is_same<network_precision_t, float>::value, float, __half>;

// Forward declarations of kernel functions
template <typename ElementAccumulator, int kCount, typename Activation>
__global__ void fused_activation_kernel(Activation activation, ElementAccumulator* matrix, int num_elements);

template <typename ElementAccumulator, int kCount, typename Activation>
__global__ void fused_activation_backward_kernel(
    Activation activation, 
    ElementAccumulator* gradient, 
    ElementAccumulator* source, 
    int num_elements
);

template <
	typename ElementOutput_,                             ///< Data type used to load and store tensors
	int Count,                                           ///< Number of elements computed per operation
	typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
	typename ElementCompute_ = ElementOutput_          ///< Data type used to compute linear combination
>
class ActivationEpilogue {
public: 
	using ElementOutput = ElementOutput_;
	using ElementAccumulator = ElementAccumulator_;
	using ElementCompute = ElementCompute_;
    static int const kCount = Count; 
	struct Params {
		Activation activation;
		bool sum_source;
	};
public:
	ActivationEpilogue(Params const &params = {Activation::None, false}) 
        : m_activation{params.activation}, m_sum_source{params.sum_source} { }
	
    bool is_source_needed() const {
		return m_sum_source;
	}
    
    // Fused activation kernel (no temporary allocations)
    __device__ __forceinline__ ElementAccumulator apply_activation(ElementAccumulator value) const {
        switch (m_activation) {
            case Activation::None:     return value;
            case Activation::ReLU:     return value > 0 ? value : 0;
            case Activation::Exponential: return expf(value);
            case Activation::Sine:     return sinf(value);
            case Activation::Sigmoid:  return 1.0f / (1.0f + expf(-value));
            case Activation::Squareplus: return 0.5f * (value + sqrtf(value * value + 4));
            case Activation::Softplus: return logf(1.0f + expf(value));
            default: return value;
        }
    }
    
    // Optimized fused GEMM + activation operator
    void operator()(ElementAccumulator* accumulator, int m, int n) const {
        const int total_elements = m * n;
        const int threads_per_block = 256;
        const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        // Launch fused kernel with proper thread indexing
        fused_activation_kernel<ElementAccumulator, kCount>
            <<<blocks, threads_per_block>>>(m_activation, accumulator, total_elements);
        cudaStreamSynchronize(0);
    } 

    // Backward pass implementation
    void operator()(ElementAccumulator* accumulator, ElementOutput* source, int m, int n) const {
        const int total_elements = m * n;
        const int threads_per_block = 256;
        const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        fused_activation_backward_kernel<ElementAccumulator, kCount>
            <<<blocks, threads_per_block>>>(m_activation, accumulator, source, total_elements);
        cudaStreamSynchronize(0);
    } 

private:
	Activation m_activation;
	bool m_sum_source;
}; 

template <
	typename ElementOutput_,                             ///< Data type used to load and store tensors
	int Count,                                           ///< Number of elements computed per operation
	typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
	typename ElementCompute_ = ElementOutput_          ///< Data type used to compute linear combination
>
class ActivationTransferEpilogue {
public: 
	using ElementOutput = ElementOutput_;
	using ElementAccumulator = ElementAccumulator_;
	using ElementCompute = ElementCompute_;
    static int const kCount = Count; 

	struct Params {
		Activation activation;
	};
public:
	ActivationTransferEpilogue(Params const &params = {Activation::None}) 
        : m_activation{params.activation} { }

	bool is_source_needed() const {
		return true;
	}

    // Optimized implementation that directly applies activation derivative
    void operator()(ElementAccumulator* gradient, ElementOutput* source, int m, int n) const {
        const int total_elements = m * n;
        const int threads_per_block = 256;
        const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        fused_activation_backward_kernel<ElementAccumulator, kCount>
            <<<blocks, threads_per_block>>>(m_activation, gradient, source, total_elements);
        cudaStreamSynchronize(0);
    }

    void operator()(ElementAccumulator* accumulator, int m, int n) const {
        std::cout << "ActivationTransferEpilogue: Source matrix required for backward pass" << std::endl;
    }

private:
	Activation m_activation;
}; 

// Vectorized element count optimized for modern GPU architectures
template <typename T>
static constexpr int n_vectorized_elements = 
    (!std::is_same<T, float>::value) ? (128 / sizeof(T)) : 
    (128 / (sizeof(T) * 4)); // Use wider vectors for float

template <typename T>
using ActivationOp = ActivationEpilogue<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

template <typename T>
using ActivationTransferOp = ActivationTransferEpilogue<T, n_vectorized_elements<T>, TypeAccumulator, TypeCompute>;

// Template structure for gemm op
template<typename EPILOGUE, typename T>
struct OurGemmWrapper; 

cudaDataType_t getCUDADatatype(const std::type_info &type)
{
    if (type == typeid(float)){
        return CUDA_R_32F;
    } else if (type == typeid(__half)){
        return CUDA_R_16F;
    }
    std::cout << "Unsupported data type" << std::endl;
    exit(EXIT_FAILURE);
}

// mimic cutlass::epilogue
// as cutlass::epilogue op is smoothly using previous GEMM grid/block/warps mapping.
// basically the fragment memory operated by the warp and threads in the warp can directly used for epilogue op again.
template<typename EPILOGUE, typename T>
void OurGemm(cublasHandle_t handle,
                  cublasOperation_t TransA,
                  cublasOperation_t TransB,
                  int m, int n, int k,
                  const void *alpha,
                  const void *A, int lda,
                  const void *B, int ldb,
                  const void *beta,
                  void *C, int ldc) {
    
    cudaDataType_t dataType = CUDA_R_32F;
    if (!std::is_same<T, float>::value){
        dataType = getCUDADatatype(typeid(__half)); 
    } 
    cublasStatus_t status = cublasGemmEx(handle, TransA, TransB,
                                         m, n, k,
                                         alpha,
                                         A, dataType, lda,
                                         B, dataType, ldb,
                                         beta,
                                         C, dataType, ldc,
                                         dataType, // Compute type
                                         CUBLAS_GEMM_DEFAULT);
                                         
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS GEMM failed");
    }
    // do activation op on matrix C 
    EPILOGUE myActivation; 
    if (myActivation.is_source_needed()){
        // Apply activation with source matrix (for backward pass)
        myActivation(
            static_cast<typename EPILOGUE::ElementAccumulator*>(C),
            static_cast<typename EPILOGUE::ElementOutput*>(C),
            m, n
        ); 
    } else {
        // Apply activation without source matrix (for forward pass)
        myActivation(
            static_cast<typename EPILOGUE::ElementAccumulator*>(C),
            m, n
        ); 
    } 
}

// specialization for float OurGemm 
template<typename EPILOGUE>
struct OurGemmWrapper<EPILOGUE, float>{
    static cublasStatus_t gemm(cublasHandle_t handle,
                  cublasOperation_t TransA,
                  cublasOperation_t TransB,
                  int m, int n, int k,
                  const void *alpha,
                  const void *A, int lda,
                  const void *B, int ldb,
                  const void *beta,
                  void *C, int ldc)
    {
#ifdef DEBUG_MODE
        std::cout << "[DEBUG]: launch float OurGemmWrapper" << std::endl; 
#endif
        return OurGemm<EPILOGUE, float>(handle, TransA, TransB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); 
    }
}; 

// specialization for __half OurGemm 
template<typename EPILOGUE>
struct OurGemmWrapper<EPILOGUE, __half>{
    static cublasStatus_t gemm(cublasHandle_t handle,
                  cublasOperation_t TransA,
                  cublasOperation_t TransB,
                  int m, int n, int k,
                  const void *alpha,
                  const void *A, int lda,
                  const void *B, int ldb,
                  const void *beta,
                  void *C, int ldc)
    {
#ifdef DEBUG_MODE
        std::cout << "[DEBUG]: launch half OurGemmWrapper" << std::endl; 
#endif
        return OurGemm<EPILOGUE, __half>(handle, TransA, TransB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); 
    }
}; 

template<typename T>
void OurSplitGemm(cublasHandle_t handle,      
                  cublasOperation_t TransA,
                  cublasOperation_t TransB,
                  const int m, const int n, const int k,
                  const void* alpha,
                  const T *A, int lda,
                  const T *B, int ldb,
                  const void* beta,
                  T *C, int ldc,
                  int split_k_slices)
{
    cudaDataType_t dataType = getCUDADatatype(typeid(network_precision_t));
    if (split_k_slices == 1){
//        std::cout << "[DEBUG] split_k_slice==1" << std::endl; 
        cublasStatus_t status = cublasGemmEx(handle, TransA, TransB,
                                            m, n, k,
                                            alpha,
                                            A, dataType, lda,
                                            B, dataType, ldb,
                                            beta,
                                            C, dataType, ldc,
                                            dataType, 
                                            CUBLAS_GEMM_DEFAULT);
                                            
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS GEMM failed");
        }       
    } else {
        // Split-K GEMM implementation
        const T* B_slice;
        const T* A_slice;   
        int k_slice = k / split_k_slices;
        for (int slice = 0; slice < split_k_slices; ++slice) {
            //TODO: data access consider layout 
            A_slice = static_cast<const T*>(A) + slice * k_slice;
            B_slice = static_cast<const T*>(B) + slice * k_slice * ldb;
            cublasStatus_t status = cublasGemmEx(handle, TransA, TransB,
                                                m, n, k_slice,
                                                alpha,
                                                A_slice, dataType, lda,
                                                B_slice, dataType, ldb,
                                                beta,
                                                C, dataType, ldc,
                                                dataType, 
                                                CUBLAS_GEMM_DEFAULT);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cuBLAS GEMM failed");
            } 
        }
    }
};

template <typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, MatrixLayout LayoutC, typename TypeD, MatrixLayout LayoutD>
void fc_multiply(cublasHandle_t &handle, cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrix<TypeC, LayoutC>& C, GPUMatrix<TypeD, LayoutD>& D, Activation act = Activation::None, bool transfer = false, bool sum_source = false) {

     cublasOperation_t TransA = (LayoutA == MatrixLayout::RowMajor) ? CUBLAS_OP_T : CUBLAS_OP_N;
     cublasOperation_t TransB = (LayoutB == MatrixLayout::RowMajor) ? CUBLAS_OP_T : CUBLAS_OP_N;
     cublasOperation_t TransC = (LayoutC == MatrixLayout::RowMajor) ? CUBLAS_OP_T : CUBLAS_OP_N;

    static_assert(std::is_same<TypeA, TypeB>::value, "Type of matrix A and B must be equal");
    static_assert(std::is_same<TypeC, TypeD>::value, "Type of matrix C and D must be equal");

	using MatmulTypeCompute = std::conditional_t<std::is_same<TypeA, float>::value, float, __half>;
	using MatmulTypeAccumulator = std::conditional_t<std::is_same<TypeC, float>::value, float, __half>;    

    if (A.n() != B.m()) {
        throw std::runtime_error("Matrices A and B cannot be multiplied together");
    }

    const int M = A.m();
    const int K = A.n();
    const int N = B.n();

    if (C.m() != M || C.n() != N) {
        throw std::runtime_error(fmt::format("Matrix C has incorrect size {}x{} != {}x{}", C.m(), C.n(), M, N));
    }

    if (D.m() != M || D.n() != N) {
        throw std::runtime_error(fmt::format("Matrix D has incorrect size {}x{} != {}x{}", D.m(), D.n(), M, N));
    }

    int lda = A.stride(); 
    int ldb = B.stride();
    int ldc = C.stride();     

    network_precision_t alpha = 1.0f;
    network_precision_t beta = sum_source ? 1.0f : 0.0f;

    if (transfer) {
        // For backward pass (transfer = true)
        typename ActivationTransferOp<MatmulTypeAccumulator>::Params params;
        params.activation = act;
        OurGemmWrapper<ActivationTransferOp<MatmulTypeAccumulator>, network_precision_t>::gemm(
            handle, TransA, TransB, M, N, K, &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc
        );
    } else {
        // For forward pass (transfer = false)
        typename ActivationOp<MatmulTypeAccumulator>::Params params;
        params.activation = act;
        params.sum_source = sum_source;
        OurGemmWrapper<ActivationOp<MatmulTypeAccumulator>, network_precision_t>::gemm(
            handle, TransA, TransB, M, N, K, &alpha, A.data(), lda, B.data(), ldb, &beta, C.data(), ldc
        );
    }
}

template <typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, typename TypeD>
void fc_multiply(cublasHandle_t &handle, cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrixDynamic<TypeC>& C, const GPUMatrixDynamic<TypeD>& D, Activation act = Activation::None, bool transfer = false, bool sum_source = false) {
	if (C.layout() != D.layout()) {
		throw std::runtime_error{"fc_multiply: Layout of GPUMatrixDynamic C and D must be equal"};
	}

	if (D.layout() == CM) {
		fc_multiply(handle, stream, A, B, C.cm(), D.cm(), act, transfer, sum_source);
	} else {
		fc_multiply(handle, stream, A, B, C.rm(), D.rm(), act, transfer, sum_source);
	}
}

template <typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply(cublasHandle_t &handle, cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeC>& C, const GPUMatrixDynamic<TypeD>& D, Activation act = Activation::None, bool transfer = false, bool sum_source = false) {
	if (B.layout() == CM) {
		fc_multiply(handle, stream, A, B.cm(), C, D, act, transfer, sum_source);
	} else {
		fc_multiply(handle, stream, A, B.rm(), C, D, act, transfer, sum_source);
	}
}

template <typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeD>
void fc_multiply(cublasHandle_t &handle, cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeD>& D, Activation act = Activation::None) {
	fc_multiply(handle, stream, A, B, D, D, act);
}

template<typename T>
__global__ void convertColumnMajorToRowMajorKernel(T* matrix_col_major, T* matrix_row_major, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < cols) {
        // Column-major index: j * rows + i
        // Row-major index: i * cols + j
        matrix_row_major[i * cols + j] = matrix_col_major[j * rows + i];
    }
}

template<typename T>
void convertColumnMajorToRowMajor_GPU(T* d_matrix_col_major, T* d_matrix_row_major, int rows, int cols) {
    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);
    convertColumnMajorToRowMajorKernel<<<gridSize, blockSize>>>(d_matrix_col_major, d_matrix_row_major, rows, cols);
    cudaDeviceSynchronize();
}

template <typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, MatrixLayout LayoutC, typename TypeD, MatrixLayout LayoutD>
void fc_multiply_split_k(cublasHandle_t handle, cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrix<TypeC, LayoutC>& C, const GPUMatrix<TypeD, LayoutD>& D, int split_k_slices = 1, float beta = 0.0f) {
    
    cublasOperation_t TransA = (LayoutA == MatrixLayout::RowMajor) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t TransB = (LayoutB == MatrixLayout::RowMajor) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t TransC = (LayoutC == MatrixLayout::RowMajor) ? CUBLAS_OP_T : CUBLAS_OP_N;   
    
    static_assert(std::is_same<TypeA, TypeB>::value, "Type of matrix A and B must be equal");
    static_assert(std::is_same<TypeC, TypeD>::value, "Type of matrix C and D must be equal");
 
 	using MatmulTypeCompute = std::conditional_t<std::is_same<TypeA, float>::value, float, __half>;
	using MatmulTypeAccumulator = std::conditional_t<std::is_same<TypeC, float>::value, float, __half>;    

    if (A.n() != B.m()) {
        throw std::runtime_error("Matrices A and B cannot be multiplied together");
    }

    const int M = A.m();
    const int K = A.n();
    const int N = B.n();

    if (C.m() != M || C.n() != N) {
        throw std::runtime_error(fmt::format("Matrix C has incorrect size {}x{} != {}x{}", C.m(), C.n(), M, N));
    }

    if (D.m() != M || D.n() != N) {
        throw std::runtime_error(fmt::format("Matrix D has incorrect size {}x{} != {}x{}", D.m(), D.n(), M, N));
    }

    // A(m, k), B(k, n), C(m, n) , leadning-dim only relate to physical memory layout, no matter T or N 
    // 因为tiny-cuda-nn 使用了自己的 gpu_matrix 类，并不是默认cublas 的layout。所以，这里leading-dim 需要跟 matrix 的stride 一致
    int lda = A.stride(); 
    int ldb = B.stride();
    int ldc = C.stride();

    // cublasSetStream(handle, stream);
    // TODO: need specify ComputeType and AccumulatorType 
    network_precision_t alpha = __float2half(1.0); 
    network_precision_t half_beta = __float2half(beta);  // for splitK case, need to accumulate C from each split to form final C matrix 

    if (TransC == CUBLAS_OP_N) {  // col-major in cublas, then use output C correctly 
        OurSplitGemm<network_precision_t>(handle, TransA, TransB, M, N, K, &alpha, A.data(), lda, B.data(), ldb, &half_beta, C.data(), ldc, split_k_slices); 
    } else if (TransC == CUBLAS_OP_T) {
        // as the memory for output C as row-major, while cublas consider C in col-major by default 
        printf("[DEBUG], matC is pre-allocated as row-major in memory, while cublas consider C in col-major. running memory layout convert here\n");
        network_precision_t *C2; 
        CUDA_CHECK_THROW(cudaMalloc((void**)&C2, C.rows() * C.cols() * sizeof(network_precision_t))); // interpret C2 as col-major
        int ldc2 = C.rows(); 
        OurSplitGemm<network_precision_t>(handle, TransA, TransB, M, N, K, &alpha, A.data(), lda, B.data(), ldb, &half_beta, C2, ldc2, split_k_slices); 
        convertColumnMajorToRowMajor_GPU(C2, C.data(), C.rows(), C.cols()); 
        CUDA_CHECK_THROW(cudaFree(C2));
    }
}

template <typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, typename TypeD>
void fc_multiply_split_k(cublasHandle_t handle, cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrix<TypeB, LayoutB>& B, const GPUMatrixDynamic<TypeC>& C, const GPUMatrixDynamic<TypeD>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (C.layout() != D.layout()) {
		throw std::runtime_error{"fc_multiply: Layout of GPUMatrixDynamic C and D must be equal"};
	}

	if (D.layout() == CM) {
		fc_multiply_split_k<TypeA, LayoutA, TypeB, LayoutB, TypeC, CM, TypeD, CM>(handle, stream, A, B, C.cm(), D.cm(), split_k_slices, beta);
	} else {        
		fc_multiply_split_k<TypeA, LayoutA, TypeB, LayoutB, TypeC, RM, TypeD, RM>(handle, stream, A, B, C.rm(), D.rm(), split_k_slices, beta);
	}
}

template <typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply_split_k(cublasHandle_t handle, cudaStream_t stream, const GPUMatrix<TypeA, LayoutA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeC>& C, const GPUMatrixDynamic<TypeD>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (B.layout() == CM) {
		fc_multiply_split_k(handle, stream, A, B.cm(), C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k(handle, stream, A, B.rm(), C, D, split_k_slices, beta);
	}
}

template <typename TypeA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply_split_k(cublasHandle_t handle, cudaStream_t stream, const GPUMatrixDynamic<TypeA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeC>& C, const GPUMatrixDynamic<TypeD>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (A.layout() == CM) {
		fc_multiply_split_k(handle, stream, A.cm(), B, C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k(handle, stream, A.rm(), B, C, D, split_k_slices, beta);
	}
}

template <typename TypeA, typename TypeB, typename TypeD>
void fc_multiply_split_k(cublasHandle_t handle, cudaStream_t stream, const GPUMatrixDynamic<TypeA>& A, const GPUMatrixDynamic<TypeB>& B, const GPUMatrixDynamic<TypeD>& D, int split_k_slices, float beta) {
	fc_multiply_split_k(handle, stream, A, B, D, D, split_k_slices, beta);
}

// Kernel implementations

// Fused activation kernel (forward pass)
template <typename ElementAccumulator, int kCount, typename Activation>
__global__ void fused_activation_kernel(Activation activation, ElementAccumulator* matrix, int num_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = idx * kCount;
    
    if (offset < num_elements) {
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < kCount; ++i) {
            const int element_idx = offset + i;
            if (element_idx < num_elements) {
                // Apply activation directly to matrix elements
                matrix[element_idx] = activation.apply_activation(matrix[element_idx]);
            }
        }
    }
}

// Fused activation kernel (backward pass)
template <typename ElementAccumulator, int kCount, typename Activation>
__global__ void fused_activation_backward_kernel(
    Activation activation, 
    ElementAccumulator* gradient, 
    ElementAccumulator* source, 
    int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = idx * kCount;
    
    if (offset < num_elements) {
        TCNN_PRAGMA_UNROLL
        for (int i = 0; i < kCount; ++i) {
            const int element_idx = offset + i;
            if (element_idx < num_elements) {
                // Apply activation derivative
                const ElementAccumulator src_val = source[element_idx];
                ElementAccumulator grad = gradient[element_idx];
                
                switch (activation.m_activation) {
                    case Activation::ReLU:
                        grad *= (src_val > 0) ? 1.0f : 0.0f;
                        break;
                    case Activation::Sigmoid: {
                        const ElementAccumulator s = 1.0f / (1.0f + expf(-src_val));
                        grad *= s * (1 - s);
                        break;
                    }
                    case Activation::Exponential:
                        grad *= expf(src_val);
                        break;
                    case Activation::Sine:
                        grad *= cosf(src_val);
                        break;
                    case Activation::Squareplus: {
                        const ElementAccumulator denom = sqrtf(src_val * src_val + 4);
                        grad *= 0.5f * (1 + src_val / denom);
                        break;
                    }
                    case Activation::Softplus:
                        grad *= 1.0f / (1.0f + expf(-src_val));
                        break;
                    default:
                        // No change for other activations
                        break;
                }
                
                gradient[element_idx] = grad;
            }
        }
    }
}

}

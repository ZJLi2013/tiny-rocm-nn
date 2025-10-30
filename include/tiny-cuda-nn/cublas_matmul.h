/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   cublas_matmul.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Matrix multiplication wrappers that call into cuBLAS.
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>

#include <cublas_v2.h>

#include <iostream>
#include <type_traits>
#include <vector>
#include <iomanip>

namespace tcnn {

// Helper function to print matrix values for debugging
template <typename T>
void print_matrix_sample(const char* name, const T* data, int m, int n, int stride, MatrixLayout layout, cudaStream_t stream = 0) {
	const int sample_size = std::min(4, std::min(m, n));
	std::vector<T> h_data(m * n);
	
	cudaStreamSynchronize(stream);
	CUDA_CHECK_THROW(cudaMemcpy(h_data.data(), data, m * n * sizeof(T), cudaMemcpyDeviceToHost));
	
	std::cout << name << " (" << m << "x" << n << ", " << (layout == CM ? "CM" : "RM") 
	          << ", stride=" << stride << ") sample:" << std::endl;
	
	for (int i = 0; i < sample_size; ++i) {
		std::cout << "  ";
		for (int j = 0; j < sample_size; ++j) {
			float val;
			if (layout == CM) {
				val = (float)h_data[i + j * stride];
			} else {
				val = (float)h_data[i * stride + j];
			}
			std::cout << std::setw(8) << std::fixed << std::setprecision(3) << val << " ";
		}
		std::cout << std::endl;
	}
}

#define CUBLAS_CHECK_THROW(x)                                                                                        \
	do {                                                                                                                   \
		cublasStatus_t _result = x;                                                                                    \
		if (_result != CUBLAS_STATUS_SUCCESS)                                                                            \
			throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + std::to_string(_result)); \
	} while(0)


inline cublasHandle_t& cublas_handle() {
	static cublasHandle_t handle;
	static bool initialized = false;
	if (!initialized) {
		CUBLAS_CHECK_THROW(cublasCreate(&handle));
		initialized = true;
	}
	return handle;
}

template <typename T>
void cublas_gemm(
	cudaStream_t stream,
	const GPUMatrix<T, RM>& A,
	const GPUMatrix<T, RM>& B,
	GPUMatrix<T, RM>& C,
	float alpha = 1.0f,
	float beta = 0.0f
) {
	if (A.n() != B.m()) {
		throw std::runtime_error("Matrices A and B can not be multiplied together");
	}

	const int m = A.m();
	const int k = A.n();
	const int n = B.n();

	if (C.m() != m || C.n() != n) {
		throw std::runtime_error{fmt::format("Matrix C has incorrect size {}x{} != {}x{}", C.m(), C.n(), m, n)};
	}

	cublasSetStream(cublas_handle(), stream);

	cudaDataType_t cuda_data_type = std::is_same<T, float>::value ? CUDA_R_32F : CUDA_R_16F;
	cublasComputeType_t compute_type = std::is_same<T, float>::value ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_16F;
	
	// Since all matrices are row-major, we can use the identity (A*B)^T = B^T * A^T
	// and compute C_cm = B_cm * A_cm, which is equivalent to C_rm = A_rm * B_rm
	// but with swapped arguments.
	CUBLAS_CHECK_THROW(cublasGemmEx(
		cublas_handle(),
		CUBLAS_OP_N, CUBLAS_OP_N,
		n, m, k,
		&alpha,
		B.data(), cuda_data_type, B.stride(),
		A.data(), cuda_data_type, A.stride(),
		&beta,
		C.data(), cuda_data_type, C.stride(),
		compute_type,
		CUBLAS_GEMM_DEFAULT_TENSOR_OP
	));
}

template <typename T>
void cublas_gemm(
	cudaStream_t stream,
	const GPUMatrix<T, CM>& A,
	const GPUMatrix<T, CM>& B,
	GPUMatrix<T, CM>& C,
	float alpha = 1.0f,
	float beta = 0.0f
) {
	if (A.n() != B.m()) {
		throw std::runtime_error("Matrices A and B can not be multiplied together");
	}

	const int m = A.m();
	const int k = A.n();
	const int n = B.n();

	if (C.m() != m || C.n() != n) {
		throw std::runtime_error{fmt::format("Matrix C has incorrect size {}x{} != {}x{}", C.m(), C.n(), m, n)};
	}

	cublasSetStream(cublas_handle(), stream);

	cudaDataType_t cuda_data_type = std::is_same<T, float>::value ? CUDA_R_32F : CUDA_R_16F;
	cublasComputeType_t compute_type = std::is_same<T, float>::value ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_16F;

	CUBLAS_CHECK_THROW(cublasGemmEx(
		cublas_handle(),
		CUBLAS_OP_N, CUBLAS_OP_N,
		m, n, k,
		&alpha,
		A.data(), cuda_data_type, A.stride(),
		B.data(), cuda_data_type, B.stride(),
		&beta,
		C.data(), cuda_data_type, C.stride(),
		compute_type,
		CUBLAS_GEMM_DEFAULT_TENSOR_OP
	));
}

// Fallback for mixed layouts (less efficient due to potential transposes)
template <typename T, MatrixLayout LA, MatrixLayout LB, MatrixLayout LC>
void cublas_gemm(
	cudaStream_t stream,
	const GPUMatrix<T, LA>& A,
	const GPUMatrix<T, LB>& B,
	GPUMatrix<T, LC>& C,
	float alpha = 1.0f,
	float beta = 0.0f
) {
	static int mixed_call_count = 0;
	mixed_call_count++;
	
	if (A.n() != B.m()) {
		throw std::runtime_error("Matrices A and B can not be multiplied together");
	}

	const int m = A.m();
	const int k = A.n();
	const int n = B.n();

	if (C.m() != m || C.n() != n) {
		throw std::runtime_error{fmt::format("Matrix C has incorrect size {}x{} != {}x{}", C.m(), C.n(), m, n)};
	}

	cublasSetStream(cublas_handle(), stream);

	cudaDataType_t cuda_data_type = std::is_same<T, float>::value ? CUDA_R_32F : CUDA_R_16F;
	cublasComputeType_t compute_type = std::is_same<T, float>::value ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_16F;
	
	// For FP16 data with CUBLAS_COMPUTE_32F_FAST_16F, alpha/beta should be float
	// This allows FP16 inputs with FP32 accumulation

	// Debug output for first few mixed layout calls
	if (mixed_call_count <= 3) {
		std::cout << "\n[cublas_gemm MIXED LAYOUT Call #" << mixed_call_count << "]" << std::endl;
		std::cout << "  A: " << m << "x" << k << " (" << (LA == CM ? "CM" : "RM") << ", stride=" << A.stride() << ")" << std::endl;
		std::cout << "  B: " << k << "x" << n << " (" << (LB == CM ? "CM" : "RM") << ", stride=" << B.stride() << ")" << std::endl;
		std::cout << "  C: " << m << "x" << n << " (" << (LC == CM ? "CM" : "RM") << ", stride=" << C.stride() << ")" << std::endl;
		std::cout << "  alpha=" << alpha << ", beta=" << beta << std::endl;
	}
	
	// For mixed layouts, we need to carefully handle the transpose operations
	// cuBLAS is column-major, so we interpret RM matrices as transposed CM matrices
	
	if (LC == RM) {
		// Output is RM: C_rm (m×n) = A (m×k) * B (k×n)
		// Use identity: C_rm = A * B ⟺ C_cm^T = B^T * A^T
		// 
		// Strategy: Compute C^T using cuBLAS (which outputs CM)
		// cuBLAS computes: C_cm^T (n×m) = first_matrix * second_matrix
		// We want: C_cm^T (n×m) = B^T (n×k) * A^T (k×m)
		// 
		// Key: RM matrices are already "transposed" when viewed as CM
		// - RM (m×n, stride=n) viewed as CM is (n×m, stride=n)
		// - So we use CUBLAS_OP_N (no additional transpose needed)
		// - CM matrices need CUBLAS_OP_T to transpose them
		cublasOperation_t op_a = LA == RM ? CUBLAS_OP_N : CUBLAS_OP_T;
		cublasOperation_t op_b = LB == RM ? CUBLAS_OP_N : CUBLAS_OP_T;
		
		if (mixed_call_count <= 3) {
			std::cout << "  Output is RM, using C^T = B^T * A^T" << std::endl;
			std::cout << "  op_b=" << (op_b == CUBLAS_OP_N ? "N" : "T") << ", op_a=" << (op_a == CUBLAS_OP_N ? "N" : "T") << std::endl;
			std::cout << "  cuBLAS call: gemm(op_b, op_a, n=" << n << ", m=" << m << ", k=" << k << ")" << std::endl;
			std::cout << "  ldb=" << B.stride() << ", lda=" << A.stride() << ", ldc=" << C.stride() << std::endl;
			
			// Print input matrix samples
			print_matrix_sample("  Input A", A.data(), m, k, A.stride(), LA, stream);
			print_matrix_sample("  Input B", B.data(), k, n, B.stride(), LB, stream);
		}
		
		// Swap the operations to match the swapped matrices
		CUBLAS_CHECK_THROW(cublasGemmEx(
			cublas_handle(),
			op_b, op_a,
			n, m, k,
			&alpha,
			B.data(), cuda_data_type, B.stride(),
			A.data(), cuda_data_type, A.stride(),
			&beta,
			C.data(), cuda_data_type, C.stride(),
			compute_type,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP
		));
		
		if (mixed_call_count <= 3) {
			print_matrix_sample("  Output C", C.data(), m, n, C.stride(), LC, stream);
		}
	} else {
		// Output is CM: use standard approach
		cublasOperation_t op_a = LA == RM ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t op_b = LB == RM ? CUBLAS_OP_T : CUBLAS_OP_N;
		
		if (mixed_call_count <= 3) {
			std::cout << "  Output is CM, using standard approach" << std::endl;
			std::cout << "  op_a=" << (op_a == CUBLAS_OP_N ? "N" : "T") << ", op_b=" << (op_b == CUBLAS_OP_N ? "N" : "T") << std::endl;
			std::cout << "  cuBLAS call: gemm(op_a, op_b, m=" << m << ", n=" << n << ", k=" << k << ")" << std::endl;
		}
		
		CUBLAS_CHECK_THROW(cublasGemmEx(
			cublas_handle(),
			op_a, op_b,
			m, n, k,
			&alpha,
			A.data(), cuda_data_type, A.stride(),
			B.data(), cuda_data_type, B.stride(),
			&beta,
			C.data(), cuda_data_type, C.stride(),
			compute_type,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP
		));
	}
}


// Base version: C and D must have the same layout (non-const C)
template <typename T, MatrixLayout LA, MatrixLayout LB, MatrixLayout LC>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrix<T, LA>& A, const GPUMatrix<T, LB>& B, GPUMatrix<T, LC>& C, const GPUMatrix<T, LC>& D, int split_k_slices = 1, float beta = 0.0f) {
	static bool first_call = true;
	static int call_count = 0;
	call_count++;
	
	if (first_call) {
		std::cout << "[DEBUG fc_multiply_split_k] A: " << A.m() << "x" << A.n() << " layout=" << (LA == CM ? "CM" : "RM") << " stride=" << A.stride() << std::endl;
		std::cout << "[DEBUG fc_multiply_split_k] B: " << B.m() << "x" << B.n() << " layout=" << (LB == CM ? "CM" : "RM") << " stride=" << B.stride() << std::endl;
		std::cout << "[DEBUG fc_multiply_split_k] C: " << C.m() << "x" << C.n() << " layout=" << (LC == CM ? "CM" : "RM") << " stride=" << C.stride() << std::endl;
		std::cout << "[DEBUG fc_multiply_split_k] split_k_slices=" << split_k_slices << " beta=" << beta << std::endl;
		first_call = false;
	}
	
	// Print first few calls for debugging
	if (call_count <= 5) {
		std::cout << "\n[fc_multiply_split_k Call #" << call_count << "]" << std::endl;
		std::cout << "  A: " << A.m() << "x" << A.n() << " (" << (LA == CM ? "CM" : "RM") << ", stride=" << A.stride() << ")" << std::endl;
		std::cout << "  B: " << B.m() << "x" << B.n() << " (" << (LB == CM ? "CM" : "RM") << ", stride=" << B.stride() << ")" << std::endl;
		std::cout << "  C: " << C.m() << "x" << C.n() << " (" << (LC == CM ? "CM" : "RM") << ", stride=" << C.stride() << ")" << std::endl;
		std::cout << "  split_k=" << split_k_slices << ", beta=" << beta << std::endl;
	}
	
	if (C.data() != D.data()) {
		throw std::runtime_error("fc_multiply_split_k with cuBLAS requires C and D to be the same matrix.");
	}

	if (split_k_slices == 1) {
		cublas_gemm(stream, A, B, C, 1.0f, beta);
		return;
	}

	const int k = A.n();
	if (k % split_k_slices != 0) {
		throw std::runtime_error("split_k_slices must evenly divide k");
	}
	const int k_slice = k / split_k_slices;

	for (int i = 0; i < split_k_slices; ++i) {
		float current_beta = (i == 0) ? beta : 1.0f;

		const GPUMatrix<T, LA> A_slice = A.slice_cols(i * k_slice, k_slice);
		const GPUMatrix<T, LB> B_slice = B.slice_rows(i * k_slice, k_slice);

		cublas_gemm(stream, A_slice, B_slice, C, 1.0f, current_beta);
	}
}

// Overload for const C and const D with same layout
// This handles calls from const GPUMatrixDynamic methods like .cm() and .rm()
template <typename T, MatrixLayout LA, MatrixLayout LB, MatrixLayout LC>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrix<T, LA>& A, const GPUMatrix<T, LB>& B, const GPUMatrix<T, LC>& C, const GPUMatrix<T, LC>& D, int split_k_slices = 1, float beta = 0.0f) {
	static bool first_call = true;
	if (first_call) {
		std::cout << "[DEBUG fc_multiply_split_k CONST] A: " << A.m() << "x" << A.n() << " layout=" << (LA == CM ? "CM" : "RM") << " stride=" << A.stride() << std::endl;
		std::cout << "[DEBUG fc_multiply_split_k CONST] B: " << B.m() << "x" << B.n() << " layout=" << (LB == CM ? "CM" : "RM") << " stride=" << B.stride() << std::endl;
		std::cout << "[DEBUG fc_multiply_split_k CONST] C: " << C.m() << "x" << C.n() << " layout=" << (LC == CM ? "CM" : "RM") << " stride=" << C.stride() << std::endl;
		std::cout << "[DEBUG fc_multiply_split_k CONST] split_k_slices=" << split_k_slices << " beta=" << beta << std::endl;
		first_call = false;
	}
	
	if (C.data() != D.data()) {
		throw std::runtime_error("fc_multiply_split_k with cuBLAS requires C and D to be the same matrix.");
	}

	// Cast away constness for C since it's used as output
	GPUMatrix<T, LC>& C_mutable = const_cast<GPUMatrix<T, LC>&>(C);

	if (split_k_slices == 1) {
		cublas_gemm(stream, A, B, C_mutable, 1.0f, beta);
		return;
	}

	const int k = A.n();
	if (k % split_k_slices != 0) {
		throw std::runtime_error("split_k_slices must evenly divide k");
	}
	const int k_slice = k / split_k_slices;

	for (int i = 0; i < split_k_slices; ++i) {
		float current_beta = (i == 0) ? beta : 1.0f;

		const GPUMatrix<T, LA> A_slice = A.slice_cols(i * k_slice, k_slice);
		const GPUMatrix<T, LB> B_slice = B.slice_rows(i * k_slice, k_slice);

		cublas_gemm(stream, A_slice, B_slice, C_mutable, 1.0f, current_beta);
	}
}

// Overloads for GPUMatrixDynamic
template <typename T, MatrixLayout LA, MatrixLayout LB, typename TC, typename TD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrix<T, LA>& A, const GPUMatrix<T, LB>& B, GPUMatrixDynamic<TC>& C, const GPUMatrixDynamic<TD>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (C.layout() != D.layout()) {
		throw std::runtime_error{"fc_multiply_split_k: Layout of GPUMatrixDynamic C and D must be equal"};
	}

	if (C.layout() == CM) {
		fc_multiply_split_k(stream, A, B, C.cm(), D.cm(), split_k_slices, beta);
	} else {
		fc_multiply_split_k(stream, A, B, C.rm(), D.rm(), split_k_slices, beta);
	}
}

template <typename T, MatrixLayout LA, typename TB, typename TC, typename TD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrix<T, LA>& A, const GPUMatrixDynamic<TB>& B, GPUMatrixDynamic<TC>& C, const GPUMatrixDynamic<TD>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (B.layout() == CM) {
		fc_multiply_split_k(stream, A, B.cm(), C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k(stream, A, B.rm(), C, D, split_k_slices, beta);
	}
}

template <typename TA, typename TB, typename TC, typename TD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrixDynamic<TA>& A, const GPUMatrixDynamic<TB>& B, GPUMatrixDynamic<TC>& C, const GPUMatrixDynamic<TD>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (A.layout() == CM) {
		fc_multiply_split_k(stream, A.cm(), B, C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k(stream, A.rm(), B, C, D, split_k_slices, beta);
	}
}

template <typename TA, typename TB, typename TD>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrixDynamic<TA>& A, const GPUMatrixDynamic<TB>& B, GPUMatrixDynamic<TD>& D, int split_k_slices, float beta) {
	fc_multiply_split_k(stream, A, B, D, D, split_k_slices, beta);
}

// Additional overloads for mixed GPUMatrix/GPUMatrixDynamic with 4 matrix parameters
template <typename TA, typename T, MatrixLayout LB, MatrixLayout LC>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrixDynamic<TA>& A, const GPUMatrix<T, LB>& B, GPUMatrix<T, LC>& C, const GPUMatrix<T, LC>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (A.layout() == CM) {
		fc_multiply_split_k(stream, A.cm(), B, C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k(stream, A.rm(), B, C, D, split_k_slices, beta);
	}
}

template <typename T, MatrixLayout LA, typename TB, MatrixLayout LC>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrix<T, LA>& A, const GPUMatrixDynamic<TB>& B, GPUMatrix<T, LC>& C, const GPUMatrix<T, LC>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (B.layout() == CM) {
		fc_multiply_split_k(stream, A, B.cm(), C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k(stream, A, B.rm(), C, D, split_k_slices, beta);
	}
}

// Base version: C and D can have different layouts
template <typename T, MatrixLayout LA, MatrixLayout LB, MatrixLayout LC, MatrixLayout LD>
void fc_multiply(
	cudaStream_t stream,
	const GPUMatrix<T, LA>& A,
	const GPUMatrix<T, LB>& B,
	const GPUMatrix<T, LC>& C,
	GPUMatrix<T, LD>& D,
	Activation activation = Activation::None,
	bool transfer = false,
	bool sum_source = false
) {
	static int fc_multiply_call_count = 0;
	fc_multiply_call_count++;
	
	if (fc_multiply_call_count <= 3) {
		std::cout << "\n[fc_multiply Call #" << fc_multiply_call_count << "]" << std::endl;
		std::cout << "  A: " << A.m() << "x" << A.n() << " (" << (LA == CM ? "CM" : "RM") << ")" << std::endl;
		std::cout << "  B: " << B.m() << "x" << B.n() << " (" << (LB == CM ? "CM" : "RM") << ")" << std::endl;
		std::cout << "  C: " << C.m() << "x" << C.n() << " (" << (LC == CM ? "CM" : "RM") << ")" << std::endl;
		std::cout << "  D: " << D.m() << "x" << D.n() << " (" << (LD == CM ? "CM" : "RM") << ")" << std::endl;
		std::cout << "  activation=" << to_string(activation) << std::endl;
		std::cout << "  transfer=" << transfer << ", sum_source=" << sum_source << std::endl;
	}
	
	// cuBLAS does not support activation fusion or transfer operations
	if (transfer) {
		throw std::runtime_error("cuBLAS fc_multiply does not support transfer=true. This requires activation backward with forward values.");
	}

	float beta = sum_source ? 1.0f : 0.0f;

	// Handle C != D case
	if (C.data() != D.data()) {
		if (sum_source) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(D.data(), C.data(), C.n_bytes(), cudaMemcpyDeviceToDevice, stream));
		}
	}

	// Perform matrix multiplication
	cublas_gemm(stream, A, B, D, 1.0f, beta);

	// Apply activation function if needed (not fused, separate kernel)
	if (activation != Activation::None) {
		if (fc_multiply_call_count <= 3) {
			std::cout << "  Applying activation: " << to_string(activation) << std::endl;
			print_matrix_sample("  Before activation", D.data(), D.m(), D.n(), D.stride(), LD, stream);
		}
		
		const uint32_t num_elements = D.m() * D.n();
		constexpr uint32_t N_THREADS = 128;
		const uint32_t n_blocks = (num_elements + N_THREADS - 1) / N_THREADS;
		
		kernel_activation<T, 1><<<n_blocks, N_THREADS, 0, stream>>>(
			num_elements, activation, D.data(), D.data()
		);
		
		if (fc_multiply_call_count <= 3) {
			print_matrix_sample("  After activation", D.data(), D.m(), D.n(), D.stride(), LD, stream);
		}
	}
}

// Overload for when C and D are the same matrix (3-argument version)
template <typename T, MatrixLayout LA, MatrixLayout LB, MatrixLayout LC>
void fc_multiply(
	cudaStream_t stream,
	const GPUMatrix<T, LA>& A,
	const GPUMatrix<T, LB>& B,
	GPUMatrix<T, LC>& D,
	Activation activation = Activation::None
) {
	fc_multiply(stream, A, B, D, D, activation, false, false);
}

// Overload for 4 GPUMatrix parameters with same layout (const C, const D)
// This handles calls from const GPUMatrixDynamic methods like .cm() and .rm()
template <typename T, MatrixLayout LA, MatrixLayout LB, MatrixLayout LCD>
void fc_multiply(
	cudaStream_t stream,
	const GPUMatrix<T, LA>& A,
	const GPUMatrix<T, LB>& B,
	const GPUMatrix<T, LCD>& C,
	const GPUMatrix<T, LCD>& D,
	Activation activation = Activation::None,
	bool transfer = false,
	bool sum_source = false
) {
	// cuBLAS does not support activation fusion or transfer operations
	if (transfer) {
		throw std::runtime_error("cuBLAS fc_multiply does not support transfer=true. This requires activation backward with forward values.");
	}

	// Since both C and D are const, we need to cast away constness for the output D
	// This is safe because D is logically the output parameter
	GPUMatrix<T, LCD>& D_mutable = const_cast<GPUMatrix<T, LCD>&>(D);

	float beta = sum_source ? 1.0f : 0.0f;

	// Handle C != D case
	if (C.data() != D.data()) {
		if (sum_source) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(D_mutable.data(), C.data(), C.n_bytes(), cudaMemcpyDeviceToDevice, stream));
		}
	}

	// Perform matrix multiplication
	cublas_gemm(stream, A, B, D_mutable, 1.0f, beta);

	// Apply activation function if needed (not fused, separate kernel)
	if (activation != Activation::None) {
		const uint32_t num_elements = D.m() * D.n();
		constexpr uint32_t N_THREADS = 128;
		const uint32_t n_blocks = (num_elements + N_THREADS - 1) / N_THREADS;
		
		kernel_activation<T, 1><<<n_blocks, N_THREADS, 0, stream>>>(
			num_elements, activation, D_mutable.data(), D_mutable.data()
		);
	}
}

// Overload for 4 GPUMatrix parameters with same layout (non-const C)
// This handles calls where C and D are non-const and have the same layout
template <typename T, MatrixLayout LA, MatrixLayout LB, MatrixLayout LCD>
void fc_multiply(
	cudaStream_t stream,
	const GPUMatrix<T, LA>& A,
	const GPUMatrix<T, LB>& B,
	GPUMatrix<T, LCD>& C,
	GPUMatrix<T, LCD>& D,
	Activation activation = Activation::None,
	bool transfer = false,
	bool sum_source = false
) {
	// cuBLAS does not support activation fusion or transfer operations
	if (transfer) {
		throw std::runtime_error("cuBLAS fc_multiply does not support transfer=true. This requires activation backward with forward values.");
	}

	float beta = sum_source ? 1.0f : 0.0f;

	// Handle C != D case
	if (C.data() != D.data()) {
		if (sum_source) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(D.data(), C.data(), C.n_bytes(), cudaMemcpyDeviceToDevice, stream));
		}
	}

	// Perform matrix multiplication
	cublas_gemm(stream, A, B, D, 1.0f, beta);

	// Apply activation function if needed (not fused, separate kernel)
	if (activation != Activation::None) {
		const uint32_t num_elements = D.m() * D.n();
		constexpr uint32_t N_THREADS = 128;
		const uint32_t n_blocks = (num_elements + N_THREADS - 1) / N_THREADS;
		
		kernel_activation<T, 1><<<n_blocks, N_THREADS, 0, stream>>>(
			num_elements, activation, D.data(), D.data()
		);
	}
}

// Overloads for GPUMatrixDynamic
template <typename T, MatrixLayout LA, MatrixLayout LB, typename TC, typename TD>
void fc_multiply(
	cudaStream_t stream,
	const GPUMatrix<T, LA>& A,
	const GPUMatrix<T, LB>& B,
	const GPUMatrixDynamic<TC>& C,
	GPUMatrixDynamic<TD>& D,
	Activation activation = Activation::None,
	bool transfer = false,
	bool sum_source = false
) {
	if (C.layout() != D.layout()) {
		throw std::runtime_error{"fc_multiply: Layout of GPUMatrixDynamic C and D must be equal"};
	}

	if (C.layout() == CM) {
		fc_multiply(stream, A, B, C.cm(), D.cm(), activation, transfer, sum_source);
	} else {
		fc_multiply(stream, A, B, C.rm(), D.rm(), activation, transfer, sum_source);
	}
}

template <typename T, MatrixLayout LA, typename TB, typename TC, typename TD>
void fc_multiply(
	cudaStream_t stream,
	const GPUMatrix<T, LA>& A,
	const GPUMatrixDynamic<TB>& B,
	GPUMatrixDynamic<TC>& C,
	GPUMatrixDynamic<TD>& D,
	Activation activation = Activation::None,
	bool transfer = false,
	bool sum_source = false
) {
	if (B.layout() == CM) {
		fc_multiply(stream, A, B.cm(), C, D, activation, transfer, sum_source);
	} else {
		fc_multiply(stream, A, B.rm(), C, D, activation, transfer, sum_source);
	}
}

template <typename T, MatrixLayout LA, typename TB, typename TD>
void fc_multiply(
	cudaStream_t stream,
	const GPUMatrix<T, LA>& A,
	const GPUMatrixDynamic<TB>& B,
	GPUMatrixDynamic<TD>& D,
	Activation activation = Activation::None
) {
	fc_multiply(stream, A, B, D, D, activation, false, false);
}


}

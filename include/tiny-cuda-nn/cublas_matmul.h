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

namespace tcnn {

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
	cublasComputeType_t compute_type = std::is_same<T, float>::value ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_16F;

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
		CUBLAS_GEMM_DEFAULT
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
	cublasComputeType_t compute_type = std::is_same<T, float>::value ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_16F;

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
		CUBLAS_GEMM_DEFAULT
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
	cublasComputeType_t compute_type = std::is_same<T, float>::value ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_16F;

	cublasOperation_t op_a = LA == RM ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t op_b = LB == RM ? CUBLAS_OP_T : CUBLAS_OP_N;

	// cuBLAS is column-major. We want to compute C_lc = op(A) * op(B)
	// If C is row-major, we compute C_cm^T = op(B) * op(A) instead.
	if (LC == RM) {
		// Swap A and B, and swap m and n
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
			CUBLAS_GEMM_DEFAULT
		));
	} else {
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
			CUBLAS_GEMM_DEFAULT
		));
	}
}


// Base version: C and D must have the same layout
template <typename T, MatrixLayout LA, MatrixLayout LB, MatrixLayout LC>
void fc_multiply_split_k(cudaStream_t stream, const GPUMatrix<T, LA>& A, const GPUMatrix<T, LB>& B, GPUMatrix<T, LC>& C, const GPUMatrix<T, LC>& D, int split_k_slices = 1, float beta = 0.0f) {
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

// Base version: all GPUMatrix with same layout for C and D
template <typename T, MatrixLayout LA, MatrixLayout LB, MatrixLayout LC>
void fc_multiply(
	cudaStream_t stream,
	const GPUMatrix<T, LA>& A,
	const GPUMatrix<T, LB>& B,
	const GPUMatrix<T, LC>& C,
	GPUMatrix<T, LC>& D,
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

// Additional overload for 4 GPUMatrix parameters where C and D have same layout
// This is needed to match calls from GPUMatrixDynamic overloads
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
	// Ensure C and D are the same matrix
	if (C.data() != D.data()) {
		throw std::runtime_error("cuBLAS fc_multiply requires C and D to be the same matrix.");
	}

	// cuBLAS does not support activation fusion or transfer operations
	if (transfer) {
		throw std::runtime_error("cuBLAS fc_multiply does not support transfer=true. This requires activation backward with forward values.");
	}

	float beta = sum_source ? 1.0f : 0.0f;

	// Perform matrix multiplication using the mixed-layout cublas_gemm
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

// Additional overload to handle const C parameter (from const GPUMatrixDynamic)
template <typename T, MatrixLayout LA, MatrixLayout LB, MatrixLayout LCD>
void fc_multiply(
	cudaStream_t stream,
	const GPUMatrix<T, LA>& A,
	const GPUMatrix<T, LB>& B,
	const GPUMatrix<T, LCD>& C,
	GPUMatrix<T, LCD>& D,
	Activation activation = Activation::None,
	bool transfer = false,
	bool sum_source = false
) {
	// Ensure C and D are the same matrix
	if (C.data() != D.data()) {
		throw std::runtime_error("cuBLAS fc_multiply requires C and D to be the same matrix.");
	}

	// cuBLAS does not support activation fusion or transfer operations
	if (transfer) {
		throw std::runtime_error("cuBLAS fc_multiply does not support transfer=true. This requires activation backward with forward values.");
	}

	float beta = sum_source ? 1.0f : 0.0f;

	// Perform matrix multiplication using the mixed-layout cublas_gemm
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

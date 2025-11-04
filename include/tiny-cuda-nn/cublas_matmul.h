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

#include <hipblas.h>

#include <type_traits>

namespace tcnn {

#define CUBLAS_CHECK_THROW(x)                                                                                        \
	do {                                                                                                                   \
		hipblasStatus_t _result = x;                                                                                    \
		if (_result != HIPBLAS_STATUS_SUCCESS)                                                                            \
			throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + std::to_string(_result)); \
	} while(0)

// Debug logging control
#define ENABLE_HIPBLAS_DEBUG_LOGGING 1
#define MAX_DEBUG_GEMM_CALLS 100  // Only log first 100 GEMM calls to reduce verbosity

#if ENABLE_HIPBLAS_DEBUG_LOGGING
static int g_gemm_call_counter = 0;
static int g_fc_multiply_call_counter = 0;
static int g_fc_multiply_split_k_call_counter = 0;

// Helper to check if we should log this call (sample every Nth call after initial burst)
inline bool should_log_gemm_call(int call_num) {
	if (call_num <= 10) return true;  // Log first 20 calls
	if (call_num >= 190 && call_num <= 200) return true;  // Log around where NaN appears (GEMM #195)
	// if (call_num % 100 == 0) return true;  // Sample every 100th call
	return false;
}

// Helper function to sample matrix values for debugging
template <typename T>
void sample_matrix_values(hipStream_t stream, const T* data, uint32_t m, uint32_t n, const char* name) {
	const int num_samples = std::min(5, (int)(m * n));
	std::vector<T> samples(num_samples);
	
	// Sample first few elements
	CUDA_CHECK_THROW(hipMemcpyAsync(samples.data(), data, num_samples * sizeof(T), hipMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(hipStreamSynchronize(stream));
	
	printf("  %s[%dx%d] samples: ", name, m, n);
	for (int i = 0; i < num_samples; ++i) {
		if (std::is_same<T, __half>::value) {
			printf("%.4f ", (float)samples[i]);
		} else {
			printf("%.4f ", (float)samples[i]);
		}
	}
	printf("\n");
}
#endif

inline hipblasHandle_t& cublas_handle() {
	static hipblasHandle_t handle;
	static bool initialized = false;
	if (!initialized) {
		CUBLAS_CHECK_THROW(hipblasCreate(&handle));
		initialized = true;
	}
	return handle;
}

template <typename T>
void cublas_gemm(
	hipStream_t stream,
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

	hipblasSetStream(cublas_handle(), stream);

	hipDataType cuda_data_type = std::is_same<T, float>::value ? HIPBLAS_R_32F : HIPBLAS_R_16F;
	// CRITICAL: Always use FP32 compute for numerical stability, even with FP16 data
	// This matches NVIDIA's CUBLAS_COMPUTE_32F_FAST_16F behavior
	// FP16 compute causes accumulation overflow and NaN
	hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F;
	hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;
	
#if ENABLE_HIPBLAS_DEBUG_LOGGING
	++g_gemm_call_counter;
	if (should_log_gemm_call(g_gemm_call_counter)) {
		printf("\n[hipBLAS GEMM #%d] RM×RM→RM\n", g_gemm_call_counter);
		printf("  Dimensions: A[%d,%d] × B[%d,%d] → C[%d,%d]\n", m, k, k, n, m, n);
		printf("  Strides: A=%d, B=%d, C=%d\n", A.stride(), B.stride(), C.stride());
		printf("  alpha=%.4f, beta=%.4f\n", alpha, beta);
		printf("  data_type=%s, compute_type=FP32\n", 
			   cuda_data_type == HIPBLAS_R_32F ? "FP32" : "FP16");
		sample_matrix_values(stream, A.data(), m, k, "A");
		sample_matrix_values(stream, B.data(), k, n, "B");
	}
#endif
	
	// Since all matrices are row-major, we can use the identity (A*B)^T = B^T * A^T
	// and compute C_cm = B_cm * A_cm, which is equivalent to C_rm = A_rm * B_rm
	// but with swapped arguments.
	
	// With FP32 compute, always use float* for alpha/beta
	CUBLAS_CHECK_THROW(hipblasGemmEx(
		cublas_handle(),
		HIPBLAS_OP_N, HIPBLAS_OP_N,
		n, m, k,
		&alpha,
		B.data(), cuda_data_type, B.stride(),
		A.data(), cuda_data_type, A.stride(),
		&beta,
		C.data(), cuda_data_type, C.stride(),
		compute_type,
		algo
	));

#if ENABLE_HIPBLAS_DEBUG_LOGGING
	if (should_log_gemm_call(g_gemm_call_counter)) {
		sample_matrix_values(stream, C.data(), m, n, "C_output");
		// Check for NaN in output
		std::vector<T> check_sample(1);
		CUDA_CHECK_THROW(hipMemcpyAsync(check_sample.data(), C.data(), sizeof(T), hipMemcpyDeviceToHost, stream));
		CUDA_CHECK_THROW(hipStreamSynchronize(stream));
		if (std::isnan((float)check_sample[0])) {
			printf("  ⚠️  WARNING: NaN detected in output at GEMM #%d!\n", g_gemm_call_counter);
			printf("  This is the FIRST occurrence of NaN in matrix multiplication output.\n");
		}
	}
#endif
}

template <typename T>
void cublas_gemm(
	hipStream_t stream,
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

	hipblasSetStream(cublas_handle(), stream);

	hipDataType cuda_data_type = std::is_same<T, float>::value ? HIPBLAS_R_32F : HIPBLAS_R_16F;
	// CRITICAL: Always use FP32 compute for numerical stability
	hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F;
	hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;

#if ENABLE_HIPBLAS_DEBUG_LOGGING
	++g_gemm_call_counter;
	if (should_log_gemm_call(g_gemm_call_counter)) {
		printf("\n[hipBLAS GEMM #%d] CM×CM→CM\n", g_gemm_call_counter);
		printf("  Dimensions: A[%d,%d] × B[%d,%d] → C[%d,%d]\n", m, k, k, n, m, n);
		printf("  Strides: A=%d, B=%d, C=%d\n", A.stride(), B.stride(), C.stride());
		printf("  alpha=%.4f, beta=%.4f\n", alpha, beta);
		printf("  data_type=%s, compute_type=FP32\n",
			   cuda_data_type == HIPBLAS_R_32F ? "FP32" : "FP16");
		sample_matrix_values(stream, A.data(), m, k, "A");
		sample_matrix_values(stream, B.data(), k, n, "B");
	}
#endif

	// With FP32 compute, always use float* for alpha/beta
	CUBLAS_CHECK_THROW(hipblasGemmEx(
		cublas_handle(),
		HIPBLAS_OP_N, HIPBLAS_OP_N,
		m, n, k,
		&alpha,
		A.data(), cuda_data_type, A.stride(),
		B.data(), cuda_data_type, B.stride(),
		&beta,
		C.data(), cuda_data_type, C.stride(),
		compute_type,
		algo
	));

#if ENABLE_HIPBLAS_DEBUG_LOGGING
	if (should_log_gemm_call(g_gemm_call_counter)) {
		sample_matrix_values(stream, C.data(), m, n, "C_output");
	}
#endif
}

// Fallback for mixed layouts (less efficient due to potential transposes)
template <typename T, MatrixLayout LA, MatrixLayout LB, MatrixLayout LC>
void cublas_gemm(
	hipStream_t stream,
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

	hipblasSetStream(cublas_handle(), stream);

	hipDataType cuda_data_type = std::is_same<T, float>::value ? HIPBLAS_R_32F : HIPBLAS_R_16F;
	// CRITICAL: Always use FP32 compute for numerical stability
	hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F;
	hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;
	
#if ENABLE_HIPBLAS_DEBUG_LOGGING
	++g_gemm_call_counter;
	bool should_log = should_log_gemm_call(g_gemm_call_counter);
	if (should_log) {
		const char* layout_a = LA == RM ? "RM" : "CM";
		const char* layout_b = LB == RM ? "RM" : "CM";
		const char* layout_c = LC == RM ? "RM" : "CM";
		printf("\n[hipBLAS GEMM #%d] MIXED LAYOUT: %s×%s→%s\n", g_gemm_call_counter, layout_a, layout_b, layout_c);
		printf("  Dimensions: A[%d,%d] × B[%d,%d] → C[%d,%d]\n", m, k, k, n, m, n);
		printf("  Strides: A=%d, B=%d, C=%d\n", A.stride(), B.stride(), C.stride());
		printf("  alpha=%.4f, beta=%.4f\n", alpha, beta);
		printf("  data_type=%s, compute_type=FP32\n",
			   cuda_data_type == HIPBLAS_R_32F ? "FP32" : "FP16");
		
		// Sample C_before if beta != 0 (accumulation mode)
		if (beta != 0.0f) {
			sample_matrix_values(stream, C.data(), m, n, "C_before");
		}
	}
#endif
	
	// For mixed layouts, we need to carefully handle the transpose operations
	// hipBLAS is column-major, so we interpret RM matrices as transposed CM matrices
	
	if (LC == RM) {
		// Output is RM: C_rm (m×n) = A (m×k) * B (k×n)
		// Use identity: C_rm = A * B ⟺ C_cm^T = B^T * A^T
		// 
		// Strategy: Compute C^T using hipBLAS (which outputs CM)
		// hipBLAS computes: C_cm^T (n×m) = first_matrix * second_matrix
		// We want: C_cm^T (n×m) = B^T (n×k) * A^T (k×m)
		// 
		// Key: RM matrices are already "transposed" when viewed as CM
		// - RM (m×n, stride=n) viewed as CM is (n×m, stride=n)
		// - So we use HIPBLAS_OP_N (no additional transpose needed)
		// - CM matrices need HIPBLAS_OP_T to transpose them
		hipblasOperation_t op_a = LA == RM ? HIPBLAS_OP_N : HIPBLAS_OP_T;
		hipblasOperation_t op_b = LB == RM ? HIPBLAS_OP_N : HIPBLAS_OP_T;
		
#if ENABLE_HIPBLAS_DEBUG_LOGGING
		if (should_log) {
			printf("  Output is RM: using transposed computation\n");
			printf("  op_b=%s, op_a=%s (swapped order)\n", 
				   op_b == HIPBLAS_OP_N ? "N" : "T",
				   op_a == HIPBLAS_OP_N ? "N" : "T");
			printf("  Using stride() directly: A.stride()=%d, B.stride()=%d, C.stride()=%d\n", 
				   A.stride(), B.stride(), C.stride());
			sample_matrix_values(stream, A.data(), m, k, "A");
			sample_matrix_values(stream, B.data(), k, n, "B");
		}
#endif
		
		// Swap the operations to match the swapped matrices
		// With FP32 compute, always use float* for alpha/beta
		CUBLAS_CHECK_THROW(hipblasGemmEx(
			cublas_handle(),
			op_b, op_a,
			n, m, k,
			&alpha,
			B.data(), cuda_data_type, B.stride(),
			A.data(), cuda_data_type, A.stride(),
			&beta,
			C.data(), cuda_data_type, C.stride(),
			compute_type,
			algo
		));
	} else {
		// Output is CM: use standard approach
		hipblasOperation_t op_a = LA == RM ? HIPBLAS_OP_T : HIPBLAS_OP_N;
		hipblasOperation_t op_b = LB == RM ? HIPBLAS_OP_T : HIPBLAS_OP_N;
		
#if ENABLE_HIPBLAS_DEBUG_LOGGING
		if (should_log) {
			printf("  Output is CM: using standard computation\n");
			printf("  op_a=%s, op_b=%s\n", 
				   op_a == HIPBLAS_OP_N ? "N" : "T",
				   op_b == HIPBLAS_OP_N ? "N" : "T");
			printf("  Using stride() directly: A.stride()=%d, B.stride()=%d, C.stride()=%d\n",
				   A.stride(), B.stride(), C.stride());
			sample_matrix_values(stream, A.data(), m, k, "A");
			sample_matrix_values(stream, B.data(), k, n, "B");
		}
#endif
		
		// With FP32 compute, always use float* for alpha/beta
		CUBLAS_CHECK_THROW(hipblasGemmEx(
			cublas_handle(),
			op_a, op_b,
			m, n, k,
			&alpha,
			A.data(), cuda_data_type, A.stride(),
			B.data(), cuda_data_type, B.stride(),
			&beta,
			C.data(), cuda_data_type, C.stride(),
			compute_type,
			algo
		));
	}

#if ENABLE_HIPBLAS_DEBUG_LOGGING
	if (should_log) {
		sample_matrix_values(stream, C.data(), m, n, "C_output");
		// Check for NaN in output
		std::vector<T> check_sample(1);
		CUDA_CHECK_THROW(hipMemcpyAsync(check_sample.data(), C.data(), sizeof(T), hipMemcpyDeviceToHost, stream));
		CUDA_CHECK_THROW(hipStreamSynchronize(stream));
		if (std::isnan((float)check_sample[0])) {
			printf("  🔴 CRITICAL: NaN detected in MIXED LAYOUT output at GEMM #%d!\n", g_gemm_call_counter);
			printf("  Operation: %s×%s→%s with beta=%.4f\n", 
				   LA == RM ? "RM" : "CM", LB == RM ? "RM" : "CM", LC == RM ? "RM" : "CM", beta);
		}
	}
#endif
}


// Base version: C and D must have the same layout (non-const C)
template <typename T, MatrixLayout LA, MatrixLayout LB, MatrixLayout LC>
void fc_multiply_split_k(hipStream_t stream, const GPUMatrix<T, LA>& A, const GPUMatrix<T, LB>& B, GPUMatrix<T, LC>& C, const GPUMatrix<T, LC>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (C.data() != D.data()) {
		throw std::runtime_error("fc_multiply_split_k with cuBLAS requires C and D to be the same matrix.");
	}

#if ENABLE_HIPBLAS_DEBUG_LOGGING
	const char* layout_a = LA == RM ? "RM" : "CM";
	const char* layout_b = LB == RM ? "RM" : "CM";
	const char* layout_c = LC == RM ? "RM" : "CM";
	printf("\n[fc_multiply_split_k #%d] %s×%s→%s, split_k=%d, beta=%.4f\n", 
		   ++g_fc_multiply_split_k_call_counter, layout_a, layout_b, layout_c, split_k_slices, beta);
	printf("  A[%d,%d], B[%d,%d], C[%d,%d]\n", A.m(), A.n(), B.m(), B.n(), C.m(), C.n());
#endif

	if (split_k_slices == 1) {
		cublas_gemm(stream, A, B, C, 1.0f, beta);
		return;
	}

	const int k = A.n();
	if (k % split_k_slices != 0) {
		throw std::runtime_error("split_k_slices must evenly divide k");
	}
	const int k_slice = k / split_k_slices;

#if ENABLE_HIPBLAS_DEBUG_LOGGING
	printf("  Splitting K=%d into %d slices of size %d\n", k, split_k_slices, k_slice);
#endif

	for (int i = 0; i < split_k_slices; ++i) {
		float current_beta = (i == 0) ? beta : 1.0f;

		const GPUMatrix<T, LA> A_slice = A.slice_cols(i * k_slice, k_slice);
		const GPUMatrix<T, LB> B_slice = B.slice_rows(i * k_slice, k_slice);

#if ENABLE_HIPBLAS_DEBUG_LOGGING
		printf("  Slice %d/%d: A_slice[%d,%d], B_slice[%d,%d], beta=%.4f\n", 
			   i+1, split_k_slices, A_slice.m(), A_slice.n(), B_slice.m(), B_slice.n(), current_beta);
#endif

		cublas_gemm(stream, A_slice, B_slice, C, 1.0f, current_beta);
	}
}

// Overload for const C and const D with same layout
// This handles calls from const GPUMatrixDynamic methods like .cm() and .rm()
template <typename T, MatrixLayout LA, MatrixLayout LB, MatrixLayout LC>
void fc_multiply_split_k(hipStream_t stream, const GPUMatrix<T, LA>& A, const GPUMatrix<T, LB>& B, const GPUMatrix<T, LC>& C, const GPUMatrix<T, LC>& D, int split_k_slices = 1, float beta = 0.0f) {
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
void fc_multiply_split_k(hipStream_t stream, const GPUMatrix<T, LA>& A, const GPUMatrix<T, LB>& B, GPUMatrixDynamic<TC>& C, const GPUMatrixDynamic<TD>& D, int split_k_slices = 1, float beta = 0.0f) {
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
void fc_multiply_split_k(hipStream_t stream, const GPUMatrix<T, LA>& A, const GPUMatrixDynamic<TB>& B, GPUMatrixDynamic<TC>& C, const GPUMatrixDynamic<TD>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (B.layout() == CM) {
		fc_multiply_split_k(stream, A, B.cm(), C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k(stream, A, B.rm(), C, D, split_k_slices, beta);
	}
}

template <typename TA, typename TB, typename TC, typename TD>
void fc_multiply_split_k(hipStream_t stream, const GPUMatrixDynamic<TA>& A, const GPUMatrixDynamic<TB>& B, GPUMatrixDynamic<TC>& C, const GPUMatrixDynamic<TD>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (A.layout() == CM) {
		fc_multiply_split_k(stream, A.cm(), B, C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k(stream, A.rm(), B, C, D, split_k_slices, beta);
	}
}

template <typename TA, typename TB, typename TD>
void fc_multiply_split_k(hipStream_t stream, const GPUMatrixDynamic<TA>& A, const GPUMatrixDynamic<TB>& B, GPUMatrixDynamic<TD>& D, int split_k_slices, float beta) {
	fc_multiply_split_k(stream, A, B, D, D, split_k_slices, beta);
}

// Additional overloads for mixed GPUMatrix/GPUMatrixDynamic with 4 matrix parameters
template <typename TA, typename T, MatrixLayout LB, MatrixLayout LC>
void fc_multiply_split_k(hipStream_t stream, const GPUMatrixDynamic<TA>& A, const GPUMatrix<T, LB>& B, GPUMatrix<T, LC>& C, const GPUMatrix<T, LC>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (A.layout() == CM) {
		fc_multiply_split_k(stream, A.cm(), B, C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k(stream, A.rm(), B, C, D, split_k_slices, beta);
	}
}

template <typename T, MatrixLayout LA, typename TB, MatrixLayout LC>
void fc_multiply_split_k(hipStream_t stream, const GPUMatrix<T, LA>& A, const GPUMatrixDynamic<TB>& B, GPUMatrix<T, LC>& C, const GPUMatrix<T, LC>& D, int split_k_slices = 1, float beta = 0.0f) {
	if (B.layout() == CM) {
		fc_multiply_split_k(stream, A, B.cm(), C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k(stream, A, B.rm(), C, D, split_k_slices, beta);
	}
}

// Base version: C and D can have different layouts
template <typename T, MatrixLayout LA, MatrixLayout LB, MatrixLayout LC, MatrixLayout LD>
void fc_multiply(
	hipStream_t stream,
	const GPUMatrix<T, LA>& A,
	const GPUMatrix<T, LB>& B,
	const GPUMatrix<T, LC>& C,
	GPUMatrix<T, LD>& D,
	Activation activation = Activation::None,
	bool transfer = false,
	bool sum_source = false
) {
	// cuBLAS does not support activation fusion or transfer operations
	if (transfer) {
		throw std::runtime_error("cuBLAS fc_multiply does not support transfer=true. This requires activation backward with forward values.");
	}

#if ENABLE_HIPBLAS_DEBUG_LOGGING
	const char* layout_a = LA == RM ? "RM" : "CM";
	const char* layout_b = LB == RM ? "RM" : "CM";
	const char* layout_c = LC == RM ? "RM" : "CM";
	const char* layout_d = LD == RM ? "RM" : "CM";
	const char* act_name = activation == Activation::None ? "None" : 
	                       activation == Activation::ReLU ? "ReLU" : "Other";
	printf("\n[fc_multiply #%d] %s×%s, C:%s, D:%s, act=%s, sum_source=%d\n", 
		   ++g_fc_multiply_call_counter, layout_a, layout_b, layout_c, layout_d, act_name, sum_source);
	printf("  A[%d,%d], B[%d,%d], C[%d,%d], D[%d,%d]\n", 
		   A.m(), A.n(), B.m(), B.n(), C.m(), C.n(), D.m(), D.n());
#endif

	float beta = sum_source ? 1.0f : 0.0f;

	// Handle C != D case
	if (C.data() != D.data()) {
		if (sum_source) {
#if ENABLE_HIPBLAS_DEBUG_LOGGING
			printf("  Copying C to D (sum_source=true)\n");
#endif
			CUDA_CHECK_THROW(hipMemcpyAsync(D.data(), C.data(), C.n_bytes(), hipMemcpyDeviceToDevice, stream));
		}
	}

	// Perform matrix multiplication
	cublas_gemm(stream, A, B, D, 1.0f, beta);

	// Apply activation function if needed (not fused, separate kernel)
	if (activation != Activation::None) {
#if ENABLE_HIPBLAS_DEBUG_LOGGING
		printf("  Applying activation: %s\n", act_name);
#endif
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
	hipStream_t stream,
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
	hipStream_t stream,
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
			CUDA_CHECK_THROW(hipMemcpyAsync(D_mutable.data(), C.data(), C.n_bytes(), hipMemcpyDeviceToDevice, stream));
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
	hipStream_t stream,
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
			CUDA_CHECK_THROW(hipMemcpyAsync(D.data(), C.data(), C.n_bytes(), hipMemcpyDeviceToDevice, stream));
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
// This overload is for when B is GPUMatrixDynamic (more specific, comes first)
template <typename T, MatrixLayout LA, typename TB, typename TC, typename TD>
void fc_multiply(
	hipStream_t stream,
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
	hipStream_t stream,
	const GPUMatrix<T, LA>& A,
	const GPUMatrixDynamic<TB>& B,
	GPUMatrixDynamic<TD>& D,
	Activation activation = Activation::None
) {
	fc_multiply(stream, A, B, D, D, activation, false, false);
}


}

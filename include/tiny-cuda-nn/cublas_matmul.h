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
#include <cstdlib>
#include <cstdio>
#include <cctype>
#include <string>
#include <algorithm>

namespace tcnn {

#define CUBLAS_CHECK_THROW(x)                                                                                        \
	do {                                                                                                                   \
		hipblasStatus_t _result = x;                                                                                    \
		if (_result != HIPBLAS_STATUS_SUCCESS)                                                                            \
			throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + std::to_string(_result)); \
	} while(0)

// Debug logging control
#define ENABLE_HIPBLAS_DEBUG_LOGGING 0
#define MAX_DEBUG_GEMM_CALLS 100  // Only log first 100 GEMM calls to reduce verbosity

#if ENABLE_HIPBLAS_DEBUG_LOGGING
static int g_gemm_call_counter = 0;
static int g_fc_multiply_call_counter = 0;
static int g_fc_multiply_split_k_call_counter = 0;
#endif 

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

// Additional helper to sample stats (NaN/Inf/max) from device matrices
template <typename T>
void sample_matrix_stats(hipStream_t stream, const T* data, uint32_t m, uint32_t n, const char* name) {
	const uint32_t sample_count = std::min<uint32_t>(256, m * n);
	std::vector<T> buf(sample_count);

	CUDA_CHECK_THROW(hipMemcpyAsync(buf.data(), data, sample_count * sizeof(T), hipMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(hipStreamSynchronize(stream));

	size_t nan_count = 0, inf_count = 0;
	float max_abs = 0.0f;
	for (uint32_t i = 0; i < sample_count; ++i) {
		float v = (float)buf[i];
		if (std::isnan(v)) ++nan_count;
		else if (std::isinf(v)) ++inf_count;
		float a = std::fabs(v);
		if (a > max_abs) max_abs = a;
	}

	// Throttle anomaly logging: only print first few anomalies globally
	static int g_anomaly_logs = 0;
	constexpr int MAX_ANOMALY_LOGS = 10;

	if ((nan_count > 0 || inf_count > 0) && g_anomaly_logs < MAX_ANOMALY_LOGS) {
		++g_anomaly_logs;
		printf("  %s[%dx%d] stats: NaN=%zu Inf=%zu max=%.4f (sample=%u)\n", name, m, n, nan_count, inf_count, max_abs, sample_count);
		// Optional: after reaching the cap, print a single suppression notice
		if (g_anomaly_logs == MAX_ANOMALY_LOGS) {
			printf("  [gemm-stats] further anomaly logs suppressed (cap=%d)\n", MAX_ANOMALY_LOGS);
		}
	}
}

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

	hipblasDatatype_t cuda_data_type = std::is_same<T, float>::value ? HIPBLAS_R_32F : HIPBLAS_R_16F;
	// CRITICAL: Always use FP32 compute for numerical stability, even with FP16 data
	// This matches NVIDIA's CUBLAS_COMPUTE_32F_FAST_16F behavior
	// FP16 compute causes accumulation overflow and NaN
	hipblasDatatype_t compute_type = HIPBLAS_R_32F;
	hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;
	
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
	sample_matrix_stats(stream, C.data(), m, n, "C_stats_RM");
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

	hipblasDatatype_t cuda_data_type = std::is_same<T, float>::value ? HIPBLAS_R_32F : HIPBLAS_R_16F;
	// CRITICAL: Always use FP32 compute for numerical stability
	hipblasDatatype_t compute_type = HIPBLAS_R_32F;
	hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;

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
	sample_matrix_stats(stream, C.data(), m, n, "C_stats_CM");
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

	hipblasDatatype_t cuda_data_type = std::is_same<T, float>::value ? HIPBLAS_R_32F : HIPBLAS_R_16F;
	// CRITICAL: Always use FP32 compute for numerical stability
	hipblasDatatype_t compute_type = HIPBLAS_R_32F;
	hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;
	

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
#if ENABLE_HIPBLAS_DEBUG_LOGGING
	sample_matrix_stats(stream, C.data(), m, n, "C_stats_MIXED_RM");
#endif

#if ENABLE_HIPBLAS_DEBUG_LOGGING
	{
		// Quick check of first up-to-256 elements of C for NaN/Inf, but keep logging concise
		const uint32_t sample_count = std::min<uint32_t>(256, (uint32_t)(m * n));
		std::vector<T> cbuf(sample_count);
		hipError_t herr = hipMemcpyAsync(cbuf.data(), C.data(), sample_count * sizeof(T), hipMemcpyDeviceToHost, stream);
		hipStreamSynchronize(stream);
		if (herr == hipSuccess) {
			size_t nan_count = 0, inf_count = 0;
			// Collect only the first few anomalous values to print
			std::vector<uint32_t> anom_idx;
			std::vector<float> anom_val;
			anom_idx.reserve(5);
			anom_val.reserve(5);

			for (uint32_t i = 0; i < sample_count; ++i) {
				float v = (float)cbuf[i];
				if (std::isnan(v)) {
					++nan_count;
					if (anom_idx.size() < 5) { anom_idx.push_back(i); anom_val.push_back(v); }
				} else if (std::isinf(v)) {
					++inf_count;
					if (anom_idx.size() < 5) { anom_idx.push_back(i); anom_val.push_back(v); }
				}
			}

			if (nan_count > 0 || inf_count > 0) {
				const char* op_a_name = (LA == RM ? "N" : "T");
				const char* op_b_name = (LB == RM ? "N" : "T");
				const char* dtype_name = (cuda_data_type == HIPBLAS_R_32F ? "FP32" : "FP16");

				// Cap total anomaly logs for this path to avoid spam
				static int g_mixed_rm_anomaly_logs = 0;
				constexpr int MAX_MIXED_RM_ANOM_LOGS = 5;

				if (g_mixed_rm_anomaly_logs < MAX_MIXED_RM_ANOM_LOGS) {
					++g_mixed_rm_anomaly_logs;

					// Single summary line + at most one line of anomaly values
					printf("[v39 MIXED_RM GEMM ANOMALY %d/%d] m=%d n=%d k=%d, op_b=%s op_a=%s, lda/ldb/ldc=%d/%d/%d, alpha=%.6f beta=%.6f, dtype=%s compute=FP32 algo=%d, NaN=%zu Inf=%zu\n",
						g_mixed_rm_anomaly_logs, MAX_MIXED_RM_ANOM_LOGS,
						m, n, k, op_b_name, op_a_name,
						B.stride(), A.stride(), C.stride(),
						alpha, beta, dtype_name, (int)algo,
						nan_count, inf_count
					);

					// Sample first 8 elements of A, B, C
					std::vector<T> abuf(8), bbuf(8);
					hipMemcpyAsync(abuf.data(), A.data(), 8 * sizeof(T), hipMemcpyDeviceToHost, stream);
					hipMemcpyAsync(bbuf.data(), B.data(), 8 * sizeof(T), hipMemcpyDeviceToHost, stream);
					hipStreamSynchronize(stream);

					printf("  A[0:8]: ");
					for (int i = 0; i < 8; ++i) printf("%.4f ", (float)abuf[i]);
					printf("\n");

					printf("  B[0:8]: ");
					for (int i = 0; i < 8; ++i) printf("%.4f ", (float)bbuf[i]);
					printf("\n");

					printf("  C[0:8]: ");
					for (int i = 0; i < 8; ++i) printf("%.4f ", (float)cbuf[i]);
					printf("\n");

					if (!anom_idx.empty()) {
						printf("  C anomalies (first %zu): ", anom_idx.size());
						for (size_t j = 0; j < anom_idx.size(); ++j) {
							printf("(%u:%.4f) ", anom_idx[j], anom_val[j]);
						}
						printf("\n");
					}

					if (g_mixed_rm_anomaly_logs == MAX_MIXED_RM_ANOM_LOGS) {
						printf("  [MIXED_RM] further anomaly logs suppressed (cap=%d)\n", MAX_MIXED_RM_ANOM_LOGS);
					}
				}
			}
		} else {
			printf("[v39 MIXED_RM GEMM ANOMALY] hipMemcpyAsync(C sample) failed: %d\n", (int)herr);
		}
	}
#endif
	} else {
		// Output is CM: use standard approach
		hipblasOperation_t op_a = LA == RM ? HIPBLAS_OP_T : HIPBLAS_OP_N;
		hipblasOperation_t op_b = LB == RM ? HIPBLAS_OP_T : HIPBLAS_OP_N;
		
	
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
#if ENABLE_HIPBLAS_DEBUG_LOGGING
		sample_matrix_stats(stream, C.data(), m, n, "C_stats_MIXED_CM");
#endif
	}

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
#if ENABLE_HIPBLAS_DEBUG_LOGGING
	sample_matrix_stats(stream, D.data(), D.m(), D.n(), "D_after_gemm");
#endif

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
#if ENABLE_HIPBLAS_DEBUG_LOGGING
	sample_matrix_stats(stream, D_mutable.data(), D_mutable.m(), D_mutable.n(), "D_after_gemm");
#endif

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

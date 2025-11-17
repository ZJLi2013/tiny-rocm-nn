#include "hip/hip_runtime.h"
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

/** @file   fully_fused_mlp.cu
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Fully fused CUDA implementation of a multi-layer perceptron. Supports online training
 *          and simultaneous inference.
 */

#include <tiny-cuda-nn/networks/fully_fused_mlp.h>

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/cublas_matmul.h>
#include <tiny-cuda-nn/multi_stream.h>

#include <rocwmma/rocwmma.hpp>
#include <vector>
#include <cstdio>
#include <cmath>

namespace tcnn {

// ROCm: AMD GPU wave size (64 threads per wave, vs NVIDIA's 32 threads per warp)
constexpr uint32_t WAVE_SIZE = 64;

// v32: Global diagnostic counters for anomaly detection
__device__ uint32_t g_nan_detected = 0;
__device__ uint32_t g_inf_detected = 0;
__device__ uint32_t g_large_val_detected = 0;
__device__ uint32_t g_first_nan_step = 0xFFFFFFFF;
__device__ uint32_t g_current_step = 0;

void check_shmem_error(hipError_t error) {
	if (error != hipSuccess) {
		throw std::runtime_error{"FullyFusedMLP: insufficient shared memory available on the GPU. Reduce `n_neurons` to fit available shared memory."};
	}
}

__device__ void sh2gmem(__half* device_mem, const __half* __restrict__ shmem, int N)
{
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	if(tid < N){
		device_mem[tid] = shmem[tid];
	}
	__syncthreads();
}

// v32: Check shared memory for anomalies before computation
template <int WIDTH, int N_ITERS>
__device__ void check_shmem_input(const __half* __restrict__ act_shmem, const char* layer_name, uint32_t layer_idx) {
	// Only first thread of first block checks
	if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
		constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
		constexpr float THRESHOLD = 100.0f;
		
		bool has_nan = false;
		bool has_inf = false;
		float max_val = 0.0f;
		int nan_count = 0;
		
		// Sample first 256 elements for efficiency
		const uint32_t max_size = 16u * N_ITERS * (WIDTH + SKEW);
		const int SAMPLE_SIZE = (max_size < 256u) ? max_size : 256u;
		
		for (int i = 0; i < SAMPLE_SIZE; i++) {
			float val = __half2float(act_shmem[i]);
			if (isnan(val)) {
				has_nan = true;
				nan_count++;
			}
			if (isinf(val)) has_inf = true;
			max_val = fmaxf(max_val, fabsf(val));
		}
		
		// Only print if NaN or Inf detected (critical issues)
		if (has_nan || has_inf) {
			printf("[v32 ALERT] %s[%u]: NaN=%d(cnt=%d), Inf=%d, max=%.2f\n",
			       layer_name, layer_idx, has_nan, nan_count, has_inf, max_val);
			
			// Update global counters
			if (has_nan) {
				atomicAdd(&g_nan_detected, 1);
				atomicMin(&g_first_nan_step, g_current_step);
			}
			if (has_inf) atomicAdd(&g_inf_detected, 1);
		} else if (max_val > THRESHOLD) {
			// Only print large values if no NaN/Inf (less critical)
			// Uncomment below if you want to track large values
			// printf("[v32 INFO] %s[%u]: max=%.2f\n", layer_name, layer_idx, max_val);
			atomicAdd(&g_large_val_detected, 1);
		}
	}
}

template <int WIDTH, int N_ITERS, typename OUT_T, bool BACKWARD=false>
__device__ void threadblock_layer(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const OUT_T* __restrict__ activation_aux = nullptr) {
	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch.
	//           Can be forward activations or backward activations, depending on caller.
	// weights_this_layer points to the weight matrix of the current layer.
	// out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
	//                  Can be nullptr if nothing should be written.
	// activation_aux points to additional arguments that the activation function may depend on. Points to the hidden forward activations when computing backward activations.

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	using namespace rocwmma;

	// If we're performing the backward pass, weights must be loaded in transposed form, which
	// is achieved by interpreting the memory in row_major instead of col_major order.
	using weights_layout_t = typename std::conditional<BACKWARD, row_major, col_major>::type;

	// v19/v34: Use OUT_T for accumulator (same as CUDA)
	// This allows matching CUDA's behavior exactly
	using MatrixA = fragment<matrix_a, 16, 16, 16, __half, row_major>;
	using MatrixB = fragment<matrix_b, 16, 16, 16, __half, weights_layout_t>;
	using Accumulator = fragment<accumulator, 16, 16, 16, OUT_T>;

	MatrixA act_frag;
	MatrixB weights_frag[N_BLOCKS];
	Accumulator result_frag[N_ITERS];

	// Indices
	const uint32_t li = threadIdx.x; // index in wave ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("wave index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * WAVE_SIZE) / WIDTH;  // Updated for 64-thread waves

	const uint32_t weights_col = 16 * wi;

	__syncthreads();

	// v33: Check weights for NaN (only first thread of first block, forward pass only)
	if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && !BACKWARD) {
		bool has_nan = false;
		int nan_count = 0;
		const uint32_t max_check = (WIDTH * WIDTH < 256u) ? WIDTH * WIDTH : 256u;
		
		for (uint32_t i = 0; i < max_check; i++) {
			float val = __half2float(weights_this_layer[i]);
			if (isnan(val)) {
				has_nan = true;
				nan_count++;
			}
		}
		
		if (has_nan) {
			printf("[v33 WEIGHT] NaN in weights: cnt=%d\n", nan_count);
		}
	}

	// Load N_BLOCKS chunks of weights from global memory into registers.
	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N_BLOCKS; ++i) {
		if (BACKWARD) {
			// If we're performing the backward pass, additional index swizzling is needed to
			// load the weights in transposed form.
			load_matrix_sync(weights_frag[i], weights_this_layer + 16 * i * WIDTH + weights_col, WIDTH);
		} else {
			load_matrix_sync(weights_frag[i], weights_this_layer + 16 * i + weights_col * WIDTH, WIDTH);
		}
	}

	TCNN_PRAGMA_UNROLL
	for (int l = 0; l < N_ITERS; ++l) {
		fill_fragment(result_frag[l], 0.0f);

		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_BLOCKS; ++i) {
			// Load FP16 from shared memory
			load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * l) * (WIDTH + SKEW), WIDTH + SKEW);
			// v27: FP16×FP16 → FP32 accumulation
			mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
		}

		// v19/v34: Activation function using OUT_T (same as CUDA)
		if (BACKWARD) {
			// Load the temporary forward matrix for the relu transfer
			load_matrix_sync(act_frag, activation_aux + weights_col + l * 16 * WIDTH, WIDTH);
			warp_activation_backward<OUT_T>(activation, result_frag[l], act_frag, result_frag[l]);
		} else {
			warp_activation<OUT_T>(activation, result_frag[l], result_frag[l]);
		}
	}

	// v19: Minimal synchronization
	__syncthreads();

	// v19/v34: Store directly (no conversion needed when OUT_T = __half)
	TCNN_PRAGMA_UNROLL
	for (int l = 0; l < N_ITERS; ++l) {
		store_matrix_sync(act_shmem + weights_col + l * 16 * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, mem_row_major);
	}

	if (out_intermediate_threadblock_this_layer != nullptr) {
		__syncthreads();

		TCNN_PRAGMA_UNROLL
		for (int l = 0; l < N_ITERS; ++l) {
			*(int4*)&out_intermediate_threadblock_this_layer[lane_offset + (row + 16 * l) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * l) * (WIDTH + SKEW)];
		}
	}
}

template <int WIDTH, int N_ITERS>
__device__ void threadblock_load_input_static(__half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock) {
	// act_shmem will be filled by the thread block's chunk of input_threadblock

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in wave ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("wave index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * WAVE_SIZE) / WIDTH;  // Updated for 64-thread waves

	TCNN_PRAGMA_UNROLL
	for (int i = 0; i < N_ITERS; ++i) {
		*(int4*)&act_shmem[lane_offset + (row + 16 * i) * (WIDTH + SKEW)] = *(int4*)&input_threadblock[lane_offset + (row + 16 * i) * WIDTH];
	}
	__syncthreads(); 
}

template <int WIDTH, int N_ITERS, Activation ACTIVATION, typename OUTPUT_LAYOUT>
__global__ void kernel_mlp_fused_backward(
	const __half* __restrict__ dL_doutput,
	const __half* __restrict__ weights,
	__half* __restrict__ out_intermediate,
	const __half* __restrict__ forward,
	__half* __restrict__ dL_dinput,
	const __half* __restrict__ weights_first_layer,
	const uint32_t output_stride,
	const uint32_t batch_size,
	const uint32_t out_width,
	const uint32_t n_hidden_matmuls
) {
	// `dL_doutput` points to the input matrix of the backward pass, i.e. the loss gradients. Assumed to be 16 neurons wide.
	// `weights` points to the weight matrices (contiguous in memory).
	// `out_intermediate` points to the memory where backpropagated activation gradients should be written.
	// `forward` points to the memory where the intermediate activations of the forward pass are located. (needed for activation backprop)

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in wave ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("wave index")
	const uint32_t bi = blockIdx.x;  // block index

	// Shared memory contains the intermediate activations of blockDim.y*16 elements.
	// A skew is applied to the matrix storage to avoid bank conflicts.
	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * WAVE_SIZE) / WIDTH;  // Updated for 64-thread waves

	// Multipying one 16-row chunk of intermediate activations with the weight matrix requires all warps of the block.
	// Thus, each block computes exactly one 16-row chunk of the next layer's intermediate activations.
	const uint32_t elem_idx_base = 16 * bi * N_ITERS;
	const uint32_t elem_idx = elem_idx_base;

	const uint32_t weights_stride = WIDTH * WIDTH;
	const uint32_t layer_stride = WIDTH * batch_size;

	// Backprop through last layer
	if (out_width <= 16) {
		using namespace rocwmma;

		// Fragments in registers
		fragment<matrix_a, 16, 16, 16, __half, OUTPUT_LAYOUT> act_frag;
		fragment<matrix_b, 16, 16, 16, __half, row_major> weights_frag;
		// v19/v34: Use __half accumulator (same as CUDA)
		fragment<accumulator, 16, 16, 16, __half> result_frag[N_ITERS];

		// Load the relevant chunk of the last layer's weight matrix from global memory into registers
		const uint32_t weights_col = 16 * wi;

		load_matrix_sync(weights_frag, weights + weights_stride * n_hidden_matmuls + weights_col, WIDTH);

		TCNN_PRAGMA_UNROLL
		for (int l = 0; l < N_ITERS; ++l) {
			fill_fragment(result_frag[l], 0.0f);

			// Load a chunk of output gradients from shared memory and multiply with previously loaded weights
			if (std::is_same<OUTPUT_LAYOUT, row_major>::value) {
				load_matrix_sync(act_frag, dL_doutput + (elem_idx + 16 * l) * output_stride, output_stride);
			} else {
				load_matrix_sync(act_frag, dL_doutput + (elem_idx + 16 * l), output_stride);
			}

			// NOTE: activation transfer of the _output_ activation is expected to be done _prior_ to calling this kernel
			//       in a separate pass, because the tranfered activation gradient is also needed to compute the weight
			//       gradient of the last weight matrix (see backward()).
			mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);

			// Load the temporary forward matrix for the relu transfer
			fragment<matrix_a, 16, 16, 16, __half, row_major> forward_frag;
			load_matrix_sync(forward_frag, forward + layer_stride * n_hidden_matmuls + weights_col + (elem_idx + l * 16) * WIDTH, WIDTH);

			// v19/v34: Use warp_activation_backward<__half> (same as CUDA)
			warp_activation_backward<__half>(ACTIVATION, result_frag[l], forward_frag, result_frag[l]);
		}

		__syncthreads();

		// v19/v34: Store directly (no conversion needed)
		TCNN_PRAGMA_UNROLL
		for (int l = 0; l < N_ITERS; ++l) {
			store_matrix_sync(act_shmem + weights_col + (16 * l) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, mem_row_major);
		}

		__syncthreads();

		TCNN_PRAGMA_UNROLL
		for (int i = 0; i < N_ITERS; ++i) {
			*(int4*)&out_intermediate[lane_offset + (row + elem_idx + i * 16) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * i) * (WIDTH + SKEW)];
		}
	} else {
		// If the output width is larger than 16, we will have used CUTLASS for backpropping through the last layer.
		// Load the resulting gradients.
		threadblock_load_input_static<WIDTH, N_ITERS>(act_shmem, out_intermediate + elem_idx * WIDTH);
	}

	// Backprop through hidden layers
	for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
		threadblock_layer<WIDTH, N_ITERS, __half, true>(ACTIVATION, act_shmem, weights + weights_stride * (n_hidden_matmuls - k - 1), out_intermediate + layer_stride * (k + 1) + elem_idx_base * WIDTH, forward + layer_stride * (n_hidden_matmuls - k - 1) + elem_idx_base * WIDTH);
	}

	// Compute loss gradients w.r.t. input if desired.
	// THIS CODE ASSUMES THAT THE INPUT WIDTH IS THE SAME AS THE NETWORK WIDTH
	// AND THAT THE INPUT LAYOUT IS THE SAME AS THE HIDDEN LAYOUT.
	// DON'T PASS A NON-NULL dL_dinput IF THIS REQUIREMENT IS NOT MET.
	if (dL_dinput != nullptr) {
		threadblock_layer<WIDTH, N_ITERS, __half, true>(Activation::None, act_shmem, weights_first_layer, dL_dinput + elem_idx_base * WIDTH);
	}
}

template <int WIDTH, typename T, Activation ACTIVATION>
std::enable_if_t<!std::is_same<__half, T>::value> mlp_fused_backward(
	hipStream_t stream,
	const GPUMatrix<T, RM>& weights_first_layer,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrix<T>& temporaries,
	const GPUMatrix<T>& forward,
	GPUMatrixDynamic<T>* dL_dinput,
	const uint32_t n_hidden_matmuls
) {
	throw std::runtime_error{"The fully fused backward pass only supports __half precision."};
}

template <int WIDTH, typename T, Activation ACTIVATION>
std::enable_if_t<std::is_same<__half, T>::value> mlp_fused_backward(
	hipStream_t stream,
	const GPUMatrix<T, RM>& weights_first_layer,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrix<T>& temporaries,
	const GPUMatrix<T>& forward,
	GPUMatrixDynamic<T>* dL_dinput,
	const uint32_t n_hidden_matmuls
) {
	const uint32_t batch_size = dL_doutput.cols();
	const uint32_t out_width = dL_doutput.rows();
	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	const int N_ITERS = WIDTH >= 256 ? 2 : 8;

	CHECK_THROW(forward.cols() == batch_size);
	CHECK_THROW(batch_size % (16 * N_ITERS) == 0);
	CHECK_THROW(!dL_dinput || dL_dinput->layout() == RM || dL_dinput->stride() == dL_dinput->m());

	// ROCm: AMD GPUs use 64-thread waves (not 32-thread warps like NVIDIA)
	// rocWMMA requires the entire wavefront to be active
	const dim3 threads = { WAVE_SIZE, N_BLOCKS, 1 }; // Full wave per row, N_BLOCKS waves per block

	uint32_t n_elems_per_block = 16 * N_ITERS;
	uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);

	int shmem_size = sizeof(__half) * ((16 * N_ITERS) * (WIDTH + SKEW)); // WIDTH rows of input and 16 * threads.z rows of weights
	const dim3 blocks = { n_blocks, 1u, 1u };
	
	// The kernels operate with transposed layouts compared with the MLP code
	if (dL_doutput.layout() == RM) {
		check_shmem_error(hipFuncSetAttribute(reinterpret_cast<const void*>(kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, rocwmma::col_major>), hipFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
		kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, rocwmma::col_major><<<blocks, threads, shmem_size, stream>>>(dL_doutput.data(), weights.data(), temporaries.data(), forward.data(), dL_dinput ? dL_dinput->data() : nullptr, weights_first_layer.data(), dL_doutput.stride(), batch_size, out_width, n_hidden_matmuls);
	} else {
		check_shmem_error(hipFuncSetAttribute(reinterpret_cast<const void*>(kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, rocwmma::row_major>), hipFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
		kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, rocwmma::row_major><<<blocks, threads, shmem_size, stream>>>(dL_doutput.data(), weights.data(), temporaries.data(), forward.data(), dL_dinput ? dL_dinput->data() : nullptr, weights_first_layer.data(), dL_doutput.stride(), batch_size, out_width, n_hidden_matmuls);
	}
}

template <int WIDTH, int N_ITERS, typename OUT_T, typename INPUT_LAYOUT>
__device__ void threadblock_input_layer_forward_dynamic(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const uint32_t in_width, const uint32_t batch_size) {
	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch
	// input_threadblock points to the thread block's chunk of the input batch in global memory
	// weights_this_layer points to the weight matrix of the current layer
	// out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
	//                  Can be nullptr if nothing should be written.
	// in_width is the dynamic width of the input layer

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t INPUT_SKEW = 8;  // used to handle padding, ensure alignment in shmem 
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	using namespace rocwmma;

	// Fragments: small tiles of matrices 
	fragment<matrix_a, 16, 16, 16, __half, INPUT_LAYOUT> act_frag;
	fragment<matrix_b, 16, 16, 16, __half, col_major> weights_frag;
	// v19/v34: Use OUT_T accumulator (same as CUDA)
	fragment<accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

	// Indices
	const uint32_t li = threadIdx.x; // index in wave ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("wave index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * WAVE_SIZE) / WIDTH;  // Updated for 64-thread waves

	const uint32_t weights_col = 16 * wi;

	__half* __restrict__ weights_shmem = act_shmem + 16 * (in_width + INPUT_SKEW);

	// Load input weight matrix (fits completely into shared memory)
	// Each thread can load 8 fp16 elements (16 bytes) at once; we have N_BLOCKS waves
	const uint32_t n_elems_per_load = N_BLOCKS * WAVE_SIZE * 8;  // Updated for 64-thread waves
	const uint32_t thread_elem_idx = (li + wi * WAVE_SIZE) * 8;  // Updated for 64-thread waves

	const uint32_t n_elems_b = WIDTH * in_width;

	TCNN_PRAGMA_UNROLL
	for (uint32_t idx = thread_elem_idx; idx < n_elems_b; idx += n_elems_per_load) {
		const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
		*(int4*)&weights_shmem[idx_skewed] = *(int4*)&weights_this_layer[idx];
	}

	const uint32_t n_tensor_ops = in_width / 16;

	if (std::is_same<INPUT_LAYOUT, col_major>::value) {
		__syncthreads();
	}

	TCNN_PRAGMA_UNROLL
	for (int l = 0; l < N_ITERS; ++l) {
		if (std::is_same<INPUT_LAYOUT, row_major>::value) {
			// Load chunk of inputs into shmem.
			// This is faster than loading it from gmem directly, even though it is only used once.
			// (Possibly due to latency hiding through staging.)
			const uint32_t n_elems_a = 16 * in_width;

			TCNN_PRAGMA_UNROLL
			for (uint32_t idx = thread_elem_idx; idx < n_elems_a; idx += n_elems_per_load) {
				const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
				*(int4*)&act_shmem[idx_skewed] = *(int4*)&input_threadblock[l * n_elems_a + idx];
			}

			__syncthreads();

		}


		// 1. `fill_fragment(result_frag[l], 0.0f);`
		fill_fragment(result_frag[l], 0.0f);
		TCNN_PRAGMA_UNROLL
		// 2. `for (uint32_t i = 0; i < n_tensor_ops; ++i)`
		for (uint32_t i = 0; i < n_tensor_ops; ++i) {
			// 3. `load_matrix_sync(...)`

			// 4. `mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);`
			// Load chunk of inputs and weights from shared memory and multiply them
			if (std::is_same<INPUT_LAYOUT, row_major>::value) {
				load_matrix_sync(act_frag, act_shmem + 16 * i, in_width + INPUT_SKEW);
			} else {
				load_matrix_sync(act_frag, input_threadblock + 16 * i * batch_size + 16 * l, batch_size);
			}
			load_matrix_sync(weights_frag, weights_shmem + 16 * i + weights_col * (in_width + INPUT_SKEW), in_width + INPUT_SKEW);
			mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);
		}

		if (std::is_same<INPUT_LAYOUT, row_major>::value) {
			__syncthreads();
		}

		// v19/v34: Activation using OUT_T (same as CUDA)
		warp_activation<OUT_T>(activation, result_frag[l], result_frag[l]);
	}

	if (std::is_same<INPUT_LAYOUT, col_major>::value) {
		__syncthreads();

	}

	// v19/v34: Store directly (no conversion needed when OUT_T = __half)
	TCNN_PRAGMA_UNROLL
	for (int l = 0; l < N_ITERS; ++l) {
		store_matrix_sync(act_shmem + weights_col + (16 * l) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, mem_row_major);
	}


	if (out_intermediate_threadblock_this_layer != nullptr) {
		__syncthreads();

		TCNN_PRAGMA_UNROLL
		for (int i = 0; i < N_ITERS; ++i) {
			*(int4*)&out_intermediate_threadblock_this_layer[lane_offset + (row + 16 * i) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * i) * (WIDTH + SKEW)];
		}
	}
}

template <int WIDTH, int N_ITERS, typename OUT_T>
__device__ void threadblock_last_layer_forward(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out, const uint32_t output_stride, const rocwmma::layout_t output_layout) {
	

	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch
	// weights_this_layer points to the weight matrix of the current layer
	// out points to the location where the result produced by the thread block should be written to.
	//   Can be nullptr if nothing should be written.

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	using namespace rocwmma;

	// Fragments
	fragment<matrix_a, 16, 16, 16, __half, row_major> act_frag;
	fragment<matrix_b, 16, 16, 16, __half, col_major> weights_frag[N_BLOCKS];
	// v19/v34: Use OUT_T accumulator (same as CUDA)
	fragment<accumulator, 16, 16, 16, OUT_T> result_frag;

	// Indices
	const uint32_t li = threadIdx.x; // index in wave ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("wave index")

	__half* __restrict__ weights_shmem = act_shmem + N_ITERS * 16 * (WIDTH + SKEW);

	const uint32_t weights_row = (8 * li) % WIDTH;
	const uint32_t weights_col = (8 * li + 8 * WAVE_SIZE * wi) / WIDTH;  // Updated for 64-thread waves

	// Load weight matrix into shared memory for the last multiplication.
	// Loading into shared memory as opposed to directly into registers is faster
	// because unlike in the previous layers, each warp uses the same entries of the weight matrix.
	*(int4*)&weights_shmem[weights_row + weights_col * (WIDTH + SKEW)] = *(int4*)&weights_this_layer[weights_row + weights_col * WIDTH];

	__syncthreads();

	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < N_BLOCKS; ++i)
		load_matrix_sync(weights_frag[i], weights_shmem + 16 * i, WIDTH + SKEW);

	// Perform last layer by parallelizing over iters
	for (uint32_t idx = wi; idx < N_ITERS; idx += N_BLOCKS) {
		fill_fragment(result_frag, 0.0f);
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < N_BLOCKS; ++i) {
			// Load a chunk of intermediate activations from shared memory and multiply with chunk of the weight matrix
			load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * idx) * (WIDTH + SKEW), WIDTH + SKEW);
			mma_sync(result_frag, act_frag, weights_frag[i], result_frag);
		}

		// v19/v34: Activation using OUT_T (same as CUDA)
		warp_activation<OUT_T>(activation, result_frag, result_frag);

		// v19/v34: Store directly (no conversion needed when OUT_T = __half)
		if (output_layout == mem_row_major) {
			store_matrix_sync(out + idx * 16 * output_stride, result_frag, output_stride, output_layout);
		} else {
			store_matrix_sync(out + idx * 16, result_frag, output_stride, output_layout);
		}
	}
}

template <int WIDTH, int N_ITERS>
__device__ void threadblock_write_output_static(const __half* __restrict__ act_shmem, __half* __restrict__ output_threadblock) {
	

	// output_threadblock will be filled by the thread block's act_shmem

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in wave ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("wave index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * WAVE_SIZE) / WIDTH;  // Updated for 64-thread waves

	__syncthreads();

	TCNN_PRAGMA_UNROLL
	for (int i = 0; i < N_ITERS; ++i) {
		*(int4*)&output_threadblock[lane_offset + (row + 16 * i) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * i) * (WIDTH + SKEW)];
	}
}

template <int WIDTH, int N_ITERS, typename OUT_T, Activation ACTIVATION, bool INFERENCE>
__global__ void kernel_mlp_fused(const Activation output_activation, const __half* __restrict__ input, const __half* __restrict__ weights, OUT_T* __restrict__ out_intermediate, OUT_T* __restrict__ out, const uint32_t output_stride, const uint32_t batch_size, const uint32_t in_width, const uint32_t out_width, const uint32_t n_hidden_matmuls, const rocwmma::layout_t input_layout, const rocwmma::layout_t output_layout,
	 __half* first_layer_post_gpu_buffer) {

	// `input` points to the input matrix. Can be any width.
	// `weights` points to the weight matrices (contiguous in memory).
	// `out_intermediate` points to the memory where intermediate activations should be written. When performing inference, a value of nullptr is expected (intermediate results are not written).
	// `out` points to the memory where the network output should be written. (Output width is assumed to be 16 neurons.)

	// Commented out due to isolated strange side-effects on Windows
	// if (INFERENCE) {
	// 	assert(out_intermediate == nullptr);
	// } else {
	// 	assert(out_intermediate);
	// }
	
	// Shared memory contains the intermediate activations of blockDim.y*16 elements.
	// In some cases, it also contains the weight matrix for the first and last layer.
	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;

	// Each block computes exactly one 16-element chunk of the batch.
	const uint32_t elem_idx = 16 * blockIdx.x * N_ITERS;

	// First layer
	if (input_layout == rocwmma::mem_col_major || in_width != WIDTH) {
		if (input_layout == rocwmma::mem_row_major) {
			threadblock_input_layer_forward_dynamic<WIDTH, N_ITERS, OUT_T, rocwmma::row_major>(ACTIVATION, act_shmem, input + elem_idx * in_width, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width, batch_size);
		} else {
			threadblock_input_layer_forward_dynamic<WIDTH, N_ITERS, OUT_T, rocwmma::col_major>(ACTIVATION, act_shmem, input + elem_idx, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width, batch_size);
		}
	} else { 
		// If the input has the same width & layout as the hidden layers, we can simply use the network's regular layer routine (with static size)
		// instead of using the slower dynamic input layer routine.
		threadblock_load_input_static<WIDTH, N_ITERS>(act_shmem, input + elem_idx * WIDTH);
		
		// v33: Check raw input (before FirstLayer computation)
		if (!INFERENCE) {
			check_shmem_input<WIDTH, N_ITERS>(act_shmem, "RawInput", 0);
		}
		
		threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr);
		
		// v33: Check FirstLayer output (after FirstLayer computation)
		if (!INFERENCE) {
			check_shmem_input<WIDTH, N_ITERS>(act_shmem, "FirstLayer_Out", 0);
		}
	}

	const uint32_t first_weights_stride = WIDTH * in_width;
	const uint32_t weights_stride = WIDTH * WIDTH;
	const uint32_t layer_stride = WIDTH * batch_size;

	// Hidden layers
	for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
		// v32: Check input before each hidden layer
		if (!INFERENCE) {
			check_shmem_input<WIDTH, N_ITERS>(act_shmem, "Hidden", k);
		}
		threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights + first_weights_stride + weights_stride * k, !INFERENCE ? (out_intermediate + layer_stride * (k + 1) + elem_idx * WIDTH) : nullptr);
	}

	if (out_width > 16) {
		// In the forward pass, intermediate activations are already written out.
		if (INFERENCE) {
			threadblock_write_output_static<WIDTH, N_ITERS>(act_shmem, out_intermediate + elem_idx * WIDTH);
		}
	} else if (out) {
		// Last layer
		// v32: Check input before last layer
		if (!INFERENCE) {
			check_shmem_input<WIDTH, N_ITERS>(act_shmem, "LastLayer", n_hidden_matmuls);
		}
		
		if (output_layout == rocwmma::mem_row_major) {
			threadblock_last_layer_forward<WIDTH, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_weights_stride + weights_stride * n_hidden_matmuls, out + elem_idx * output_stride, output_stride, output_layout);
		} else {
			threadblock_last_layer_forward<WIDTH, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_weights_stride + weights_stride * n_hidden_matmuls, out + elem_idx, output_stride, output_layout);
		}
	}
}

template <int WIDTH, typename T, Activation ACTIVATION, bool INFERENCE>
std::enable_if_t<!std::is_same<__half, T>::value> mlp_fused_forward(
	hipStream_t stream,
	Activation output_activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& input,
	GPUMatrix<T>& output_intermediate,
	GPUMatrixDynamic<T>* output,
	const uint32_t n_hidden_layers
) {
	throw std::runtime_error{"The fully fused forward pass only supports __half precision."};
}

template <int WIDTH, typename T, Activation ACTIVATION, bool INFERENCE>
std::enable_if_t<std::is_same<__half, T>::value> mlp_fused_forward(
	hipStream_t stream,
	Activation output_activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& input,
	GPUMatrix<T>& output_intermediate,
	GPUMatrixDynamic<T>* output,
	const uint32_t n_hidden_layers
) {
	const uint32_t batch_size = input.cols();
	const uint32_t in_width = input.rows();

	// v35: Host-side sampling of forward input matrix (diagnostics)
	{
		static int s_fwd_calls = 0;
		++s_fwd_calls;

		const size_t rows = in_width;
		const size_t cols = batch_size;
		const size_t total = rows * cols;
		const size_t sample = total < 4096 ? total : (size_t)4096;

		std::vector<__half> host(sample);
		hipError_t err = hipMemcpy(host.data(), input.data(), sample * sizeof(__half), hipMemcpyDeviceToHost);

		size_t nan_count = 0, inf_count = 0;
		float max_abs = 0.0f;

		if (err == hipSuccess) {
			for (size_t i = 0; i < sample; ++i) {
				float v = __half2float(host[i]);
				if (std::isnan(v)) { ++nan_count; }
				else if (std::isinf(v)) { ++inf_count; }
				float a = fabsf(v);
				if (a > max_abs) max_abs = a;
			}

			const char* layout_name = input.layout() == RM ? "RM" : "CM";
			const char* mapped_layout = input.layout() == RM ? "mem_col_major" : "mem_row_major";

			// Print first few calls and then every 500 calls to limit verbosity
			if (s_fwd_calls <= 5 || (s_fwd_calls % 500) == 0) {
				printf("[v35 FWD-IN] call=%d layout=%s rows=%zu cols=%zu stride=%u mapped=%s NaN=%zu Inf=%zu max=%.2f\n",
					s_fwd_calls, layout_name, rows, cols, input.stride(), mapped_layout, nan_count, inf_count, max_abs);
			}
		} else {
			printf("[v35 FWD-IN] call=%d hipMemcpy failed: %d\n", s_fwd_calls, (int)err);
		}
	}

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0; // <- always going to be 8 as we only support multiple-of-16 widths
	constexpr uint32_t INPUT_SKEW = 8; // <- likewise with inputs
	constexpr uint32_t N_BLOCK_ROWS = WIDTH / 16;

	static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");

	CHECK_THROW(in_width % 16 == 0);
	CHECK_THROW(weights.rows() == WIDTH);
	CHECK_THROW(weights.cols() % 16 == 0);
	CHECK_THROW(output_intermediate.cols() == batch_size);
	CHECK_THROW(!output || output->cols() == batch_size);
	CHECK_THROW(input.layout() == RM || input.stride() == input.m());

	const int N_ITERS = WIDTH >= 256 ? 2 : 8;

	if (batch_size % (16 * N_ITERS) != 0) {
		throw std::runtime_error{fmt::format("Batch size must be a multiple of {}.", 16 * N_ITERS)};
	}

	// ROCm: AMD GPUs use 64-thread waves (not 32-thread warps like NVIDIA)
	// rocWMMA requires the entire wavefront to be active
	// Reference: https://rocwmma.readthedocs.io/en/latest/conceptual/programmers-guide.html
	const dim3 threads = { WAVE_SIZE, N_BLOCK_ROWS, 1 }; // Full wave per row, N_BLOCK_ROWS waves per block

	uint32_t n_elems_per_block = 16 * N_ITERS;
	uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);

	size_t shmem_size = sizeof(__half) * (16 + 16 * N_ITERS) * (WIDTH + SKEW); // 16*WIDTH rows of weights (for the last layer; others are in registers only) + 16*WIDTH*N_ITERS rows of intermediate activations
	if (in_width != WIDTH || input.layout() == RM) {
		// If the input width is dynamic, the input weight matrix as well as part of the input will live in extra shared memory
		shmem_size = std::max(shmem_size, sizeof(__half) * (WIDTH + 16) * (in_width + INPUT_SKEW));
	}

	const dim3 blocks = { n_blocks, 1u, 1u };

	// add tmp gpu & host buffer for shmem print (Sep-23) 
	__half* first_layer_post_gpu_buffer = nullptr ;
	__half* first_layer_post_host_buffer = nullptr ; 

	check_shmem_error(hipFuncSetAttribute(reinterpret_cast<const void*>(kernel_mlp_fused<WIDTH, N_ITERS, __half, ACTIVATION, INFERENCE>), hipFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));
	kernel_mlp_fused<WIDTH, N_ITERS, __half, ACTIVATION, INFERENCE><<<blocks, threads, shmem_size, stream>>>(
		output_activation,
		input.data(),
		weights.data(),
		output_intermediate.data(),
		output ? output->data() : nullptr,
		output ? output->stride() : 0,
		batch_size,
		in_width,
		output ? output->rows() : 0,
		n_hidden_layers,
		// The kernels operate with transposed layouts compared with the MLP code
		input.layout() == RM ? rocwmma::mem_col_major : rocwmma::mem_row_major,
		output && output->layout() == RM ? rocwmma::mem_col_major : rocwmma::mem_row_major,
		first_layer_post_gpu_buffer
	);
}

template <typename T, int WIDTH>
FullyFusedMLP<T, WIDTH>::FullyFusedMLP(
	uint32_t input_width,
	uint32_t output_width,
	uint32_t n_hidden_layers,
	Activation activation,
	Activation output_activation
) :
m_input_width{input_width}, // 64 
m_network_width{WIDTH},	 // 64 
m_output_width{output_width}, 
m_n_hidden_layers{n_hidden_layers},
m_activation{activation},  // ReLU
m_output_activation{output_activation}   // None 
{
	if (m_n_hidden_layers <= 0) {
		throw std::runtime_error("FullyFusedMLP requires at least 1 hidden layer (3 layers in total).");
	}

	m_n_hidden_matmuls = n_hidden_layers-1;  //  n_hidden_layers(1),  m_n_hidden_matmuls(0)

	m_padded_output_width = next_multiple(m_output_width, REQUIRED_ALIGNMENT());

	// Create matrices related to weights
	m_weight_matrices.emplace_back(nullptr, m_network_width, m_input_width);
	m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_input_width);
	m_gradient_matrices.emplace_back(nullptr, m_network_width, m_input_width);

	for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
		m_weight_matrices.emplace_back(nullptr, m_network_width, m_network_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_network_width);
		m_gradient_matrices.emplace_back(nullptr, m_network_width, m_network_width);
	}

	m_weight_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);
	m_weight_matrices_inference.emplace_back(nullptr, m_padded_output_width, m_network_width);
	m_gradient_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);

	// Determine total number of memory entries and set it
	m_total_n_params = 0;
	for (const auto& m : m_weight_matrices) {
		m_total_n_params += m.n_elements();
	}
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::inference_mixed_precision_impl(hipStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>& output, bool use_inference_params) {
	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();

	GPUMatrix<T> inference_tmp = m_output_width > 16 ? GPUMatrix<T>{m_network_width, batch_size, stream} : GPUMatrix<T>{nullptr, m_network_width, batch_size};

	// ASSUMPTION: weight matrices are contiguous in memory
	switch (m_activation) {
		case Activation::None:        mlp_fused_forward<WIDTH, T, Activation::None, true>(       stream, m_output_activation, input_weight_matrix(use_inference_params), input, inference_tmp, &output, m_n_hidden_matmuls); break;
		case Activation::Exponential: mlp_fused_forward<WIDTH, T, Activation::Exponential, true>(stream, m_output_activation, input_weight_matrix(use_inference_params), input, inference_tmp, &output, m_n_hidden_matmuls); break;
		case Activation::Sigmoid:     mlp_fused_forward<WIDTH, T, Activation::Sigmoid, true>(    stream, m_output_activation, input_weight_matrix(use_inference_params), input, inference_tmp, &output, m_n_hidden_matmuls); break;
		case Activation::ReLU:        mlp_fused_forward<WIDTH, T, Activation::ReLU, true>(       stream, m_output_activation, input_weight_matrix(use_inference_params), input, inference_tmp, &output, m_n_hidden_matmuls); break;
		case Activation::LeakyReLU:   mlp_fused_forward<WIDTH, T, Activation::LeakyReLU, true>(  stream, m_output_activation, input_weight_matrix(use_inference_params), input, inference_tmp, &output, m_n_hidden_matmuls); break;
		case Activation::Squareplus:  mlp_fused_forward<WIDTH, T, Activation::Squareplus, true>( stream, m_output_activation, input_weight_matrix(use_inference_params), input, inference_tmp, &output, m_n_hidden_matmuls); break;
		case Activation::Softplus:    mlp_fused_forward<WIDTH, T, Activation::Softplus, true>(   stream, m_output_activation, input_weight_matrix(use_inference_params), input, inference_tmp, &output, m_n_hidden_matmuls); break;
		case Activation::Tanh:        mlp_fused_forward<WIDTH, T, Activation::Tanh, true>(       stream, m_output_activation, input_weight_matrix(use_inference_params), input, inference_tmp, &output, m_n_hidden_matmuls); break;
		default: throw std::runtime_error{"Unsupported activation."};
	}

	// If we have more than 16 output dimensions, these will be taken care of by CUTLASS rather than
	// the fully fused kernel (which will have written out the second-to-last layer activations).
	if (m_output_width > 16) {
		fc_multiply(stream, output_weight_matrix(use_inference_params), inference_tmp, output, m_output_activation);
	}
}

template <typename T, int WIDTH>
std::unique_ptr<Context> FullyFusedMLP<T, WIDTH>::forward_impl(hipStream_t stream, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* output, bool use_inference_params, bool prepare_input_gradients) {
	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();
	auto forward = allocate_forward_buffers(stream, batch_size);

	// v36: Host-side weight global sampling (diagnostics)
	{
		size_t total_params = 0;
		for (const auto& m : m_weight_matrices) total_params += m.n_elements();

		const char* set_name = use_inference_params ? "inference" : "train";
		printf("[v36 WEIGHTS] set=%s total_params=%zu\n", set_name, total_params);

		for (size_t i = 0; i < m_weight_matrices.size(); ++i) {
			const auto& W = use_inference_params ? m_weight_matrices_inference[i] : m_weight_matrices[i];
			const size_t n = W.n_elements();
			const size_t sample_n = n < 4096 ? n : (size_t)4096;

			std::vector<__half> host(sample_n);
			hipError_t err = hipMemcpyAsync(host.data(), W.data(), sample_n * sizeof(__half), hipMemcpyDeviceToHost, stream);
			if (err != hipSuccess) {
				printf("[v36 WEIGHTS] idx=%zu m=%u n=%u memcpy_failed=%d\n", i, W.m(), W.n(), (int)err);
				continue;
			}
			hipStreamSynchronize(stream);

			size_t nan_count = 0, inf_count = 0;
			float max_abs = 0.0f;
			for (size_t k = 0; k < sample_n; ++k) {
				float v = __half2float(host[k]);
				if (std::isnan(v)) { ++nan_count; }
				else if (std::isinf(v)) { ++inf_count; }
				float a = fabsf(v);
				if (a > max_abs) max_abs = a;
			}

			printf("[v36 WEIGHTS] idx=%zu shape=[%u,%u] stride=%u NaN=%zu Inf=%zu max=%.4f\n",
				i, W.m(), W.n(), W.stride(), nan_count, inf_count, max_abs);
		}
	}

	// ASSUMPTION: weight matrices & forward_tmp matrices are contiguous in memory
	switch (m_activation) {
		case Activation::None:        mlp_fused_forward<WIDTH, T, Activation::None, false>(       stream, m_output_activation, input_weight_matrix(use_inference_params), input, forward->hidden.at(0), output, m_n_hidden_matmuls); break;
		case Activation::Exponential: mlp_fused_forward<WIDTH, T, Activation::Exponential, false>(stream, m_output_activation, input_weight_matrix(use_inference_params), input, forward->hidden.at(0), output, m_n_hidden_matmuls); break;
		case Activation::Sigmoid:     mlp_fused_forward<WIDTH, T, Activation::Sigmoid, false>(    stream, m_output_activation, input_weight_matrix(use_inference_params), input, forward->hidden.at(0), output, m_n_hidden_matmuls); break;
		case Activation::ReLU:        mlp_fused_forward<WIDTH, T, Activation::ReLU, false>(       stream, m_output_activation, input_weight_matrix(use_inference_params), input, forward->hidden.at(0), output, m_n_hidden_matmuls); break;
		case Activation::LeakyReLU:   mlp_fused_forward<WIDTH, T, Activation::LeakyReLU, false>(  stream, m_output_activation, input_weight_matrix(use_inference_params), input, forward->hidden.at(0), output, m_n_hidden_matmuls); break;
		case Activation::Squareplus:  mlp_fused_forward<WIDTH, T, Activation::Squareplus, false>( stream, m_output_activation, input_weight_matrix(use_inference_params), input, forward->hidden.at(0), output, m_n_hidden_matmuls); break;
		case Activation::Softplus:    mlp_fused_forward<WIDTH, T, Activation::Softplus, false>(   stream, m_output_activation, input_weight_matrix(use_inference_params), input, forward->hidden.at(0), output, m_n_hidden_matmuls); break;
		case Activation::Tanh:        mlp_fused_forward<WIDTH, T, Activation::Tanh, false>(       stream, m_output_activation, input_weight_matrix(use_inference_params), input, forward->hidden.at(0), output, m_n_hidden_matmuls); break;
		default: throw std::runtime_error{"Unsupported activation."};
	}

	// If we have more than 16 output dimensions, these will be taken care of by CUTLASS rather than
	// the fully fused kernel (which will have written out the second-to-last layer activations).
	if (output && m_output_width > 16) {
		fc_multiply(stream, output_weight_matrix(use_inference_params), forward->hidden.back(), *output, *output, m_output_activation);
	}

	return forward;
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::backward_impl(
	hipStream_t stream,
	const Context& ctx,
	const GPUMatrixDynamic<T>& input,
	const GPUMatrixDynamic<T>& output,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrixDynamic<T>* dL_dinput,
	bool use_inference_params,
	GradientMode param_gradients_mode
) {
	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = dL_doutput.n();

	std::vector<GPUMatrix<T>> backward_tmp(num_forward_activations());
	for (uint32_t i = 0; i < num_forward_activations(); ++i) {
		backward_tmp[i].set_size_unsafe(m_network_width, batch_size);
	}
	auto backward_tmp_alloc = GPUMatrixBase::allocate_shared_memory(stream, backward_tmp);

	GPUMatrixDynamic<T> backward_output_tmp;
	if (m_output_activation != Activation::None) {
		backward_output_tmp = {m_padded_output_width, batch_size, stream, dL_doutput.layout()};
		activation_backward_output_gpu(stream, dL_doutput.n_elements(), m_output_activation, output.data(), dL_doutput.data(), backward_output_tmp.data());
	}

	const float param_gradient_beta = param_gradients_mode == GradientMode::Accumulate ? 1.0f : 0.0f;

	std::vector<SyncedMultiStream> multi_streams;

	const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

	int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);

	const GPUMatrixDynamic<T>& tmp_dL_doutput = m_output_activation == Activation::None ? dL_doutput : backward_output_tmp;

	uint32_t tmp_idx = m_n_hidden_matmuls; // 0
	uint32_t backward_tmp_idx = 0;

	if (param_gradients_mode != GradientMode::Ignore) {
		multi_streams.emplace_back(stream, 2);
		fc_multiply_split_k(multi_streams.back().get(1), tmp_dL_doutput, forward.hidden.at(tmp_idx).transposed(), output_gradient_matrix(), split_k_factor, param_gradient_beta);
	}

	if (m_output_width > 16) {
		fc_multiply(stream, output_weight_matrix(use_inference_params).transposed(), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), backward_tmp.at(backward_tmp_idx), m_activation, true, true);
	}

	auto dL_dinput_fused = input.m() == forward.hidden.at(0).m() && input.layout() == CM ? dL_dinput : nullptr;

	switch (m_activation) {
		case Activation::None:        mlp_fused_backward<WIDTH, T, Activation::None>(       stream, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
		case Activation::Exponential: mlp_fused_backward<WIDTH, T, Activation::Exponential>(stream, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
		case Activation::Sigmoid:     mlp_fused_backward<WIDTH, T, Activation::Sigmoid>(    stream, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
		case Activation::ReLU:        mlp_fused_backward<WIDTH, T, Activation::ReLU>(       stream, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
		case Activation::LeakyReLU:   mlp_fused_backward<WIDTH, T, Activation::LeakyReLU>(  stream, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
		case Activation::Squareplus:  mlp_fused_backward<WIDTH, T, Activation::Squareplus>( stream, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
		case Activation::Softplus:    mlp_fused_backward<WIDTH, T, Activation::Softplus>(   stream, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
		case Activation::Tanh:        mlp_fused_backward<WIDTH, T, Activation::Tanh>(       stream, input_weight_matrix(use_inference_params), weight_matrix_at(use_inference_params, 0), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), forward.hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
		default: throw std::runtime_error{"Unsupported activation."};
	}

	tmp_idx -= 1;
	++backward_tmp_idx;

	for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
		uint32_t matrix_idx = m_n_hidden_matmuls - i - 1;

		if (param_gradients_mode != GradientMode::Ignore) {
			multi_streams.emplace_back(stream, 2);
			fc_multiply_split_k(multi_streams.back().get(1), backward_tmp.at(backward_tmp_idx-1), forward.hidden.at(tmp_idx).transposed(), gradient_matrix_at(matrix_idx), split_k_factor, param_gradient_beta);
		}

		tmp_idx -= 1;
		++backward_tmp_idx;
	}

	if (param_gradients_mode != GradientMode::Ignore) {
		multi_streams.emplace_back(stream, 2);
		fc_multiply_split_k(multi_streams.back().get(1), backward_tmp.at(backward_tmp_idx-1), input.transposed(), input_gradient_matrix(), split_k_factor, param_gradient_beta);
	}

	if (dL_dinput && !dL_dinput_fused) {
		fc_multiply(stream, input_weight_matrix(use_inference_params).transposed(), backward_tmp.at(backward_tmp_idx-1), *dL_dinput, *dL_dinput);
	}
}

template <typename T, int WIDTH>
std::unique_ptr<typename FullyFusedMLP<T, WIDTH>::ForwardContext> FullyFusedMLP<T, WIDTH>::allocate_forward_buffers(hipStream_t stream, uint32_t batch_size) {
	auto forward = std::make_unique<ForwardContext>();

	// Use GPUMatrixBase::allocate_shared_memory to ensure the matrices occupy contiguous memory.
	// (Needed in the fully-fused kernels.)
	forward->hidden.resize(num_forward_activations());
	for (uint32_t i = 0; i < num_forward_activations(); ++i) {
		forward->hidden[i].set_size_unsafe(m_network_width, batch_size);
	}

	forward->alloc = GPUMatrixBase::allocate_shared_memory(stream, forward->hidden);

	return forward;
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::set_params_impl(T* params, T* inference_params, T* gradients) {
	size_t current_pos = 0;
	for (size_t i = 0; i < m_weight_matrices.size(); ++i) {
		m_weight_matrices[i].set_data_unsafe(params + current_pos);
		m_weight_matrices_inference[i].set_data_unsafe(inference_params + current_pos);
		m_gradient_matrices[i].set_data_unsafe(gradients + current_pos);
		current_pos += m_weight_matrices[i].n_elements();
	}
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::initialize_params(pcg32& rnd, float* params_full_precision, float scale) {
	// Construct weight matrices
	std::vector<GPUMatrix<float, RM>> weight_matrices_full_precision;
	weight_matrices_full_precision.emplace_back(params_full_precision, m_network_width, m_input_width);
	params_full_precision += weight_matrices_full_precision.back().n_elements();

	for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
		weight_matrices_full_precision.emplace_back(params_full_precision, m_network_width, m_network_width);
		params_full_precision += weight_matrices_full_precision.back().n_elements();
	}

	weight_matrices_full_precision.emplace_back(params_full_precision, m_padded_output_width, m_network_width);

	// Initialize matrices
	for (size_t i = 0; i < weight_matrices_full_precision.size(); ++i) {
		if (m_activation == Activation::Sine) {
			if (i == 0) {
				weight_matrices_full_precision[i].initialize_siren_uniform_first(rnd, scale);
			} else {
				weight_matrices_full_precision[i].initialize_siren_uniform(rnd, scale);
			}
		} else {
			weight_matrices_full_precision[i].initialize_xavier_uniform(rnd, scale);
		}
	}
}

template class FullyFusedMLP<network_precision_t, 128>;
template class FullyFusedMLP<network_precision_t, 64>;
template class FullyFusedMLP<network_precision_t, 32>;
template class FullyFusedMLP<network_precision_t, 16>;

}

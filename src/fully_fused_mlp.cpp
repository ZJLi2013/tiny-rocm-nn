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

namespace tcnn {

void check_shmem_error(hipError_t error) {
	if (error != hipSuccess) {
		throw std::runtime_error{"FullyFusedMLP: insufficient shared memory available on the GPU. Reduce `n_neurons` to fit available shared memory."};
	}
}

__device__ void sh2gmem(__half* device_mem, const __half* __restrict__ shmem, int N)
{
	// block_level_thread_index=threadIdx.z×(blockDim.x×blockDim.y)+threadIdx.y×blockDim.x+threadIdx.x
	// threadIdx.x, threadIdx.y, threadIdx.z are thread index within the block in x,y,z dimensions
	// blockDim.x, blockDim.y, blockDim.z are dimension/size of the t-block. a.k.a 32, 4, 1 
	int tid =  threadIdx.x + threadIdx.y * blockDim.x ; 
    if(tid< N){
        device_mem[tid] = shmem[tid]; 
    }
	__syncthreads(); 
}

template <int WIDTH, int N_ITERS, typename OUT_T, bool BACKWARD=false>
__device__ void threadblock_layer(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const OUT_T* __restrict__ activation_aux = nullptr) {
	// --- 中文注释 ---
	// `threadblock_layer`: 这是融合MLP中最核心的设备函数，负责处理一个“标准”隐藏层（即输入和输出宽度都等于网络宽度WIDTH）的前向或反向传播。
	//
	// 工作流程:
	// 1.  **加载权重**: 每个warp从全局内存中加载它所负责处理的那部分权重矩阵到寄存器中。
	//     - 在反向传播时 (`BACKWARD=true`)，权重会以转置的方式加载，这是通过改变内存访问模式实现的。
	// 2.  **循环计算**: 通过一个 `N_ITERS` 的循环，处理一个线程块所负责的所有数据。在每次迭代中：
	//     a. **加载激活值**: 从共享内存 `act_shmem` 中加载当前层的输入激活值（或梯度）到一个 `wmma::fragment` (WMMA硬件单元的操作对象) 中。
	//     b. **矩阵乘法**: 使用 `mma_sync` 指令，执行一次 16x16x16 的矩阵乘法累加操作。这个操作利用了Tensor Core，是性能的关键。
	//        它将从共享内存加载的激活值与寄存器中的权重相乘，结果累加到 `result_frag` 中。
	//     c. **激活函数**: 对矩阵乘法的结果应用指定的激活函数（或其导数，在反向传播时）。
	// 3.  **写回结果**: 将计算完成的结果（下一层的激活值或梯度）写回到共享内存 `act_shmem` 的相应位置，供下一层计算使用。
	// 4.  **(可选)写回全局内存**: 如果提供了 `out_intermediate_threadblock_this_layer` 指针，则将最终结果从共享内存写回到全局内存，
	//     以便在反向传播时可以访问到前向传播的中间激活值。

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

	// Fragments
	using MatrixA = fragment<matrix_a, 16, 16, 16, __half, row_major>;
	using MatrixB = fragment<matrix_b, 16, 16, 16, __half, weights_layout_t>;
	using Accumulator = fragment<accumulator, 16, 16, 16, OUT_T>;

	MatrixA act_frag;
	MatrixB weights_frag[N_BLOCKS];
	Accumulator result_frag[N_ITERS];

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	const uint32_t weights_col = 16 * wi;

	__syncthreads();

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
			// Load a chunk of intermediate activations from shared memory and multiply with chunk of weights
			load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * l) * (WIDTH + SKEW), WIDTH + SKEW);
			mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
		}

		// Activation
		if (BACKWARD) {
			// Load the temporary forward matrix for the relu transfer
			load_matrix_sync(act_frag, activation_aux + weights_col + l * 16 * WIDTH, WIDTH);
			warp_activation_backward<__half>(activation, result_frag[l], act_frag, result_frag[l]);
		} else {
			warp_activation<__half>(activation, result_frag[l], result_frag[l]);
		}
	}

	__syncthreads();

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
	// --- 中文注释 ---
	// `threadblock_load_input_static`: 一个简单的辅助设备函数。
	// 当输入层的维度 `in_width` 恰好等于隐藏层宽度 `WIDTH` 时（静态情况），
	// 这个函数被调用，负责将输入数据从全局内存 `input_threadblock` 高效地加载到共享内存 `act_shmem` 中。
	// 它使用 `int4` 类型转换来一次性加载8个 `__half` 元素，以最大化内存带宽。

	// act_shmem will be filled by the thread block's chunk of input_threadblock

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

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
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")
	const uint32_t bi = blockIdx.x;  // block index

	// Shared memory contains the intermediate activations of blockDim.y*16 elements.
	// A skew is applied to the matrix storage to avoid bank conflicts.
	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

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

			warp_activation_backward<__half>(ACTIVATION, result_frag[l], forward_frag, result_frag[l]);
		}

		__syncthreads();

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

	const dim3 threads = { 32u, N_BLOCKS, 1 }; // 32 threads = 1 warp, 8 warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)

	uint32_t n_elems_per_block = 16 * N_ITERS;
	uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);

	int shmem_size = sizeof(__half) * ((16 * N_ITERS) * (WIDTH + SKEW)); // WIDTH rows of input and 16 * threads.z rows of weights
	const dim3 blocks = { n_blocks, 1u, 1u };
	
	// The kernels operate with transposed layouts compared with the MLP code
	if (dL_doutput.layout() == RM) {
		check_shmem_error(hipFuncSetAttribute(reinterpret_cast<const void*>(kernel_mlp_fused_backward<WIDTH), N_ITERS, ACTIVATION, rocwmma::col_major>, hipFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
		kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, rocwmma::col_major><<<blocks, threads, shmem_size, stream>>>(dL_doutput.data(), weights.data(), temporaries.data(), forward.data(), dL_dinput ? dL_dinput->data() : nullptr, weights_first_layer.data(), dL_doutput.stride(), batch_size, out_width, n_hidden_matmuls);
	} else {
		check_shmem_error(hipFuncSetAttribute(reinterpret_cast<const void*>(kernel_mlp_fused_backward<WIDTH), N_ITERS, ACTIVATION, rocwmma::row_major>, hipFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
		kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, rocwmma::row_major><<<blocks, threads, shmem_size, stream>>>(dL_doutput.data(), weights.data(), temporaries.data(), forward.data(), dL_dinput ? dL_dinput->data() : nullptr, weights_first_layer.data(), dL_doutput.stride(), batch_size, out_width, n_hidden_matmuls);
	}
}

template <int WIDTH, int N_ITERS, typename OUT_T, typename INPUT_LAYOUT>
__device__ void threadblock_input_layer_forward_dynamic(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const uint32_t in_width, const uint32_t batch_size) {
	// --- 中文注释 ---
	// `threadblock_input_layer_forward_dynamic`: 这是一个设备函数，专门用于处理输入层的前向传播，特别是当输入维度 `in_width`
	// 不等于网络隐藏层宽度 `WIDTH` 时。这是一个“动态”版本，因为它需要处理可变的输入宽度。
	//
	// 工作流程:
	// 1.  **加载权重到共享内存**: 由于输入层的权重矩阵 (`in_width` x `WIDTH`) 相对较小，可以完全加载到共享内存 `weights_shmem` 中。
	//     
	// 2.  **分块处理输入**:
	//     - 如果输入是行主序 (`row_major`)，会先将一小块输入数据从全局内存加载到共享内存 `act_shmem` 中。这利用了暂存（staging）来隐藏延迟。
	//     - 如果输入是列主序 (`col_major`)，则直接从全局内存加载。
	// 3.  **矩阵乘法**: 在一个循环中，使用 `mma_sync` 指令将输入块与从共享内存加载的权重块进行矩阵乘法。
	// 4.  **激活与写回**: 对结果应用激活函数，然后将最终的激活值写回到共享内存 `act_shmem` 中，准备给下一个（隐藏）层使用。
	// 5.  **(可选)写回全局内存**: 与 `threadblock_layer` 类似，如果需要保存中间结果，则将其写回到全局内存。

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
	fragment<accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	const uint32_t weights_col = 16 * wi;

	__half* __restrict__ weights_shmem = act_shmem + 16 * (in_width + INPUT_SKEW);

	// Load input weight matrix (fits completely into shared memory)
	// Each thread can load 8 fp16 elements (16 bytes) at once; we have N_BLOCKS warps
	const uint32_t n_elems_per_load = N_BLOCKS * 32 * 8;
	const uint32_t thread_elem_idx = (li + wi * 32) * 8;

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

		// --- 中文注释: WMMA 指令流程 ---
		// 下面的代码块是利用NVIDIA GPU的Tensor Core进行高效矩阵乘法的核心。
		// 它遵循一个标准的 "load-multiply-accumulate-store" 模式，但所有操作都在warp级别上，并由硬件加速。

		// 1. `fill_fragment(result_frag[l], 0.0f);`
		//    初始化累加器片段 `result_frag`。每个warp都有自己的一组寄存器来存储这个片段，
		//    这里我们将其所有元素设置为0，为接下来的乘加操作做准备。
		fill_fragment(result_frag[l], 0.0f);
		TCNN_PRAGMA_UNROLL
		// 2. `for (uint32_t i = 0; i < n_tensor_ops; ++i)`
		//    这个循环将一个大的矩阵乘法拆分成一系列小的 16x16x16 的块乘法。
		//    `n_tensor_ops` 代表需要多少个这样的块乘法来覆盖整个输入维度。
		for (uint32_t i = 0; i < n_tensor_ops; ++i) {
			// 3. `load_matrix_sync(...)`
			//    这两条指令从共享内存或全局内存中加载一小块（16x16）的输入激活矩阵(A)和权重矩阵(B)
			//    到专门的寄存器片段 `act_frag` 和 `weights_frag` 中。
			//    `_sync` 后缀表示这是一个warp内的同步操作，确保所有线程都完成了加载。

			// 4. `mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);`
			//    这是执行矩阵乘法累加的核心指令： D = A * B + C。
			//    它调用Tensor Core硬件来计算 `act_frag` 和 `weights_frag` 的乘积，
			//    然后将结果累加到 `result_frag` 中。整个操作对warp中的所有线程是同步的。
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

		// --- 中文注释 ---
		// `warp_activation`: 这是一个设备函数，负责对一个 warp 所持有的 `wmma::fragment`（即矩阵分块）
		// 中的所有元素，逐个地应用指定的非线性激活函数（例如 ReLU）。
		//
		// 参数解析:
		// - `activation`: 一个枚举值，指定要应用的激活函数类型 (e.g., `Activation::ReLU`)。
		// - `result_frag[l]`: 输入和输出参数。它是一个 `wmma::fragment`，在调用前存储着矩阵乘法的线性结果，
		//   函数执行后，其内部的值会被更新为应用了激活函数之后的结果。
		//
		// 这个函数是实现 "矩阵乘法 -> 激活" 这一神经网络基本操作中的第二步，并且是为 WMMA 的数据结构优化的。
		//  in  common_device.h
		warp_activation<__half>(activation, result_frag[l], result_frag[l]);
	}

	if (std::is_same<INPUT_LAYOUT, col_major>::value) {
		__syncthreads();
	}

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
	// --- 中文注释 ---
	// `threadblock_last_layer_forward`: 这个设备函数专门用于处理网络的最后一层的前向传播，前提是输出维度 `<= 16`。
	//
	// 工作流程:
	// 1.  **加载权重到共享内存**: 与输入层类似，最后一层的权重矩阵 (`WIDTH` x `out_width`) 也被加载到共享内存中。
	// 2.  **加载权重到寄存器**: 从共享内存中将权重加载到每个warp的 `weights_frag` 寄存器片段中。
	// 3.  **循环计算**: 循环遍历 `N_ITERS` 次，处理线程块负责的所有数据。
	//     a. **加载激活值**: 从 `act_shmem` 加载倒数第二层的激活值。
	//     b. **矩阵乘法**: 使用 `mma_sync` 将激活值与权重相乘。
	//     c. **激活函数**: 应用最终的输出层激活函数。
	// 4.  **写回最终输出**: 使用 `store_matrix_sync` 将计算结果直接写入全局内存的最终输出位置 `out`。

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
	fragment<accumulator, 16, 16, 16, OUT_T> result_frag;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	__half* __restrict__ weights_shmem = act_shmem + N_ITERS * 16 * (WIDTH + SKEW);

	const uint32_t weights_row = (8 * li) % WIDTH;
	const uint32_t weights_col = (8 * li + 8 * 32 * wi) / WIDTH;

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

		warp_activation<__half>(activation, result_frag, result_frag);

		if (output_layout == mem_row_major) {
			store_matrix_sync(out + idx * 16 * output_stride, result_frag, output_stride, output_layout);
		} else {
			store_matrix_sync(out + idx * 16, result_frag, output_stride, output_layout);
		}
	}
}

template <int WIDTH, int N_ITERS>
__device__ void threadblock_write_output_static(const __half* __restrict__ act_shmem, __half* __restrict__ output_threadblock) {
	// --- 中文注释 ---
	// `threadblock_write_output_static`: 另一个简单的辅助设备函数。
	// 当输出维度 `out_width` > 16 时，融合核函数计算到最后一层隐藏层的激活值后，需要将这些激活值写出到全局内存，
	// 以便后续的 `fc_multiply` (CUTLASS) 调用可以读取它们。这个函数就负责这个写出操作，
	// 同样使用 `int4` 来一次性写入8个 `__half` 元素以提升效率。

	// output_threadblock will be filled by the thread block's act_shmem

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	__syncthreads();

	TCNN_PRAGMA_UNROLL
	for (int i = 0; i < N_ITERS; ++i) {
		*(int4*)&output_threadblock[lane_offset + (row + 16 * i) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * i) * (WIDTH + SKEW)];
	}
}

template <int WIDTH, int N_ITERS, typename OUT_T, Activation ACTIVATION, bool INFERENCE>
__global__ void kernel_mlp_fused(const Activation output_activation, const __half* __restrict__ input, const __half* __restrict__ weights, OUT_T* __restrict__ out_intermediate, OUT_T* __restrict__ out, const uint32_t output_stride, const uint32_t batch_size, const uint32_t in_width, const uint32_t out_width, const uint32_t n_hidden_matmuls, const rocwmma::layout_t input_layout, const rocwmma::layout_t output_layout,
	 __half* first_layer_post_gpu_buffer) {
	// --- 中文注释: 关于此核函数与 `FullyFusedMLP` 类的关系 ---
	//
	// 	// 1. **主机端管理者 (`FullyFusedMLP` 类)**:
	//    - 负责在CPU上进行管理工作，如存储网络配置、分配GPU内存、计算启动参数等。
	//    - 它通过 `mlp_fused_forward` 函数，最终调用 `hipLaunchKernel` 来启动这个核函数。
	//    - 所有核函数需要的数据，如权重指针、输入/输出指针、网络尺寸等，都由 `FullyFusedMLP` 类作为参数传递给核函数。
	//
	// 2. **设备端工作者 (此核函数)**:
	//    - 作为一个纯粹的计算单元在GPU上运行。
	//    - 它没有 `this` 指针，也无法访问 `FullyFusedMLP` 类的任何成员变量。
	//    - 它仅根据从参数中接收到的指针和数值来执行计算任务。
	//


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

	// --- 中文注释: 关于此 `if/else` 分支的判断 ---
	// 在 `mlp_learning_an_image.cu` 的默认配置下，会走 `threadblock_input_layer_forward_dynamic()` 这个动态路径？
	//
	// 1.  进入此 `if` 分支（即动态路径）的条件是 `input_layout == rocwmma::mem_col_major || in_width != WIDTH`。
	// 2.  在 `mlp_learning_an_image.cu` 中，网络的 `WIDTH` (隐藏层宽度) 由 `"n_neurons"` 决定，默认值为 64。
	// 3.  网络的输入维度 `in_width` 由其前面的 `Encoding` 层的输出维度决定。
	// 4.  默认配置中，`Encoding` 为 `"otype": "OneBlob", "n_bins": 32`。`OneBlob` 编码的输出维度就是 `n_bins` 的值。
	//     因此，`in_width` 等于 32。
	// 5.  比较两者，`in_width` (32) 不等于 `WIDTH` (64)。
	//
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
		threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr);
	}

	//Sep-11: 对于n_hidden_matmuls==0, out_intermediate 跟新就发生在first-layer。输出不一致，就说明是这里的不一致
	const uint32_t first_weights_stride = WIDTH * in_width;
	const uint32_t weights_stride = WIDTH * WIDTH;
	const uint32_t layer_stride = WIDTH * batch_size;

	// Hidden layers
	for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
		threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights + first_weights_stride + weights_stride * k, !INFERENCE ? (out_intermediate + layer_stride * (k + 1) + elem_idx * WIDTH) : nullptr);
	}

	if (out_width > 16) {
		// In the forward pass, intermediate activations are already written out.
		if (INFERENCE) {
			threadblock_write_output_static<WIDTH, N_ITERS>(act_shmem, out_intermediate + elem_idx * WIDTH);
		}
	} else if (out) {
		// Last layer
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

	const dim3 threads = { 32u, N_BLOCK_ROWS, 1 }; // 32 threads = 1 warp, N_BLOCK_ROWS warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)

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

	check_shmem_error(hipFuncSetAttribute(reinterpret_cast<const void*>(kernel_mlp_fused<WIDTH), N_ITERS, __half, ACTIVATION, INFERENCE>, hipFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));
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
	// 主要工作流程:
	// 1. 调用 `allocate_forward_buffers` 为存储中间层的激活值分配GPU内存。这些激活值在反向传播时需要用到。
	// 2. 根据网络配置的隐藏层激活函数 (m_activation)，通过一个 `switch` 语句选择并调用对应的 `mlp_fused_forward` 模板函数。
	// 3. `mlp_fused_forward` 函数会启动一个高度优化的、完全融合的CUDA核函数 `kernel_mlp_fused`。
	//    这个核函数在一个单一的kernel launch中完成所有隐藏层的“矩阵乘法 -> 激活”操作，从而最大化性能。
	//    它会将每一层的激活值（如果不在推理模式下）存储到 `forward->hidden` 中。
	//
	//    关于输入层和输出层是否单独处理：
	//    这个融合核的设计目标是尽可能多地将整个网络（包括输入、隐藏和输出层）的计算融合。但根据网络结构，处理方式有别：
	//    - 输入层: 总是在融合核函数内部处理。如果输入维度不等于隐藏层宽度，会走一个特殊的动态路径。
	//    - 隐藏层: 总是被完全融合，这是性能优势的核心。
	//    - 输出层: 分情况处理。如果输出维度<=16，则在核函数内部融合计算；如果>16，则由核函数外部的另一个cuBLAS/CUTLASS调用单独处理。
	//
	// 4. 函数最后返回一个 `forward` 上下文对象，其中包含了所有中间激活值，供反向传播使用。

	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();
	auto forward = allocate_forward_buffers(stream, batch_size);

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

	/*
		由于 kernel_mlp_fused 以及 __device__ 函数（threadblock_layer, threadblock_input_layer_forward_dynamic 等）完全不依赖 CUTLASS 或 cuBLAS。
		它们是纯粹的 WMMA (Warp Matrix Multiply-Accumulate) 指令实现，直接使用 NVIDIA Tensor Core 硬件。
		对于 FullyFusedMLP 需要从 cutlass 替换到 cublas 的 kernel 只有  fc_multiply() 相关的实现
			* 进一步， mlp_learning_an_image 中 default confgi:  n_input_dims(2), n_output_dims(3)，即 m_output_wdith(3) < 16
			* 说明，前向传播即使完全不会用到 fc_multiply 
	*/	

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
	// --- 中文注释 ---
	// `backward_impl` 是 `FullyFusedMLP` 反向传播的核心实现。
	// 核心思想是交替进行“计算激活梯度”（通过与权重矩阵的转置相乘）和“计算权重梯度”（通过与前一层的激活相乘）。
	// 融合核 (`mlp_fused_backward`) 将多层的“计算激活梯度”操作合并，以减少kernel launch开销和内存访问。

	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = dL_doutput.n();

	// 1. 分配用于存储反向传播过程中产生的中间梯度（即损失对每层激活值的梯度，∂L/∂a）的临时缓冲区 `backward_tmp`。
	std::vector<GPUMatrix<T>> backward_tmp(num_forward_activations());
	for (uint32_t i = 0; i < num_forward_activations(); ++i) {
		backward_tmp[i].set_size_unsafe(m_network_width, batch_size);
	}
	auto backward_tmp_alloc = GPUMatrixBase::allocate_shared_memory(stream, backward_tmp);

	// 2. (可选) 如果输出层有激活函数，首先计算损失对输出层激活函数输入（即 pre-activation）的梯度。
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

	// 3. 计算输出层权重的梯度：∂L/∂W_out = (∂L/∂a_out) * (a_hidden)^T。
	//    这里使用 `fc_multiply_split_k` 函数，它将矩阵乘法在K维度上拆分，以提高并行度和性能。
	if (param_gradients_mode != GradientMode::Ignore) {
		multi_streams.emplace_back(stream, 2);
		fc_multiply_split_k(multi_streams.back().get(1), tmp_dL_doutput, forward.hidden.at(tmp_idx).transposed(), output_gradient_matrix(), split_k_factor, param_gradient_beta);
	}

	// 4. (特殊情况) 如果输出维度 > 16，则使用 `fc_multiply` 计算损失对倒数第二层激活值的梯度 ∂L/∂a_hidden。
	if (m_output_width > 16) {
		fc_multiply(stream, output_weight_matrix(use_inference_params).transposed(), tmp_dL_doutput, backward_tmp.at(backward_tmp_idx), backward_tmp.at(backward_tmp_idx), m_activation, true, true);
	}

	auto dL_dinput_fused = input.m() == forward.hidden.at(0).m() && input.layout() == CM ? dL_dinput : nullptr;

	// 5. 调用 `mlp_fused_backward` 函数，该函数启动 `kernel_mlp_fused_backward` CUDA核。
	//    这个融合核函数从倒数第二层开始，一路反向传播，计算所有隐藏层的激活梯度。
	//    如果输入层和隐藏层尺寸相同，它甚至可以直接计算出 ∂L/∂input。
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

	// 6. 对于每一层，在计算出其输入的梯度 ∂L/∂a_{l-1} 之后，立即计算该层权重的梯度 ∂L/∂W_l = (∂L/∂a_l) * (a_{l-1})^T。
	//    这同样是通过 `fc_multiply_split_k` 实现的。
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

	// 7. 最后，如果需要且融合核未完成此工作，则单独计算 ∂L/∂input。
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

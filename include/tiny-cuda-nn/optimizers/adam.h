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

/** @file   adam.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the adam optimizer with support for
 *          the AdaBound paper: https://openreview.net/pdf?id=Bkg3g2R9FX
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/gpu_memory_json.h>
#include <json/json.hpp>

#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

namespace tcnn {

// v27 Diagnostic: Compute gradient statistics
template <typename T>
__global__ void compute_gradient_stats(
	const uint32_t n_elements,
	const T* __restrict__ gradients,
	const float loss_scale,
	float* __restrict__ partial_sums,
	float* __restrict__ partial_max
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	
	__shared__ float shmem_sum[256];
	__shared__ float shmem_max[256];
	
	float sum = 0.0f;
	float max_val = 0.0f;
	if (i < n_elements) {
		float g = (float)gradients[i] / loss_scale;
		sum = g * g;
		max_val = fabsf(g);
	}
	
	shmem_sum[threadIdx.x] = sum;
	shmem_max[threadIdx.x] = max_val;
	__syncthreads();
	
	// Reduction
	for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			shmem_sum[threadIdx.x] += shmem_sum[threadIdx.x + s];
			shmem_max[threadIdx.x] = fmaxf(shmem_max[threadIdx.x], shmem_max[threadIdx.x + s]);
		}
		__syncthreads();
	}
	
	if (threadIdx.x == 0) {
		partial_sums[blockIdx.x] = shmem_sum[0];
		partial_max[blockIdx.x] = shmem_max[0];
	}
}

// v27 Diagnostic: Compute weight statistics
template <typename T>
__global__ void compute_weight_stats(
	const uint32_t n_elements,
	const float* __restrict__ weights_fp32,
	const T* __restrict__ weights_fp16,
	float* __restrict__ partial_max_fp32,
	float* __restrict__ partial_max_fp16
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	
	__shared__ float shmem_max_fp32[256];
	__shared__ float shmem_max_fp16[256];
	
	float max_fp32 = 0.0f;
	float max_fp16 = 0.0f;
	if (i < n_elements) {
		max_fp32 = fabsf(weights_fp32[i]);
		max_fp16 = fabsf((float)weights_fp16[i]);
	}
	
	shmem_max_fp32[threadIdx.x] = max_fp32;
	shmem_max_fp16[threadIdx.x] = max_fp16;
	__syncthreads();
	
	for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			shmem_max_fp32[threadIdx.x] = fmaxf(shmem_max_fp32[threadIdx.x], shmem_max_fp32[threadIdx.x + s]);
			shmem_max_fp16[threadIdx.x] = fmaxf(shmem_max_fp16[threadIdx.x], shmem_max_fp16[threadIdx.x + s]);
		}
		__syncthreads();
	}
	
	if (threadIdx.x == 0) {
		partial_max_fp32[blockIdx.x] = shmem_max_fp32[0];
		partial_max_fp16[blockIdx.x] = shmem_max_fp16[0];
	}
}

template <typename T>
__global__ void adam_step(
	const uint32_t n_elements,
	const uint32_t n_matrix_weights,
	const float relative_weight_decay,
	const float absolute_weight_decay,
	const float weight_clipping_magnitude,
	const float loss_scale,
	float learning_rate,
	const float non_matrix_learning_rate_factor,
	const bool optimize_matrix_params,
	const bool optimize_non_matrix_params,
	const float beta1,
	const float beta2,
	const float epsilon,
	const float lower_lr_bound,
	const float upper_lr_bound,
	const float l2_reg,
	float* __restrict__ weights_full_precision,
	T* __restrict__ weights,
	const T* __restrict__ gradients,
	float* __restrict__ first_moments,
	float* __restrict__ second_moments,
	uint32_t* __restrict__ param_steps
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	float gradient = (float)gradients[i] / loss_scale;
	if (i >= n_matrix_weights) {
		if (!optimize_non_matrix_params || gradient == 0) {
			return;
		}
	} else {
		if (!optimize_matrix_params) {
			return;
		}
	}

	const float weight_fp = weights_full_precision[i];

	if (i < n_matrix_weights) {
		// No L2 reg for non-matrix params
		gradient += l2_reg * weight_fp;
	}

	const float gradient_sq = gradient * gradient;

	float first_moment = first_moments[i] = beta1 * first_moments[i] + (1 - beta1) * gradient;
	const float second_moment = second_moments[i] = beta2 * second_moments[i] + (1 - beta2) * gradient_sq;

	if (i >= n_matrix_weights) {
		// Potentially different learning rate for non-matrix params
		learning_rate *= non_matrix_learning_rate_factor;
	}

	// Debiasing. Since some parameters might see fewer steps than others, they each need their own step counter.
	const uint32_t current_step = ++param_steps[i];
	learning_rate *= sqrtf(1 - powf(beta2, (float)current_step)) / (1 - powf(beta1, (float)current_step));

	// Follow AdaBound paradigm
	const float effective_learning_rate = fminf(fmaxf(learning_rate / (sqrtf(second_moment) + epsilon), lower_lr_bound), upper_lr_bound);

	const float decayed_weight = weight_decay(relative_weight_decay * learning_rate, absolute_weight_decay * learning_rate, weight_fp);
	float new_weight = decayed_weight - effective_learning_rate * first_moment;

	if (weight_clipping_magnitude != 0.0f) {
		new_weight = clamp(new_weight, -weight_clipping_magnitude, weight_clipping_magnitude);
	}

	weights_full_precision[i] = new_weight;
	weights[i] = (T)new_weight;
}

template <typename T>
class AdamOptimizer : public Optimizer<T> {
public:
	AdamOptimizer(const json& params) {
		update_hyperparams(params);
	}

	void allocate(uint32_t n_weights, const std::vector<std::pair<uint32_t, uint32_t>>& layer_sizes) override {
		m_n_weights = n_weights;
		if (m_n_weights <= m_first_moments.size()) {
			return;
		}

		m_first_moments.resize(m_n_weights);
		m_first_moments.memset(0);

		m_second_moments.resize(m_n_weights);
		m_second_moments.memset(0);

		m_param_steps.resize(m_n_weights);
		m_param_steps.memset(0);

		m_n_weights_covered_by_matrices = 0;

		for (size_t i = 0; i < layer_sizes.size(); ++i) {
			m_n_weights_covered_by_matrices += layer_sizes[i].first * layer_sizes[i].second;
		}
	}

	void step(hipStream_t stream, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) override {
		++m_current_step;

		// v27 Diagnostic: Monitor gradient and weight statistics every 100 steps
		if (m_current_step % 100 == 0) {
			const uint32_t n_threads = 256;
			const uint32_t n_blocks = div_round_up(n_weights(), n_threads);
			
			// Allocate buffers on first use
			if (m_diagnostic_buffers.size() == 0) {
				m_diagnostic_buffers.resize(n_blocks * 2);  // For sums and maxes
				m_diagnostic_host.resize(n_blocks * 2);
			}
			
			// Compute gradient statistics
			linear_kernel(compute_gradient_stats<T>, 0, stream,
				n_weights(),
				gradients,
				loss_scale,
				m_diagnostic_buffers.data(),
				m_diagnostic_buffers.data() + n_blocks
			);
			
			// Compute weight statistics
			linear_kernel(compute_weight_stats<T>, 0, stream,
				n_weights(),
				weights_full_precision,
				weights,
				m_diagnostic_buffers.data(),
				m_diagnostic_buffers.data() + n_blocks
			);
			
			// Copy to host and compute final statistics
			hipStreamSynchronize(stream);
			hipMemcpy(m_diagnostic_host.data(), m_diagnostic_buffers.data(),
			          n_blocks * 2 * sizeof(float), hipMemcpyDeviceToHost);
			
			// Gradient norm and max
			float grad_sum_sq = 0.0f;
			float grad_max = 0.0f;
			for (uint32_t i = 0; i < n_blocks; ++i) {
				grad_sum_sq += m_diagnostic_host[i];
				grad_max = fmaxf(grad_max, m_diagnostic_host[i + n_blocks]);
			}
			float grad_norm = sqrtf(grad_sum_sq);
			
			// Weight max (FP32 and FP16)
			float weight_max_fp32 = 0.0f;
			float weight_max_fp16 = 0.0f;
			for (uint32_t i = 0; i < n_blocks; ++i) {
				weight_max_fp32 = fmaxf(weight_max_fp32, m_diagnostic_host[i]);
				weight_max_fp16 = fmaxf(weight_max_fp16, m_diagnostic_host[i + n_blocks]);
			}
			
			printf("[v27 Diagnostic] Step %u: grad_norm=%.4f, grad_max=%.4f, weight_max_fp32=%.4f, weight_max_fp16=%.4f\n",
			       m_current_step, grad_norm, grad_max, weight_max_fp32, weight_max_fp16);
		}

		float lower_lr_bound = 0;
		float upper_lr_bound = std::numeric_limits<float>::max();

		// AdaBound paper: https://openreview.net/pdf?id=Bkg3g2R9FX
		if (m_adabound) {
			lower_lr_bound = 0.1f - 0.1f / ((1 - m_beta2) * (float)step() + 1);
			upper_lr_bound = 0.1f + 0.1f / ((1 - m_beta2) * (float)step());
		}

		linear_kernel(adam_step<T>, 0, stream,
			n_weights(),
			m_n_weights_covered_by_matrices,
			m_relative_weight_decay,
			m_absolute_weight_decay,
			m_weight_clipping_magnitude,
			loss_scale,
			m_base_learning_rate,
			m_non_matrix_learning_rate_factor,
			m_optimize_matrix_params,
			m_optimize_non_matrix_params,
			m_beta1,
			m_beta2,
			m_epsilon,
			lower_lr_bound,
			upper_lr_bound,
			m_l2_reg,
			weights_full_precision,
			weights,
			gradients,
			m_first_moments.data(),
			m_second_moments.data(),
			m_param_steps.data()
		);
	}

	float learning_rate() const override {
		return m_base_learning_rate;
	}

	void set_learning_rate(float val) override {
		m_base_learning_rate = val;
	}

	uint32_t step() const override {
		return m_current_step;
	}

	uint32_t n_weights() const override {
		return m_n_weights;
	}

	T* custom_weights() const override {
		return nullptr;
	}

	void update_hyperparams(const json& params) override {
		if (params.contains("beta1")) {
			m_beta1 = params["beta1"];
		}

		if (params.contains("beta2")) {
			m_beta2 = params["beta2"];
		}

		if (params.contains("epsilon")) {
			m_epsilon = params["epsilon"];
		}

		if (params.contains("learning_rate")) {
			m_base_learning_rate = params["learning_rate"];
		}

		if (params.contains("l2_reg")) {
			m_l2_reg = params["l2_reg"];
		}

		if (params.contains("adabound")) {
			m_adabound = params["adabound"];
		}

		if (params.contains("relative_decay")) {
			m_relative_weight_decay = params["relative_decay"];
		}

		if (params.contains("absolute_decay")) {
			m_absolute_weight_decay = params["absolute_decay"];
		}

		if (params.contains("clipping_magnitude")) {
			m_weight_clipping_magnitude = params["clipping_magnitude"];
		}

		if (params.contains("non_matrix_learning_rate_factor")) {
			m_non_matrix_learning_rate_factor = params["non_matrix_learning_rate_factor"];
		}

		if (params.contains("optimize_matrix_params")) {
			m_optimize_matrix_params = params["optimize_matrix_params"];
		}

		if (params.contains("optimize_non_matrix_params")) {
			m_optimize_non_matrix_params = params["optimize_non_matrix_params"];
		}
	}

	json hyperparams() const override {
		return {
			{"otype", "Adam"},
			{"beta1", m_beta1},
			{"beta2", m_beta2},
			{"epsilon", m_epsilon},
			{"learning_rate", m_base_learning_rate},
			{"l2_reg", m_l2_reg},
			{"adabound", m_adabound},
			{"relative_decay", m_relative_weight_decay},
			{"absolute_decay", m_absolute_weight_decay},
			{"clipping_magnitude", m_weight_clipping_magnitude},
			{"non_matrix_learning_rate_factor", m_non_matrix_learning_rate_factor},
			{"optimize_matrix_params", m_optimize_matrix_params},
			{"optimize_non_matrix_params", m_optimize_non_matrix_params},
		};
	}

	json serialize() const override {
		json data;
		data["current_step"] = m_current_step;
		data["base_learning_rate"] = m_base_learning_rate;
		data["first_moments_binary"] = m_first_moments;
		data["second_moments_binary"] = m_second_moments;
		data["param_steps_binary"] = m_param_steps;
		return data;
	}

	void deserialize(const json& data) override {
		m_first_moments = data["first_moments_binary"];
		m_second_moments = data["second_moments_binary"];
		if (data.contains("param_steps_binary")) {
			m_param_steps = data["param_steps_binary"];
		} else {
			m_param_steps.resize(m_second_moments.size());
			m_param_steps.memset(0);
		}
		m_current_step = data["current_step"];
		m_base_learning_rate = data["base_learning_rate"];
	}

private:
	uint32_t m_n_weights;
	uint32_t m_n_weights_covered_by_matrices;

	GPUMemory<float> m_first_moments;
	GPUMemory<float> m_second_moments;
	GPUMemory<uint32_t> m_param_steps;

	// v27 Diagnostic: Buffers for monitoring
	GPUMemory<float> m_diagnostic_buffers;
	std::vector<float> m_diagnostic_host;

	uint32_t m_current_step = 0;

	// Hyperparameters
	float m_non_matrix_learning_rate_factor = 1.0f;
	float m_base_learning_rate = 1e-3f;
	float m_beta1 = 0.9f;
	float m_beta2 = 0.999f;
	float m_epsilon = 1e-8f;
	float m_l2_reg = 1e-8f;

	float m_relative_weight_decay = 0.0f;
	float m_absolute_weight_decay = 0.0f;
	float m_weight_clipping_magnitude = 0.0f;

	bool m_adabound = false;

	bool m_optimize_matrix_params = true;
	bool m_optimize_non_matrix_params = true;
};

}

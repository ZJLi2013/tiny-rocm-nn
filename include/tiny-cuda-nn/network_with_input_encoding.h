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

/** @file   network_with_input_encoding.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  A model that includes its encoding
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>

namespace tcnn {

/**
 * @brief 将输入编码与神经网络相结合的模型。
 *
 * 这个类是一个包装器，它将一个输入编码层 (tcnn::Encoding) 和一个后续的神经网络 (tcnn::Network) 串联起来。
 * 输入数据首先通过编码层进行特征提取或转换，然后将编码后的结果作为输入传递给神经网络。
 * 这种结构在许多现代神经网络应用中很常见，例如神经辐射场 (NeRF)。
 *
 * @tparam T 内部计算的精度类型 (例如 half, float)。
 */
template <typename T>
class NetworkWithInputEncoding : public Network<float, T> {
public:
	/**
	 * @brief 构造函数，使用预先创建的编码和网络对象。
	 * @param encoding 输入编码模块的共享指针。
	 * @param network  神经网络模块的共享指针。
	 */
	NetworkWithInputEncoding(std::shared_ptr<Encoding<T>> encoding, std::shared_ptr<Network<T>> network) : m_encoding{encoding}, m_network{network} {}

	/**
	 * @brief 构造函数，使用预先创建的编码对象和网络的JSON配置。
	 *
	 * 此构造函数会自动创建网络，并将网络的输入维度设置为编码的输出维度。
	 *
	 * @param encoding      输入编码模块的共享指针。
	 * @param n_output_dims 最终输出的维度。
	 * @param network       网络的JSON配置。
	 */
	NetworkWithInputEncoding(std::shared_ptr<Encoding<T>> encoding, uint32_t n_output_dims, const json& network) : m_encoding{encoding} {
		encoding->set_alignment(minimum_alignment(network));

		json local_network_config = network;
		local_network_config["n_input_dims"] = m_encoding->padded_output_width();
		local_network_config["n_output_dims"] = n_output_dims;
		m_network.reset(create_network<T>(local_network_config));
	}

	/**
	 * @brief 构造函数，使用编码和网络的JSON配置。
	 *
	 * 这是最常用的构造函数，它会根据JSON配置自动创建编码和网络两个模块。
	 *
	 * @param n_dims_to_encode 待编码的输入维度。
	 * @param n_output_dims    最终输出的维度。
	 * @param encoding         编码模块的JSON配置。
	 * @param network          神经网络模块的JSON配置。
	 */
	NetworkWithInputEncoding(uint32_t n_dims_to_encode, uint32_t n_output_dims, const json& encoding, const json& network)
	: NetworkWithInputEncoding{std::shared_ptr<Encoding<T>>{create_encoding<T>(n_dims_to_encode, encoding)}, n_output_dims, network} { }

	virtual ~NetworkWithInputEncoding() { }

	void inference_mixed_precision_impl(hipStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {
		GPUMatrixDynamic<T> network_input = {m_encoding->padded_output_width(), input.n(), stream, m_encoding->preferred_output_layout()};
		m_encoding->inference_mixed_precision(stream, input, network_input, use_inference_params);
		m_network->inference_mixed_precision(stream, network_input, output, use_inference_params);
	}

	uint32_t num_encoded_dims() const {
		return m_encoding->padded_output_width();
	}

	/**
	 * @brief 执行前向传播。
	 *
	 * 该函数按顺序执行两个步骤：
	 * 1. 输入数据 `input` 通过 `m_encoding` 编码层，生成中间特征 `network_input`。
	 * 2. 中间特征 `network_input` 通过 `m_network` 神经网络层，生成最终的输出。
	 *
	 * @param stream                CUDA流。
	 * @param input                 输入数据矩阵，精度为float。
	 * @param output                可选的输出矩阵指针。如果提供，将在此处写入结果。
	 * @param use_inference_params  是否使用为推理优化的参数。
	 * @param prepare_input_gradients 是否为输入梯度计算做准备。
	 * @return 返回一个包含前向传播上下文的unique_ptr，用于后续的反向传播。
	 */
	std::unique_ptr<Context> forward_impl(hipStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		// GPUMatrixDynamic<T> 是一个辅助类，用于表示和管理存储在GPU上的动态大小的矩阵。
		// "Dynamic" 意味着它的维度（宽度和高度）和内存布局（行优先或列优先）是在运行时确定的，而不是在编译时。
		// 它封装了指向GPU内存的指针，并提供了对矩阵属性（如维度、布局）的访问，简化了在CUDA核心函数之间传递矩阵数据的过程。
		// 在下面这行代码中，它被用来创建一个临时的GPU矩阵 `network_input`，
		// 用于存储编码层(m_encoding)的输出结果。这个结果随后将作为下一层神经网络(m_network)的输入。
		forward->network_input = GPUMatrixDynamic<T>{m_encoding->padded_output_width(), input.n(), stream, m_encoding->preferred_output_layout()};
		forward->encoding_ctx = m_encoding->forward(stream, input, &forward->network_input, use_inference_params, prepare_input_gradients);
		forward->network_ctx = m_network->forward(stream, forward->network_input, output, use_inference_params, true);

		return forward;
	}

	/**
	 * @brief 执行反向传播。
	 *
	 * 该函数按逆序执行两个步骤，计算梯度：
	 * 1. 首先，根据输出的梯度 `dL_doutput`，通过 `m_network` 计算网络参数的梯度，以及关于其输入（即编码层的输出）的梯度 `dL_dnetwork_input`。
	 * 2. 然后，如果需要（例如编码层有可训练参数，或需要计算关于最原始输入的梯度），
	 *    将 `dL_dnetwork_input` 通过 `m_encoding` 反向传播，计算编码参数的梯度，以及可选的关于最原始输入的梯度 `dL_dinput`。
	 *
	 * @param stream                CUDA流。
	 * @param ctx                   前向传播返回的上下文。
	 * @param input                 原始的输入数据矩阵。
	 * @param output                前向传播的输出矩阵。
	 * @param dL_doutput            损失函数关于输出的梯度。
	 * @param dL_dinput             可选的指针，用于存储损失函数关于原始输入的梯度。
	 * @param use_inference_params  是否使用为推理优化的参数。
	 * @param param_gradients_mode  参数梯度的更新模式（覆盖、累加等）。
	 */
	void backward_impl(
		hipStream_t stream,
		const Context& ctx,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite
	) override {
		GPUMatrixDynamic<T> dL_dnetwork_input;
		if (m_encoding->n_params() > 0 || dL_dinput) {
			dL_dnetwork_input = {m_encoding->padded_output_width(), input.n(), stream, m_encoding->preferred_output_layout()};
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		m_network->backward(stream, *forward.network_ctx, forward.network_input, output, dL_doutput, dL_dnetwork_input.data() ? &dL_dnetwork_input : nullptr, use_inference_params, param_gradients_mode);
		if (dL_dnetwork_input.data()) {
			m_encoding->backward(
				stream,
				*forward.encoding_ctx,
				input,
				forward.network_input,
				dL_dnetwork_input,
				dL_dinput,
				use_inference_params,
				param_gradients_mode
			);
		}
	}

	void set_params_impl(T* params, T* inference_params, T* gradients) override {
		size_t offset = 0;
		m_network->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_network->n_params();

		m_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_encoding->n_params();
	}

	void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
		m_network->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_network->n_params();

		m_encoding->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_encoding->n_params();
	}

	size_t n_params() const override {
		return m_encoding->n_params() + m_network->n_params();
	}

	uint32_t padded_output_width() const override {
		return m_network->padded_output_width();
	}

	uint32_t output_width() const override {
		return m_network->output_width();
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required due to encoding
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		return m_network->layer_sizes();
	}

	uint32_t width(uint32_t layer) const override {
		return layer == 0 ? m_encoding->padded_output_width() : m_network->width(layer - 1);
	}

	uint32_t num_forward_activations() const override {
		return m_network->num_forward_activations() + 1;
	}

	std::pair<const T*, MatrixLayout> forward_activations(const Context& ctx, uint32_t layer) const override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		return layer == 0 ? std::make_pair<const T*, MatrixLayout>(forward.network_input.data(), m_encoding->preferred_output_layout()) : m_network->forward_activations(*forward.network_ctx, layer - 1);
	}

	uint32_t input_width() const override {
		return m_encoding->input_width();
	}

	const std::shared_ptr<Encoding<T>>& encoding() const {
		return m_encoding;
	}

	json hyperparams() const override {
		return {
			{"otype", "NetworkWithInputEncoding"},
			{"encoding", m_encoding->hyperparams()},
			{"network", m_network->hyperparams()},
		};
	}

private:
	std::shared_ptr<Encoding<T>> m_encoding;
	std::shared_ptr<Network<T>> m_network;

	struct ForwardContext : public Context {
		GPUMatrixDynamic<T> network_input;
		std::unique_ptr<Context> encoding_ctx;
		std::unique_ptr<Context> network_ctx;
	};
};

}

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

/** @file   trainer.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Class that performs training of a differentiable cuda object, given an optimizer and a loss.
 */

#pragma once

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/cuda_graph.h>
#include <tiny-cuda-nn/gpu_memory_json.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/object.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/reduce_sum.h>

#include <random>

#include <tiny-cuda-nn/debug_config.h>

namespace tcnn {

template <typename T, typename PARAMS_T, typename COMPUTE_T=T>
class Trainer : public ObjectWithMutableHyperparams {
public:
	Trainer(std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> model, std::shared_ptr<Optimizer<PARAMS_T>> optimizer, std::shared_ptr<Loss<COMPUTE_T>> loss, uint32_t seed = 1337, float perturbation_sigma = 0)
	: m_model{model}, m_optimizer{optimizer}, m_loss{loss}, m_perturbation_sigma{perturbation_sigma} {
		std::seed_seq seq{seed};
		std::vector<uint32_t> seeds(2);
		seq.generate(std::begin(seeds), std::end(seeds));
		m_rng = pcg32{seeds.front()};
		initialize_params();
	}

	virtual ~Trainer() {}

	void set_loss(std::shared_ptr<Loss<COMPUTE_T>> loss) {
		if (!loss) {
			throw std::runtime_error{"Trainer: may not set loss to nullptr"};
		}
		m_loss = loss;
	}

	void initialize_params() {
		size_t n_params = m_model->n_params();
		log_debug("Trainer: initializing {} params and resetting training.", n_params);

		// Allocate auxiliary optimizer buffers
		m_optimizer->allocate(m_model);

		m_params_buffer.resize(sizeof(PARAMS_T) * n_params * 2 + sizeof(float) * n_params * 1);
		m_params_buffer.memset(0);

		reset_param_pointers();

		m_model->initialize_params(m_rng, m_params_full_precision);

		// initialize_params is only expected to initialize m_params_full_precision. Cast and copy these over!
		parallel_for_gpu(n_params, [params_fp=m_params_full_precision, params=m_params] __device__ (size_t i) {
			params[i] = (PARAMS_T)params_fp[i];
		});
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	// ForwardContext 结构体用于存储前向传播过程中的中间结果，这些结果在后续的反向传播和损失计算中是必需的。
	// 它继承自 tcnn::Context，作为一个通用的上下文句柄。
	struct ForwardContext : public Context {
		// `perturbed_output`: 存储添加了随机扰动后的网络输出。
		// 当训练时设置了 `m_perturbation_sigma > 0`，会在原始输出上增加噪声。
		// 这是一种正则化技术，可以提高训练的鲁棒性。损失函数将基于这个被扰动的输出来计算。
		// 如果没有设置扰动，此张量可能未使用。
		GPUMatrix<COMPUTE_T> perturbed_output;

		// `output` 是神经网络前向计算的结果，即模型的预测值，称之为 y_pred。
		GPUMatrix<COMPUTE_T> output;

		// 损失函数 L 会比较这个预测值 `output` 和真实目标 `target`，例如 L = f(y_pred, target)。
		// `dL_doutput` 是损失函数 L 相对于网络输出 `output` 的偏导数（梯度），即 ∂L/∂y_pred。
		// 在训练过程中，`output` 是前向传播的终点，而 `dL_doutput` 则是反向传播的起点。
		GPUMatrix<COMPUTE_T> dL_doutput;

		// `L`: 存储每个训练样本的损失值。
		// 这是一个矩阵，其元素对应批处理中每个样本计算出的损失。对该矩阵所有元素求和可以得到整个批次的总体损失。
		GPUMatrix<float> L;

		// `model_ctx`: 存储底层模型（`m_model`）在前向传播过程中产生的内部上下文。
		// 这通常包含了模型中间层的激活值等信息，是底层模型执行其自身反向传播（`m_model->backward()`）所必需的数据。
		// Trainer 作为一个高级包装器，需要保存并传递这个底层上下文。
		std::unique_ptr<Context> model_ctx;
	};

	// 执行一次完整的前向传播过程。
	// 这个函数负责：
	// 1. 调用底层模型(m_model)的 `forward` 方法，得到网络的原始输出 `output`。
	// 2. (可选) 如果 `m_perturbation_sigma` > 0，在 `output` 上添加随机噪声，作为一种正则化手段。默认情况下，`m_perturbation_sigma` 为0，此步骤被跳过。
	// 3. 计算损失和梯度。这里有两种模式：
	//    a) 如果提供了 `external_dL_dy` (外部传入的梯度)，则直接使用它作为反向传播的起始梯度 `dL_doutput`。
	//       `external_dL_dy` (dL/dy) 是一个高级用法，允许将 tiny-cuda-nn 集成到更大的计算图中（例如 PyTorch），
	//       其中损失函数的计算和其对输出 `y` 的梯度是在框架外部完成的。
	//    b) 如果 `external_dL_dy` 为 nullptr (默认情况)，则使用内部的损失函数 `m_loss` 来计算损失值 `L` 和关于输出的梯度 `dL_doutput`。
	//
	// @return 返回一个 `ForwardContext` 结构体，其中包含了反向传播所需的所有中间变量。
	std::unique_ptr<ForwardContext> forward(
		cudaStream_t stream,
		const float loss_scale,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrix<float>& target,
		const GPUMatrix<float>* data_pdf = nullptr,
		bool use_inference_params = false,
		bool prepare_input_gradients = false,
		const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr
	) {
		const uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->output = GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size, stream};

		// 这里的 `m_model->forward` 调用同样遵循虚函数分派链，与 `backward` 类似。
		// 调用路径:
		// 1. `m_model` 指向一个 `NetworkWithInputEncoding` 对象。
		// 2. 调用 `DifferentiableObject::forward()` (位于 object.h), 该公有方法内部调用虚方法 `forward_impl()`。
		// 3. 实际执行的是 `NetworkWithInputEncoding::forward_impl()`。
		// 4. 在 `NetworkWithInputEncoding` 内部，它会先调用 `m_encoding->forward()`，然后调用 `m_network->forward()`。
		// 5. `m_network` 指向一个具体的网络实例 (例如 `FullyFusedMLP`)。
		// 6. 这个调用最终会到达具体网络 (如 `FullyFusedMLP`) 的 `forward_impl()`，执行前向传播的CUDA核函数。
		// 7. `forward_impl` 会返回一个包含中间结果（如各层激活值）的上下文 `ctx`，这里被赋值给 `forward->model_ctx` 以便反向传播时使用。
		forward->model_ctx = m_model->forward(stream, input, &forward->output, use_inference_params, prepare_input_gradients);

		if (m_perturbation_sigma > 0) {
			GPUMatrix<float> perturbation{m_model->padded_output_width(), batch_size, stream};
			forward->perturbed_output = GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size, stream};

			const uint32_t n_elements = perturbation.n_elements();
			generate_random_logistic<float>(stream, m_rng, n_elements, perturbation.data(), 0.0f, m_perturbation_sigma);
			add<<<n_blocks_linear(n_elements), N_THREADS_LINEAR, 0, stream>>>(n_elements, forward->output.data(), perturbation.data(), forward->perturbed_output.data());
		}

		auto& loss_input = m_perturbation_sigma > 0 ? forward->perturbed_output : forward->output;

		forward->L = GPUMatrix<float>{m_model->padded_output_width(), batch_size, stream};

		if (external_dL_dy) {
			CHECK_THROW(external_dL_dy->m() == m_model->padded_output_width());
			CHECK_THROW(external_dL_dy->n() == batch_size);

			forward->dL_doutput = GPUMatrix<COMPUTE_T>{external_dL_dy->data(), m_model->padded_output_width(), batch_size};
		} else {
			CHECK_THROW(input.n() == target.n());
			CHECK_THROW(m_model->output_width() == target.m());

			forward->dL_doutput = GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size, stream};

			// 调用损失函数 m_loss 的 evaluate 方法来计算损失以及反向传播的起始梯度。
			// 参数解析:
			// 1. loss_scale: 损失缩放因子。在混合精度训练中，为了防止半精度（half）浮点数的梯度下溢（变为0），
			//    通常会将损失值乘以一个较大的系数（如 65536.0），从而将梯度也相应放大。在优化器更新权重之前，会再将梯度缩放回来。
			//
			// 2. loss_input: 用于计算损失的模型输出。它就是模型的预测值（`forward->output`），如果开启了扰动，则是加了噪声的 `perturbed_output`。
			//
			// 3. target: 训练数据的真值（Ground Truth）。损失函数会比较 `loss_input` 和 `target` 的差异。
			//
			// 4. forward->L, forward->dL_doutput: 这两个是输出参数。它们是指向预先分配好形状的GPU内存的指针。
			//    `evaluate` 函数会将计算出的逐样本损失值写入 `forward->L`，并将损失关于 `loss_input` 的梯度（∂L/∂y）写入 `forward->dL_doutput`。
			//    所以你的理解“传入的是空gpu buffer ptr，但给定 tensor shape”是正确的。
			//
			// 5. data_pdf: (可选) 数据点的概率密度函数（Probability Density Function）。在一些非均匀采样策略中（如 NeRF 的重要性采样），
			//    每个训练样本的重要性是不同的。`data_pdf` 提供了每个样本的采样概率，损失函数可以利用它来对每个样本的损失进行加权，
			//    以得到一个无偏的估计。如果不需要，则为 nullptr。
			m_loss->evaluate(stream, loss_scale, loss_input, target, forward->L, forward->dL_doutput, data_pdf);
		}

		return forward;
	}

	std::unique_ptr<ForwardContext> forward(const float loss_scale, const GPUMatrixDynamic<T>& input, const GPUMatrix<float>& target, const GPUMatrix<float>* data_pdf = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false, const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr) {
		return forward(nullptr, loss_scale, input, target, data_pdf, use_inference_params, prepare_input_gradients, external_dL_dy);
	}

	// 执行反向传播过程。
	// 这个函数是 Trainer 类中反向传播的入口点。它本身不做复杂的计算，
	// 而是作为一个调度器，调用底层模型（m_model）的 `backward` 方法来真正执行计算。
	//
	// 工作流程:
	// 1. 从前向传播的上下文 `ctx` 中提取所需的所有信息，例如：
	//    - `ctx.model_ctx`: 底层模型自身的上下文，可能包含中间层的激活值。
	//    - `ctx.output`: 前向传播时模型的输出。
	//    - `ctx.dL_doutput`: 损失函数关于模型输出的梯度，这是反向传播的起点。
	// 2. 将这些信息以及其他参数（如原始输入 `input`）传递给 `m_model->backward()`。
	// 3. 底层模型（例如 `NetworkWithInputEncoding`）会利用这些信息，根据链式法则计算其可训练参数的梯度，
	//    并将这些梯度存储在 `m_param_gradients` 指向的内存中。
	// 4. (可选) 如果 `dL_dinput` 不是 nullptr，模型还会计算并传出损失关于最开始输入的梯度。
	void backward(cudaStream_t stream, const ForwardContext& ctx, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* dL_dinput = nullptr, bool use_inference_params = false, GradientMode param_gradients_mode = GradientMode::Overwrite) {
		// 调用路径如下:
		// 1. `m_model` 是一个 `DifferentiableObject` 指针，在我们的例子中，它指向一个 `NetworkWithInputEncoding` 对象。
		// 2. 对 `m_model->backward()` 的调用首先进入 `DifferentiableObject` 类（在 object.h 中）的 `backward` 公有非虚方法。
		// 3. 这个公有方法内部会调用虚方法 `backward_impl()`。
		// 4. 由于多态，实际执行的是 `NetworkWithInputEncoding::backward_impl()`。
		// 5. 在 `NetworkWithInputEncoding::backward_impl()` 内部，它又会调用其持有的 `m_network` 成员的 `backward()` 方法。
		// 6. `m_network` 指向的是根据JSON配置创建的具体网络实例，例如 `FullyFusedMLP`。
		// 7. 因此，调用链最终到达 `FullyFusedMLP::backward_impl()`，从而执行其高度优化的反向传播CUDA核函数。
		m_model->backward(stream, *ctx.model_ctx, input, ctx.output, ctx.dL_doutput, dL_dinput, use_inference_params, param_gradients_mode);
	}

	void backward(const ForwardContext& ctx, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* dL_dinput = nullptr, bool use_inference_params = false, GradientMode param_gradients_mode = GradientMode::Overwrite) {
		backward(nullptr, ctx, input, dL_dinput, use_inference_params, param_gradients_mode);
	}

	void optimizer_step(cudaStream_t stream, float loss_scale) {
		m_optimizer->step(stream, loss_scale, m_params_full_precision, m_params, m_param_gradients);
	}

	void optimizer_step(float loss_scale) {
		optimizer_step(nullptr, loss_scale);
	}

	// 执行一个完整的训练步骤，包括前向传播、反向传播和（可选的）优化器更新。
	std::unique_ptr<ForwardContext> training_step(
		cudaStream_t stream,
		const GPUMatrixDynamic<T>& input,
		const GPUMatrix<float>& target,
		const GPUMatrix<float>* data_pdf = nullptr,
		bool run_optimizer = true,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite,
		const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr
	) {
		const float loss_scale = default_loss_scale<PARAMS_T>();

		// Execute forward and backward in a CUDA graph for maximum performance.
		std::unique_ptr<ForwardContext> ctx;
		{
			// Execute forward and backward in a CUDA graph for maximum performance.
			ctx = forward(stream, loss_scale, input, target, data_pdf, use_inference_params, dL_dinput, external_dL_dy);
			backward(stream, *ctx, input, dL_dinput, use_inference_params, param_gradients_mode);
		}

		if (run_optimizer) {
			optimizer_step(stream, loss_scale);
		}

		return ctx;
	}

	std::unique_ptr<ForwardContext> training_step(
		const GPUMatrixDynamic<T>& input,
		const GPUMatrix<float>& target,
		const GPUMatrix<float>* data_pdf = nullptr,
		bool run_optimizer = true,
		GPUMatrixDynamic<T>* dL_dinput = nullptr,
		bool use_inference_params = false,
		GradientMode param_gradients_mode = GradientMode::Overwrite,
		const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr
	) {
		return training_step(nullptr, input, target, data_pdf, run_optimizer, dL_dinput, use_inference_params, param_gradients_mode, external_dL_dy);
	}

	float loss(cudaStream_t stream, const ForwardContext& ctx) const {
		return reduce_sum(ctx.L.data(), ctx.L.n_elements(), stream);
	}

	float loss(const ForwardContext& ctx) const {
		return loss(nullptr, ctx);
	}

	void update_hyperparams(const json& params) override {
		m_optimizer->update_hyperparams(params.value("optimizer", json::object()));
		m_loss->update_hyperparams(params.value("loss", json::object()));
	}

	json hyperparams() const override {
		return {
			{"otype", "Trainer"},
			{"optimizer", m_optimizer->hyperparams()},
			{"loss", m_loss->hyperparams()},
		};
	}

	float* params_full_precision() const {
		return m_params_full_precision;
	}

	PARAMS_T* params() const {
		return m_params;
	}

	PARAMS_T* params_inference() const {
		return m_params_inference;
	}

	PARAMS_T* param_gradients() const {
		return m_param_gradients;
	}

	void set_params_full_precision(const float* params, size_t n_params, bool device_ptr = false) {
		if (n_params != m_model->n_params()) {
			throw std::runtime_error{"Can't set fp params because buffer has the wrong size."};
		}
		CUDA_CHECK_THROW(cudaMemcpy(m_params_full_precision, params, sizeof(float)*n_params, device_ptr ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));

		parallel_for_gpu(n_params, [params_fp=m_params_full_precision, params_inference=m_params_inference] __device__ (size_t i) {
			params_inference[i] = (PARAMS_T)params_fp[i];
		});

		CUDA_CHECK_THROW(cudaMemcpy(m_params, m_params_inference, sizeof(PARAMS_T)*n_params, cudaMemcpyDeviceToDevice));
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	void set_params(const PARAMS_T* params, size_t n_params, bool device_ptr = false) {
		if (n_params != m_model->n_params()) {
			throw std::runtime_error{"Can't set params because buffer has the wrong size."};
		}

		CUDA_CHECK_THROW(cudaMemcpy(m_params_inference, params, sizeof(PARAMS_T)*n_params, device_ptr ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
		CUDA_CHECK_THROW(cudaMemcpy(m_params, m_params_inference, sizeof(PARAMS_T)*n_params, cudaMemcpyDeviceToDevice));

		parallel_for_gpu(n_params, [params_fp=m_params_full_precision, params_inference=m_params_inference] __device__ (size_t i) {
			params_fp[i] = (float)params_inference[i];
		});

		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> model() {
		return m_model;
	}

	json serialize(bool serialize_optimizer = false) {
		size_t n_params = m_model->n_params();

		json data;
		data["n_params"] = n_params;
		data["params_type"] = type_to_string<PARAMS_T>();
		data["params_binary"] = gpu_memory_to_json_binary(m_params_inference, sizeof(PARAMS_T)*n_params);

		if (serialize_optimizer) {
			data["optimizer"] = m_optimizer->serialize();
		}

		return data;
	}

	void deserialize(const json& data) {
		std::string type = data.value("params_type", type_to_string<PARAMS_T>());
		if (type == "float") {
			GPUMemory<float> params = data["params_binary"];
			set_params_full_precision(params.data(), params.size(), true);
		} else if (type == "__half") {
			GPUMemory<__half> params_hp = data["params_binary"];
			size_t n_params = params_hp.size();

			GPUMemory<PARAMS_T> params(n_params);
			parallel_for_gpu(n_params, [params=params.data(), params_hp=params_hp.data()] __device__ (size_t i) {
				params[i] = (PARAMS_T)params_hp[i];
			});

			set_params(params.data(), params.size(), true);
		} else {
			throw std::runtime_error{"Trainer: snapshot parameters must be of type float of __half"};
		}

		if (data.contains("optimizer")) {
			m_optimizer->deserialize(data["optimizer"]);
		}

		reset_param_pointers();
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	void set_param_gradients_pointer(PARAMS_T* gradients) {
		reset_param_pointers();
		m_model->set_params(m_params, m_params_inference, gradients);
	}

	void reset_param_pointers() {
		size_t n_params = m_model->n_params();

		m_params_full_precision = (float*)(m_params_buffer.data());
		m_params                = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params);
		m_param_gradients       = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params + sizeof(PARAMS_T) * n_params);

		// Use the optimizer's custom params for inference, if they exist.
		m_params_inference = m_optimizer ? m_optimizer->custom_weights() : nullptr;
		if (m_params_inference == nullptr) {
			m_params_inference = m_params;
		}

		m_model->set_params(m_params, m_params_inference, m_param_gradients);
	}

	size_t n_params() const {
		return m_model->n_params();
	}

private:
	std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> m_model;
	std::shared_ptr<Optimizer<PARAMS_T>> m_optimizer;
	std::shared_ptr<Loss<COMPUTE_T>> m_loss;

	CudaGraph m_graph;

	GPUMemory<char> m_params_buffer;

	float* m_params_full_precision = nullptr;
	PARAMS_T* m_params_inference = nullptr;
	PARAMS_T* m_params = nullptr;
	PARAMS_T* m_param_gradients = nullptr;

	float m_perturbation_sigma;

	std::unique_ptr<Context> m_training_ctx;

	pcg32 m_rng;
};

}

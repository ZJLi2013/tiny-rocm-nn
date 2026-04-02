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

/** @file   network.cu
 *  @author Thomas Müller, NVIDIA
 *  @brief  API interface of a neural network implementation
 */

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/network.h>

#if TCNN_MIN_GPU_ARCH > 70
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>
#endif

namespace tcnn {

template <typename T>
void extract_dimension_pos_neg(hipStream_t stream, const uint32_t num_elements, const uint32_t dim, const uint32_t fan_in, const uint32_t fan_out, const T* encoded, MatrixLayout layout, float* output) {
	linear_kernel(extract_dimension_pos_neg_kernel<T>, 0, stream, num_elements, dim, fan_in, fan_out, encoded, layout, output);
}

template void extract_dimension_pos_neg(hipStream_t stream, const uint32_t num_elements, const uint32_t dim, const uint32_t fan_in, const uint32_t fan_out, const network_precision_t* encoded, MatrixLayout layout, float* output);

std::string select_network(const json& network) {
	std::string otype = network.value("otype", "MLP");
	bool want_fully_fused_mlp = equals_case_insensitive(otype, "MegakernelMLP") || equals_case_insensitive(otype, "FullyFusedMLP") || equals_case_insensitive(otype, "MLP");

#ifndef __HIP_PLATFORM_AMD__
	// NVIDIA-specific architecture check (AMD GPUs use different architecture numbering)
	if (MIN_GPU_ARCH <= 70 || std::is_same<network_precision_t, float>::value) {
		if (want_fully_fused_mlp && MIN_GPU_ARCH <= 70) {
			throw std::runtime_error{fmt::format(
				"FullyFusedMLP is not supported for GPU architecture {}. "
				"Requires architecture 75+.",
				MIN_GPU_ARCH
			)};
		}
		if (want_fully_fused_mlp && std::is_same<network_precision_t, float>::value) {
			throw std::runtime_error{"FullyFusedMLP requires half precision (__half), not float."};
		}
	}
#else
	// AMD GPUs: only check precision requirement
	if (want_fully_fused_mlp && std::is_same<network_precision_t, float>::value) {
		throw std::runtime_error{"FullyFusedMLP requires half precision (__half), not float."};
	}
#endif

	if (want_fully_fused_mlp) {
		return "FullyFusedMLP";
	}
	
	return otype;
}

uint32_t minimum_alignment(const json& network) {
	std::string network_type = select_network(network);

	if (equals_case_insensitive(network_type, "FullyFusedMLP")) {
#if TCNN_MIN_GPU_ARCH > 70
		uint32_t n_neurons = network.value("n_neurons", 128u);
		// Return the network width so encoding output is padded to match.
		// The fused backward kernel's dL_dinput path requires input_width == WIDTH;
		// the fc_multiply fallback for mismatched widths has issues on ROCm/hipBLAS.
		switch (n_neurons) {
			case  16: return 16;
			case  32: return 32;
			case  64: return 64;
			case 128: return 128;
			default: throw std::runtime_error{fmt::format("FullyFusedMLP only supports 16, 32, 64, and 128 neurons, but got {}.", n_neurons)};
		}
#else
		throw std::runtime_error{"FullyFusedMLP was not compiled due to insufficient GPU arch of <=70."};
#endif
	}
	
	throw std::runtime_error{fmt::format("Unsupported network type: {}", network_type)};
}

template <typename T>
Network<T>* create_network(const json& network) {
	std::string network_type = select_network(network);

	if (equals_case_insensitive(network_type, "FullyFusedMLP")) {
		if (!std::is_same<network_precision_t, __half>::value) {
			throw std::runtime_error{"FullyFusedMLP can only be used if the network precision is set to __half."};
		} else {
#if TCNN_MIN_GPU_ARCH > 70
#  define TCNN_FULLY_FUSED_PARAMS \
	network["n_input_dims"], \
	network["n_output_dims"], \
	network.value("n_hidden_layers", 5u), \
	string_to_activation(network.value("activation", "ReLU")), \
	string_to_activation(network.value("output_activation", "None")),

			uint32_t n_neurons = network.value("n_neurons", 128u);
			switch (n_neurons) {
				case  16: return new FullyFusedMLP<T,  16>{TCNN_FULLY_FUSED_PARAMS};
				case  32: return new FullyFusedMLP<T,  32>{TCNN_FULLY_FUSED_PARAMS};
				case  64: return new FullyFusedMLP<T,  64>{TCNN_FULLY_FUSED_PARAMS};
				case 128: return new FullyFusedMLP<T, 128>{TCNN_FULLY_FUSED_PARAMS};
				default: throw std::runtime_error{fmt::format("FullyFusedMLP only supports 16, 32, 64, and 128 neurons, but got {}.", n_neurons)};
			}
#  undef TCNN_FULLY_FUSED_PARAMS
#else //TCNN_MIN_GPU_ARCH > 70
			throw std::runtime_error{"FullyFusedMLP was not compiled due to insufficient GPU arch of <=70."};
#endif //TCNN_MIN_GPU_ARCH > 70
		}
	}

	throw std::runtime_error{fmt::format("Unsupported network type: {}", network_type)};
}

template Network<network_precision_t>* create_network(const json& network);

std::vector<std::string> builtin_networks() {
	return {
		"FullyFusedMLP",
	};
}
}

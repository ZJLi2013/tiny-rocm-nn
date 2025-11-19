#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/cublas_matmul.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <stdexcept>

using namespace tcnn;

static void print_sample(const char* name, const std::vector<__half>& v, size_t m, size_t n, size_t max_rows=2, size_t max_cols=8) {
	printf("%s [%zux%zu]:\n", name, m, n);
	for (size_t i = 0; i < std::min(max_rows, m); ++i) {
		for (size_t j = 0; j < std::min(max_cols, n); ++j) {
			printf("%.4f ", __half2float(v[i + j * m]));
		}
		printf("\n");
	}
}

int main(int argc, char** argv) {
	try {
		// Test hyper-params
		const uint32_t WIDTH = 64;
		const uint32_t BATCH = 256; // must be multiple of 256 for fused kernel (N_ITERS=8 for WIDTH<=128)
		const uint32_t N_HIDDEN_LAYERS = 2;
		const Activation H_ACT = Activation::ReLU;

		std::cout << "=== fused vs unfused (act + gemm) sanity test ===\n";
		std::cout << "WIDTH=" << WIDTH << " BATCH=" << BATCH << " HIDDEN=" << N_HIDDEN_LAYERS << " ACT=ReLU\n";

		// Network: input_width=WIDTH, output_width=WIDTH so last layer path (out>16) is not executed inside fused kernel
		FullyFusedMLP<__half, WIDTH> net(WIDTH, WIDTH, N_HIDDEN_LAYERS, H_ACT, Activation::None);

		// Allocate params
		GPUMemory<__half> w_train(net.n_params());
		GPUMemory<__half> w_infer(net.n_params());
		GPUMemory<__half> grads(net.n_params());
		net.set_params(w_train.data(), w_infer.data(), grads.data());

		// Initialize weights in FP32 then convert to FP16
		GPUMemory<float> w_fp32(net.n_params());
		pcg32 rng(1337);
		net.initialize_params(rng, w_fp32.data(), 1.0f);

		std::vector<float> h_w_fp32(net.n_params());
		std::vector<__half> h_w_fp16(net.n_params());
		CUDA_CHECK_THROW(hipMemcpy(h_w_fp32.data(), w_fp32.data(), net.n_params()*sizeof(float), hipMemcpyDeviceToHost));
		for (size_t i = 0; i < h_w_fp16.size(); ++i) {
			h_w_fp16[i] = __float2half(h_w_fp32[i]);
		}
		CUDA_CHECK_THROW(hipMemcpy(w_train.data(),  h_w_fp16.data(), net.n_params()*sizeof(__half), hipMemcpyHostToDevice));
		CUDA_CHECK_THROW(hipMemcpy(w_infer.data(),  h_w_fp16.data(), net.n_params()*sizeof(__half), hipMemcpyHostToDevice));

		// Inputs: use CM layout to match hidden layers' default
		GPUMatrixDynamic<__half> input(WIDTH, BATCH, CM);
		input.initialize_xavier_uniform(rng);

		// --- 1) Fused path ---
		auto fwd_ctx = net.forward(hipStreamDefault, input, nullptr, /*use_inference=*/false, /*prepare_input_grad*/false);
		CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));

		// Take last hidden layer output (post-activation)
		const uint32_t last_idx = net.num_forward_activations() - 1;
		auto [fused_ptr, fused_layout] = net.forward_activations(*fwd_ctx, last_idx);
		if (fused_layout != CM) {
			std::cerr << "Unexpected fused hidden layout. Expected CM, got RM.\n";
			// Still proceed treating memory as CM view
		}
		GPUMatrix<__half, CM> fused_last(const_cast<__half*>(fused_ptr), WIDTH, BATCH);

		// --- 2) Unfused path (hipBLAS GEMM + activation; NO transpose of weights) ---
		// Weight tensors from the network
		auto& W0 = net.input_weight_matrix(false);   // [WIDTH, WIDTH] RM
		auto& W1 = net.weight_matrix_at(false, 0);   // [WIDTH, WIDTH] RM

		// Intermediates (CM to match fused)
		GPUMatrix<__half, CM> unfused_l0(WIDTH, BATCH);
		GPUMatrix<__half, CM> unfused_l1(WIDTH, BATCH);

		// Layer 0: D = W0 * input
		// fc_multiply handles mixed layouts and computes in FP32 accumulate then applies activation in a separate kernel.
		fc_multiply(hipStreamDefault, W0, input.cm(), unfused_l0, H_ACT);
		CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));

		// Layer 1: D = W1 * unfused_l0
		fc_multiply(hipStreamDefault, W1, unfused_l0, unfused_l1, H_ACT);
		CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));

		// --- 3) Compare fused vs unfused (element-wise) ---
		std::vector<__half> h_fused(WIDTH * BATCH);
		std::vector<__half> h_unfused(WIDTH * BATCH);
		CUDA_CHECK_THROW(hipMemcpy(h_fused.data(), fused_last.data(),  h_fused.size()*sizeof(__half), hipMemcpyDeviceToHost));
		CUDA_CHECK_THROW(hipMemcpy(h_unfused.data(), unfused_l1.data(), h_unfused.size()*sizeof(__half), hipMemcpyDeviceToHost));

		size_t mismatches = 0;
		float max_diff = 0.0f;
		size_t max_diff_idx = 0;
		const float eps = 1e-2f;

		for (size_t i = 0; i < h_fused.size(); ++i) {
			const float a = __half2float(h_fused[i]);
			const float b = __half2float(h_unfused[i]);
			const float d = std::fabs(a - b);
			if (d > eps) {
				++mismatches;
				if (d > max_diff) {
					max_diff = d;
					max_diff_idx = i;
				}
			}
		}

		if (mismatches == 0) {
			std::cout << "✓ SUCCESS: fused vs unfused outputs match within eps=" << eps << "\n";
			return 0;
		}

		std::cerr << "✗ FAILURE: fused vs unfused differ. mismatches=" << mismatches
		          << "/" << h_fused.size() << " max_diff=" << max_diff
		          << " at idx=" << max_diff_idx << "\n";

		// Print small samples for quick inspection
		print_sample("Fused (CM)",   h_fused,   WIDTH, BATCH);
		print_sample("Unfused (CM)", h_unfused, WIDTH, BATCH);

		return 1;
	} catch (const std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
		return 2;
	}
}

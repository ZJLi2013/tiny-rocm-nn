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
#include <algorithm>

using namespace tcnn;

static void print_sample(const char *name, const std::vector<__half> &v, size_t m, size_t n, size_t max_rows = 2, size_t max_cols = 8)
{
	printf("%s [%zux%zu]:\n", name, m, n);
	for (size_t i = 0; i < std::min(max_rows, m); ++i)
	{
		for (size_t j = 0; j < std::min(max_cols, n); ++j)
		{
			printf("%.4f ", __half2float(v[i + j * m]));
		}
		printf("\n");
	}
}

struct DiffStats
{
	size_t mismatches = 0;
	float max_diff = 0.0f;
	size_t max_diff_idx = 0;
};

static DiffStats diff_stats_cm(const std::vector<__half> &a, const std::vector<__half> &b, float eps)
{
	if (a.size() != b.size())
	{
		throw std::runtime_error("diff_stats_cm: size mismatch");
	}

	DiffStats s{};
	for (size_t i = 0; i < a.size(); ++i)
	{
		const float va = __half2float(a[i]);
		const float vb = __half2float(b[i]);
		const float d = std::fabs(va - vb);
		if (d > eps)
		{
			++s.mismatches;
			if (d > s.max_diff)
			{
				s.max_diff = d;
				s.max_diff_idx = i;
			}
		}
	}
	return s;
}

static void print_patch_cm(const char *name, const std::vector<__half> &v, size_t m, size_t n, size_t center_r, size_t center_c, size_t patch = 4)
{
	// v is CM: idx = r + c*m
	const int half = (int)patch / 2; // patch=4 => half=2
	const int r0 = std::max(0, (int)center_r - half);
	const int c0 = std::max(0, (int)center_c - half);
	const int r1 = std::min((int)m, r0 + (int)patch);
	const int c1 = std::min((int)n, c0 + (int)patch);

	printf("%s patch (CM) center=(r=%zu,c=%zu), rows=[%d,%d), cols=[%d,%d):\n", name, center_r, center_c, r0, r1, c0, c1);
	for (int r = r0; r < r1; ++r)
	{
		for (int c = c0; c < c1; ++c)
		{
			printf("% .6f ", __half2float(v[(size_t)r + (size_t)c * m]));
		}
		printf("\n");
	}
}

int main(int argc, char **argv)
{
	try
	{
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
		CUDA_CHECK_THROW(hipMemcpy(h_w_fp32.data(), w_fp32.data(), net.n_params() * sizeof(float), hipMemcpyDeviceToHost));
		for (size_t i = 0; i < h_w_fp16.size(); ++i)
		{
			h_w_fp16[i] = __float2half(h_w_fp32[i]);
		}
		CUDA_CHECK_THROW(hipMemcpy(w_train.data(), h_w_fp16.data(), net.n_params() * sizeof(__half), hipMemcpyHostToDevice));
		CUDA_CHECK_THROW(hipMemcpy(w_infer.data(), h_w_fp16.data(), net.n_params() * sizeof(__half), hipMemcpyHostToDevice));

		// Inputs: use CM layout to match hidden layers' default
		GPUMatrixDynamic<__half> input(WIDTH, BATCH, CM);
		input.initialize_xavier_uniform(rng);

		// --- 1) Fused path ---
		auto fwd_ctx = net.forward(hipStreamDefault, input, nullptr, /*use_inference=*/false, /*prepare_input_grad*/ false);
		CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));

		// Take last hidden layer output (post-activation)
		const uint32_t last_idx = net.num_forward_activations() - 1;
		auto [fused_ptr, fused_layout] = net.forward_activations(*fwd_ctx, last_idx);
		if (fused_layout != CM)
		{
			std::cerr << "Unexpected fused hidden layout. Expected CM, got RM.\n";
			// Still proceed treating memory as CM view
		}
		GPUMatrix<__half, CM> fused_last(const_cast<__half *>(fused_ptr), WIDTH, BATCH);

		// --- 2) Unfused path (hipBLAS GEMM + activation) ---
		// Experiment-2: build 4 references to test whether fused assumes transposed weights semantics
		//   nn: W0,    W1
		//   tn: W0^T,  W1
		//   nt: W0,    W1^T
		//   tt: W0^T,  W1^T
		auto &W0 = net.input_weight_matrix(false); // [WIDTH, WIDTH] RM
		auto &W1 = net.weight_matrix_at(false, 0); // [WIDTH, WIDTH] RM

		GPUMatrix<__half, CM> l0_nn(WIDTH, BATCH);
		GPUMatrix<__half, CM> l0_tn(WIDTH, BATCH);

		GPUMatrix<__half, CM> l1_nn(WIDTH, BATCH);
		GPUMatrix<__half, CM> l1_tn(WIDTH, BATCH);
		GPUMatrix<__half, CM> l1_nt(WIDTH, BATCH);
		GPUMatrix<__half, CM> l1_tt(WIDTH, BATCH);

		// Layer 0
		fc_multiply(hipStreamDefault, W0, input.cm(), l0_nn, H_ACT);
		CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));

		fc_multiply(hipStreamDefault, W0.transposed(), input.cm(), l0_tn, H_ACT);
		CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));

		// Layer 1 (4 combos)
		fc_multiply(hipStreamDefault, W1, l0_nn, l1_nn, H_ACT);
		CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));

		fc_multiply(hipStreamDefault, W1, l0_tn, l1_tn, H_ACT);
		CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));

		fc_multiply(hipStreamDefault, W1.transposed(), l0_nn, l1_nt, H_ACT);
		CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));

		fc_multiply(hipStreamDefault, W1.transposed(), l0_tn, l1_tt, H_ACT);
		CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));

		// --- 3) Compare fused vs references (element-wise) ---
		std::vector<__half> h_fused(WIDTH * BATCH);
		std::vector<__half> h_nn(WIDTH * BATCH);
		std::vector<__half> h_tn(WIDTH * BATCH);
		std::vector<__half> h_nt(WIDTH * BATCH);
		std::vector<__half> h_tt(WIDTH * BATCH);

		CUDA_CHECK_THROW(hipMemcpy(h_fused.data(), fused_last.data(), h_fused.size() * sizeof(__half), hipMemcpyDeviceToHost));
		CUDA_CHECK_THROW(hipMemcpy(h_nn.data(), l1_nn.data(), h_nn.size() * sizeof(__half), hipMemcpyDeviceToHost));
		CUDA_CHECK_THROW(hipMemcpy(h_tn.data(), l1_tn.data(), h_tn.size() * sizeof(__half), hipMemcpyDeviceToHost));
		CUDA_CHECK_THROW(hipMemcpy(h_nt.data(), l1_nt.data(), h_nt.size() * sizeof(__half), hipMemcpyDeviceToHost));
		CUDA_CHECK_THROW(hipMemcpy(h_tt.data(), l1_tt.data(), h_tt.size() * sizeof(__half), hipMemcpyDeviceToHost));

		const float eps = 1e-2f;
		const DiffStats s_nn = diff_stats_cm(h_fused, h_nn, eps);
		const DiffStats s_tn = diff_stats_cm(h_fused, h_tn, eps);
		const DiffStats s_nt = diff_stats_cm(h_fused, h_nt, eps);
		const DiffStats s_tt = diff_stats_cm(h_fused, h_tt, eps);

		auto better = [](const DiffStats &a, const DiffStats &b)
		{
			return (a.mismatches < b.mismatches) || (a.mismatches == b.mismatches && a.max_diff < b.max_diff);
		};

		const DiffStats *best = &s_nn;
		const char *best_name = "nn (W0, W1)";
		const std::vector<__half> *best_ref = &h_nn;

		if (better(s_tn, *best))
		{
			best = &s_tn;
			best_name = "tn (W0^T, W1)";
			best_ref = &h_tn;
		}
		if (better(s_nt, *best))
		{
			best = &s_nt;
			best_name = "nt (W0, W1^T)";
			best_ref = &h_nt;
		}
		if (better(s_tt, *best))
		{
			best = &s_tt;
			best_name = "tt (W0^T, W1^T)";
			best_ref = &h_tt;
		}

		if (best->mismatches == 0)
		{
			std::cout << "✓ SUCCESS: fused matches best hipBLAS reference within eps=" << eps
					  << " (best=" << best_name << ")\n";
			return 0;
		}

		std::cerr << "✗ FAILURE: fused mismatch vs hipBLAS references (eps=" << eps << ")\n";
		std::cerr << "  nn (W0,W1)       mismatches=" << s_nn.mismatches << "/" << h_fused.size() << " max_diff=" << s_nn.max_diff << " idx=" << s_nn.max_diff_idx << "\n";
		std::cerr << "  tn (W0^T,W1)     mismatches=" << s_tn.mismatches << "/" << h_fused.size() << " max_diff=" << s_tn.max_diff << " idx=" << s_tn.max_diff_idx << "\n";
		std::cerr << "  nt (W0,W1^T)     mismatches=" << s_nt.mismatches << "/" << h_fused.size() << " max_diff=" << s_nt.max_diff << " idx=" << s_nt.max_diff_idx << "\n";
		std::cerr << "  tt (W0^T,W1^T)   mismatches=" << s_tt.mismatches << "/" << h_fused.size() << " max_diff=" << s_tt.max_diff << " idx=" << s_tt.max_diff_idx << "\n";
		std::cerr << "  best=" << best_name << "\n";

		// Experiment-2: map max_diff_idx -> (row,col) in CM layout
		const size_t row = best->max_diff_idx % WIDTH;
		const size_t col = best->max_diff_idx / WIDTH;

		const size_t tile_r = row / 16;
		const size_t tile_c = col / 16;
		const size_t intra_r = row % 16;
		const size_t intra_c = col % 16;

		std::cerr << "  max_diff_idx mapping (CM): idx=" << best->max_diff_idx
				  << " -> (row=" << row << ", col=" << col << ")"
				  << " tile=(" << tile_r << "," << tile_c << ") intra=(" << intra_r << "," << intra_c << ")\n";

		// Print small samples for quick inspection
		print_sample("Fused (CM)", h_fused, WIDTH, BATCH);
		print_sample("BestRef (CM)", *best_ref, WIDTH, BATCH);

		// 4x4 patches around the worst point
		print_patch_cm("Fused", h_fused, WIDTH, BATCH, row, col, 4);
		print_patch_cm("BestRef", *best_ref, WIDTH, BATCH, row, col, 4);

		return 1;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Exception: " << e.what() << "\n";
		return 2;
	}
}

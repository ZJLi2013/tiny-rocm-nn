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

static void print_sample(const char *name, const std::vector<__half> &v, size_t m, size_t n, size_t max_rows = 2, size_t max_cols = 8)
{
    printf("%s [%zux%zu]:\n", name, m, n);
    for (size_t i = 0; i < std::min(max_rows, m); ++i)
    {
        for (size_t j = 0; j < std::min(max_cols, n); ++j)
        {
            printf("%.6f ", __half2float(v[i + j * m]));
        }
        printf("\n");
    }
}

// Purpose:
// - Compare fused kernel (rocWMMA) vs unfused (hipBLAS) for a SINGLE layer with Activation::None.
// - This removes ReLU sign-flip effects and focuses on GEMM + layout semantics.
int main(int argc, char **argv)
{
    try
    {
        const uint32_t WIDTH = 64;
        const uint32_t BATCH = 256;         // for WIDTH=64, N_ITERS=8 => needs multiple of 128; 256 is ok
        const uint32_t N_HIDDEN_LAYERS = 1; // => m_n_hidden_matmuls = 0, only first layer executed in fused kernel
        const Activation ACT = Activation::None;

        std::cout << "=== fused vs unfused GEMM-only test (Activation::None) ===\n";
        std::cout << "WIDTH=" << WIDTH << " BATCH=" << BATCH << " HIDDEN=" << N_HIDDEN_LAYERS << " ACT=None\n";

        // output_width=WIDTH (>16) ensures fused kernel does NOT execute "last layer" path;
        // and we call forward() with output=nullptr so only output_intermediate is produced.
        FullyFusedMLP<__half, WIDTH> net(WIDTH, WIDTH, N_HIDDEN_LAYERS, ACT, Activation::None);

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

        // Input: CM layout; fused kernel static path expects CM->mem_row_major mapping
        GPUMatrixDynamic<__half> input(WIDTH, BATCH, CM);
        input.initialize_xavier_uniform(rng);

        // --- 1) Fused (rocWMMA) path ---
        auto fwd_ctx = net.forward(hipStreamDefault, input, nullptr, /*use_inference=*/false, /*prepare_input_grad*/ false);
        CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));

        const uint32_t last_idx = net.num_forward_activations() - 1; // should be 0
        auto [fused_ptr, fused_layout] = net.forward_activations(*fwd_ctx, last_idx);
        if (fused_layout != CM)
        {
            std::cerr << "[WARN] fused layout expected CM, got RM. Proceeding as CM view for comparison.\n";
        }
        GPUMatrix<__half, CM> fused_out(const_cast<__half *>(fused_ptr), WIDTH, BATCH);

        // --- 2) Unfused (hipBLAS) path ---
        auto &W0 = net.input_weight_matrix(false); // [WIDTH, WIDTH] RM

        GPUMatrix<__half, CM> unfused_out(WIDTH, BATCH);
        fc_multiply(hipStreamDefault, W0, input.cm(), unfused_out, ACT);
        CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));

        // --- 3) Compare ---
        std::vector<__half> h_fused(WIDTH * BATCH);
        std::vector<__half> h_unfused(WIDTH * BATCH);
        CUDA_CHECK_THROW(hipMemcpy(h_fused.data(), fused_out.data(), h_fused.size() * sizeof(__half), hipMemcpyDeviceToHost));
        CUDA_CHECK_THROW(hipMemcpy(h_unfused.data(), unfused_out.data(), h_unfused.size() * sizeof(__half), hipMemcpyDeviceToHost));

        size_t mismatches = 0;
        float max_diff = 0.0f;
        size_t max_diff_idx = 0;
        const float eps = 2e-2f; // looser than ReLU test; compare raw GEMM

        for (size_t i = 0; i < h_fused.size(); ++i)
        {
            const float a = __half2float(h_fused[i]);
            const float b = __half2float(h_unfused[i]);
            const float d = std::fabs(a - b);
            if (d > eps)
            {
                ++mismatches;
                if (d > max_diff)
                {
                    max_diff = d;
                    max_diff_idx = i;
                }
            }
        }

        if (mismatches == 0)
        {
            std::cout << "✓ SUCCESS: fused vs unfused match within eps=" << eps << "\n";
            return 0;
        }

        std::cerr << "✗ FAILURE: fused vs unfused differ. mismatches=" << mismatches
                  << "/" << h_fused.size() << " max_diff=" << max_diff
                  << " at idx=" << max_diff_idx << " (eps=" << eps << ")\n";

        print_sample("Fused (CM)", h_fused, WIDTH, BATCH);
        print_sample("Unfused (CM)", h_unfused, WIDTH, BATCH);

        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
        return 2;
    }
}

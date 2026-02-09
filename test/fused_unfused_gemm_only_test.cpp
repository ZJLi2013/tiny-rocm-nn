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
            printf("%.6f ", __half2float(v[i + j * m]));
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
    // v is CM: idx = r + c*m, where r in [0,m), c in [0,n)
    const int half = (int)patch / 2; // patch=4 => half=2 => rows [r-2, r+1]
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
        // Experiment-2: build TWO references:
        //   ref0: use W0 as-is (current behavior)
        //   ref1: use W0.transposed() (tests whether fused assumes transposed weights semantics)
        auto &W0 = net.input_weight_matrix(false); // [WIDTH, WIDTH] RM

        GPUMatrix<__half, CM> ref0_out(WIDTH, BATCH);
        GPUMatrix<__half, CM> ref1_out(WIDTH, BATCH);

        fc_multiply(hipStreamDefault, W0, input.cm(), ref0_out, ACT);
        CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));

        fc_multiply(hipStreamDefault, W0.transposed(), input.cm(), ref1_out, ACT);
        CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));

        // --- 3) Compare ---
        std::vector<__half> h_fused(WIDTH * BATCH);
        std::vector<__half> h_ref0(WIDTH * BATCH);
        std::vector<__half> h_ref1(WIDTH * BATCH);

        CUDA_CHECK_THROW(hipMemcpy(h_fused.data(), fused_out.data(), h_fused.size() * sizeof(__half), hipMemcpyDeviceToHost));
        CUDA_CHECK_THROW(hipMemcpy(h_ref0.data(), ref0_out.data(), h_ref0.size() * sizeof(__half), hipMemcpyDeviceToHost));
        CUDA_CHECK_THROW(hipMemcpy(h_ref1.data(), ref1_out.data(), h_ref1.size() * sizeof(__half), hipMemcpyDeviceToHost));

        const float eps = 2e-2f; // compare raw GEMM
        const DiffStats s0 = diff_stats_cm(h_fused, h_ref0, eps);
        const DiffStats s1 = diff_stats_cm(h_fused, h_ref1, eps);

        const bool ref1_better = (s1.mismatches < s0.mismatches) || (s1.mismatches == s0.mismatches && s1.max_diff < s0.max_diff);
        const DiffStats best = ref1_better ? s1 : s0;

        if (best.mismatches == 0)
        {
            std::cout << "✓ SUCCESS: fused matches best hipBLAS reference within eps=" << eps
                      << " (best=" << (ref1_better ? "W0.transposed()" : "W0") << ")\n";
            return 0;
        }

        std::cerr << "✗ FAILURE: fused mismatch vs hipBLAS references (eps=" << eps << ")\n";
        std::cerr << "  ref0=W0            mismatches=" << s0.mismatches << "/" << h_fused.size() << " max_diff=" << s0.max_diff << " idx=" << s0.max_diff_idx << "\n";
        std::cerr << "  ref1=W0.transposed mismatches=" << s1.mismatches << "/" << h_fused.size() << " max_diff=" << s1.max_diff << " idx=" << s1.max_diff_idx << "\n";
        std::cerr << "  best=" << (ref1_better ? "ref1 (W0.transposed)" : "ref0 (W0)") << "\n";

        // Experiment-2: map max_diff_idx -> (row,col) in CM layout
        const size_t row = best.max_diff_idx % WIDTH;
        const size_t col = best.max_diff_idx / WIDTH;

        const size_t tile_r = row / 16;
        const size_t tile_c = col / 16;
        const size_t intra_r = row % 16;
        const size_t intra_c = col % 16;

        std::cerr << "  max_diff_idx mapping (CM): idx=" << best.max_diff_idx
                  << " -> (row=" << row << ", col=" << col << ")"
                  << " tile=(" << tile_r << "," << tile_c << ") intra=(" << intra_r << "," << intra_c << ")\n";

        print_sample("Fused (CM)", h_fused, WIDTH, BATCH);
        print_sample("Ref0 hipBLAS (CM)  W0", h_ref0, WIDTH, BATCH);
        print_sample("Ref1 hipBLAS (CM)  W0.transposed()", h_ref1, WIDTH, BATCH);

        // 4x4 patches around the worst point
        print_patch_cm("Fused", h_fused, WIDTH, BATCH, row, col, 4);
        print_patch_cm("Ref0 (W0)", h_ref0, WIDTH, BATCH, row, col, 4);
        print_patch_cm("Ref1 (W0^T)", h_ref1, WIDTH, BATCH, row, col, 4);

        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
        return 2;
    }
}

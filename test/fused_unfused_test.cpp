/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * This file is inspired by and based on the test infrastructure of tiny-cuda-nn,
 * and is used for debugging and verification of the ROCm port.
 */

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/cublas_matmul.h>

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>

using namespace tcnn;

// Helper to compare two GPU matrices
template <typename T>
bool check_matrices_equal(const GPUMatrixDynamic<T>& mat1, const GPUMatrixDynamic<T>& mat2, float epsilon = 1e-2) {
    if (mat1.m() != mat2.m() || mat1.n() != mat2.n()) {
        std::cerr << "Matrix dimensions do not match: [" << mat1.m() << "x" << mat1.n() 
                  << "] vs [" << mat2.m() << "x" << mat2.n() << "]" << std::endl;
        return false;
    }

    std::vector<T> h_mat1(mat1.n_elements());
    std::vector<T> h_mat2(mat2.n_elements());

    CUDA_CHECK_THROW(hipMemcpy(h_mat1.data(), mat1.data(), mat1.n_bytes(), hipMemcpyDeviceToHost));
    CUDA_CHECK_THROW(hipMemcpy(h_mat2.data(), mat2.data(), mat2.n_bytes(), hipMemcpyDeviceToHost));

    size_t mismatch_count = 0;
    float max_diff = 0.0f;
    size_t first_mismatch_idx = 0;

    for (size_t i = 0; i < h_mat1.size(); ++i) {
        float v1 = std::is_same<T, __half>::value ? __half2float(h_mat1[i]) : (float)h_mat1[i];
        float v2 = std::is_same<T, __half>::value ? __half2float(h_mat2[i]) : (float)h_mat2[i];
        float diff = std::abs(v1 - v2);
        
        if (diff > epsilon) {
            if (mismatch_count == 0) {
                first_mismatch_idx = i;
            }
            mismatch_count++;
            max_diff = std::max(max_diff, diff);
        }
    }

    if (mismatch_count > 0) {
        std::cerr << "Mismatch: " << mismatch_count << "/" << h_mat1.size() 
                  << " elements, max_diff=" << max_diff 
                  << ", first at idx=" << first_mismatch_idx << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char** argv) {
    try {
        // Test parameters
        const uint32_t BATCH_SIZE = 4096;
        const uint32_t WIDTH = 64;
        const uint32_t N_HIDDEN_LAYERS = 2;
        const Activation ACTIVATION = Activation::ReLU;

        std::cout << "=== Fused vs Unfused MLP Test ===" << std::endl;
        std::cout << "Config: WIDTH=" << WIDTH << ", N_HIDDEN_LAYERS=" << N_HIDDEN_LAYERS 
                  << ", BATCH_SIZE=" << BATCH_SIZE << std::endl;

        // Initialize network
        FullyFusedMLP<__half, WIDTH> network(WIDTH, WIDTH, N_HIDDEN_LAYERS, ACTIVATION, Activation::None);

        // Allocate and initialize weights
        GPUMemory<__half> weights_memory(network.n_params());
        GPUMemory<__half> weights_inference_memory(network.n_params());
        GPUMemory<__half> gradients_memory(network.n_params());
        
        network.set_params(weights_memory.data(), weights_inference_memory.data(), gradients_memory.data());
        
        // Initialize weights with Xavier uniform
        GPUMemory<float> weights_fp32(network.n_params());
        pcg32 rnd(1337);
        network.initialize_params(rnd, weights_fp32.data(), 1.0f);
        
        // Convert FP32 weights to FP16
        std::vector<float> h_weights_fp32(network.n_params());
        std::vector<__half> h_weights_fp16(network.n_params());
        CUDA_CHECK_THROW(hipMemcpy(h_weights_fp32.data(), weights_fp32.data(), 
                                    network.n_params() * sizeof(float), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < network.n_params(); ++i) {
            h_weights_fp16[i] = __float2half(h_weights_fp32[i]);
        }
        CUDA_CHECK_THROW(hipMemcpy(weights_memory.data(), h_weights_fp16.data(), 
                                    network.n_params() * sizeof(__half), hipMemcpyHostToDevice));
        CUDA_CHECK_THROW(hipMemcpy(weights_inference_memory.data(), h_weights_fp16.data(), 
                                    network.n_params() * sizeof(__half), hipMemcpyHostToDevice));

        // Create and initialize input
        GPUMatrixDynamic<__half> input(WIDTH, BATCH_SIZE, RM);
        input.initialize_xavier_uniform(rnd);

        std::cout << "Network initialized with " << network.n_params() << " parameters" << std::endl;

        // --- 1. Fused Path ---
        std::cout << "\n[1] Running fused forward pass..." << std::endl;
        GPUMatrixDynamic<__half> fused_output(WIDTH, BATCH_SIZE, RM);
        
        auto fused_ctx = network.forward(hipStreamDefault, input, &fused_output, false, false);
        CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));
        
        std::cout << "    Fused forward completed" << std::endl;

        // --- 2. Unfused Path ---
        std::cout << "\n[2] Running unfused forward pass..." << std::endl;
        
        // Get weight matrices from network
        auto& W0 = network.input_weight_matrix(false);      // First layer: [WIDTH, WIDTH]
        auto& W1 = network.weight_matrix_at(false, 0);      // Hidden layer: [WIDTH, WIDTH]
        auto& W2 = network.output_weight_matrix(false);     // Output layer: [WIDTH, WIDTH]
        
        std::cout << "    Weight shapes: W0=[" << W0.m() << "x" << W0.n() << "], "
                  << "W1=[" << W1.m() << "x" << W1.n() << "], "
                  << "W2=[" << W2.m() << "x" << W2.n() << "]" << std::endl;

        // Allocate intermediate buffers
        GPUMatrixDynamic<__half> layer0_out(WIDTH, BATCH_SIZE, RM);
        GPUMatrixDynamic<__half> layer1_out(WIDTH, BATCH_SIZE, RM);
        GPUMatrixDynamic<__half> unfused_output(WIDTH, BATCH_SIZE, RM);

        // Layer 0: input -> layer0_out (with ReLU)
        fc_multiply(hipStreamDefault, W0.transposed(), input.rm(), layer0_out.rm(), ACTIVATION);
        CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));
        std::cout << "    Layer 0 completed" << std::endl;

        // Layer 1: layer0_out -> layer1_out (with ReLU)
        fc_multiply(hipStreamDefault, W1.transposed(), layer0_out.rm(), layer1_out.rm(), ACTIVATION);
        CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));
        std::cout << "    Layer 1 completed" << std::endl;

        // Layer 2: layer1_out -> unfused_output (no activation)
        fc_multiply(hipStreamDefault, W2.transposed(), layer1_out.rm(), unfused_output.rm(), Activation::None);
        CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));
        std::cout << "    Layer 2 completed" << std::endl;

        // --- 3. Compare Results ---
        std::cout << "\n[3] Comparing outputs..." << std::endl;
        bool success = check_matrices_equal(fused_output, unfused_output);

        if (success) {
            std::cout << "\n✓ SUCCESS: Fused and unfused outputs match!" << std::endl;
            return 0;
        } else {
            std::cerr << "\n✗ FAILURE: Fused and unfused outputs differ!" << std::endl;
            
            // Print sample values for debugging
            std::vector<__half> h_fused(std::min<size_t>(16, fused_output.n_elements()));
            std::vector<__half> h_unfused(std::min<size_t>(16, unfused_output.n_elements()));
            CUDA_CHECK_THROW(hipMemcpy(h_fused.data(), fused_output.data(), 
                                        h_fused.size() * sizeof(__half), hipMemcpyDeviceToHost));
            CUDA_CHECK_THROW(hipMemcpy(h_unfused.data(), unfused_output.data(), 
                                        h_unfused.size() * sizeof(__half), hipMemcpyDeviceToHost));
            
            std::cout << "First 16 elements:" << std::endl;
            std::cout << "Fused:   ";
            for (size_t i = 0; i < h_fused.size(); ++i) {
                std::cout << __half2float(h_fused[i]) << " ";
            }
            std::cout << std::endl;
            std::cout << "Unfused: ";
            for (size_t i = 0; i < h_unfused.size(); ++i) {
                std::cout << __half2float(h_unfused[i]) << " ";
            }
            std::cout << std::endl;
            
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

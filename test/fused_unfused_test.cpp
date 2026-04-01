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
bool check_matrices_equal(const GPUMatrix<T, RM>& mat1, const GPUMatrix<T, RM>& mat2, float epsilon = 1e-2) {
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
        const uint32_t BATCH_SIZE = 256;  // Must be multiple of BATCH_SIZE_GRANULARITY (256)
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

        // Create and initialize input (CM layout to match hidden layers)
        GPUMatrixDynamic<__half> input(WIDTH, BATCH_SIZE, CM);
        input.initialize_xavier_uniform(rnd);

        std::cout << "Network initialized with " << network.n_params() << " parameters" << std::endl;

        // --- 1. Fused Path ---
        std::cout << "\n[1] Running fused forward pass..." << std::endl;
        
        // Use the public interface to get forward activations
        auto fused_ctx = network.forward(hipStreamDefault, input, nullptr, false, false);
        CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));
        
        std::cout << "    Fused forward completed" << std::endl;

        // Get the last hidden layer activation using public interface
        // forward_activations returns (data_ptr, layout) for the specified layer
        uint32_t last_layer_idx = network.num_forward_activations() - 1;
        auto [fused_data, fused_layout] = network.forward_activations(*fused_ctx, last_layer_idx);
        
        std::cout << "    Fused last hidden layer: idx=" << last_layer_idx 
                  << ", layout=" << (fused_layout == CM ? "CM" : "RM") << std::endl;
        
        // Create a view of the fused output
        GPUMatrix<__half, CM> fused_last_hidden(const_cast<__half*>(fused_data), WIDTH, BATCH_SIZE);

        // --- 2. Unfused Path ---
        std::cout << "\n[2] Running unfused forward pass..." << std::endl;
        
        // Get weight matrices from network
        auto& W0 = network.input_weight_matrix(false);      // First layer: [WIDTH, WIDTH]
        auto& W1 = network.weight_matrix_at(false, 0);      // Hidden layer: [WIDTH, WIDTH]
        
        std::cout << "    Weight shapes: W0=[" << W0.m() << "x" << W0.n() << "], "
                  << "W1=[" << W1.m() << "x" << W1.n() << "]" << std::endl;

        // Allocate intermediate buffers (CM layout to match fused path)
        GPUMatrix<__half, CM> layer0_out(WIDTH, BATCH_SIZE);
        GPUMatrix<__half, CM> layer1_out(WIDTH, BATCH_SIZE);

        // Layer 0: input -> layer0_out (with ReLU)
        // Use W0 directly (not transposed) -- verified by fused_unfused_layers_test "nn" combo
        fc_multiply(hipStreamDefault, W0, input.cm(), layer0_out, ACTIVATION);
        CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));
        std::cout << "    Layer 0 completed" << std::endl;

        // Layer 1: layer0_out -> layer1_out (with ReLU)
        fc_multiply(hipStreamDefault, W1, layer0_out, layer1_out, ACTIVATION);
        CUDA_CHECK_THROW(hipStreamSynchronize(hipStreamDefault));
        std::cout << "    Layer 1 completed" << std::endl;

        // --- 3. Compare Results ---
        std::cout << "\n[3] Comparing last hidden layer outputs..." << std::endl;
        
        // Both are CM layout, compare directly by copying to host
        std::vector<__half> h_fused(WIDTH * BATCH_SIZE);
        std::vector<__half> h_unfused(WIDTH * BATCH_SIZE);
        
        CUDA_CHECK_THROW(hipMemcpy(h_fused.data(), fused_last_hidden.data(), 
                                    WIDTH * BATCH_SIZE * sizeof(__half), hipMemcpyDeviceToHost));
        CUDA_CHECK_THROW(hipMemcpy(h_unfused.data(), layer1_out.data(), 
                                    WIDTH * BATCH_SIZE * sizeof(__half), hipMemcpyDeviceToHost));
        
        // Compare element by element
        size_t mismatch_count = 0;
        float max_diff = 0.0f;
        size_t first_mismatch_idx = 0;
        
        for (size_t i = 0; i < h_fused.size(); ++i) {
            float v1 = __half2float(h_fused[i]);
            float v2 = __half2float(h_unfused[i]);
            float diff = std::abs(v1 - v2);
            
            if (diff > 1e-2f) {
                if (mismatch_count == 0) {
                    first_mismatch_idx = i;
                }
                mismatch_count++;
                max_diff = std::max(max_diff, diff);
            }
        }
        
        bool success = (mismatch_count == 0);
        
        if (mismatch_count > 0) {
            std::cerr << "Mismatch: " << mismatch_count << "/" << h_fused.size() 
                      << " elements, max_diff=" << max_diff 
                      << ", first at idx=" << first_mismatch_idx << std::endl;
        }

        if (success) {
            std::cout << "\n✓ SUCCESS: Fused and unfused outputs match!" << std::endl;
            return 0;
        } else {
            std::cerr << "\n✗ FAILURE: Fused and unfused outputs differ!" << std::endl;
            
            // Print first 16 elements for debugging
            std::cout << "First 16 elements:" << std::endl;
            std::cout << "Fused:   ";
            for (size_t i = 0; i < 16 && i < h_fused.size(); ++i) {
                std::cout << __half2float(h_fused[i]) << " ";
            }
            std::cout << std::endl;
            std::cout << "Unfused: ";
            for (size_t i = 0; i < 16 && i < h_unfused.size(); ++i) {
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

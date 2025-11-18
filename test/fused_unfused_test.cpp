/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * This file is inspired by and based on the test infrastructure of tiny-cuda-nn,
 * and is used for debugging and verification of the ROCm port.
 */

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>
#include <tiny-cuda-nn/misc.h>

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace tcnn;

// Helper to compare two GPU matrices
template <typename T>
bool check_matrices_equal(const GPUMatrix<T>& mat1, const GPUMatrix<T>& mat2, float epsilon = 1e-3) {
    if (mat1.m() != mat2.m() || mat1.n() != mat2.n()) {
        std::cerr << "Matrix dimensions do not match!" << std::endl;
        return false;
    }

    std::vector<T> h_mat1(mat1.n_elements());
    std::vector<T> h_mat2(mat2.n_elements());

    mat1.copy_to_host(h_mat1.data());
    mat2.copy_to_host(h_mat2.data());

    for (size_t i = 0; i < h_mat1.size(); ++i) {
        float diff = std::abs((float)h_mat1[i] - (float)h_mat2[i]);
        if (diff > epsilon) {
            std::cerr << "Mismatch at index " << i << ": " << (float)h_mat1[i] << " vs " << (float)h_mat2[i] << ", diff: " << diff << std::endl;
            return false;
        }
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

        // Initialize network
        FullyFusedMLP<__half, WIDTH> network(WIDTH, WIDTH, N_HIDDEN_LAYERS, ACTIVATION, Activation::None);

        // Create input and weight matrices
        GPUMatrix<__half, RM> weights(network.n_params(), 1);
        GPUMatrixDynamic<__half> input(WIDTH, BATCH_SIZE, RM);
        
        // Initialize with random data
        pcg32 rnd(1337);
        weights.initialize_xavier_uniform(rnd);
        input.initialize_xavier_uniform(rnd);

        // --- 1. Fused Path ---
        GPUMatrixDynamic<__half> fused_output(WIDTH, BATCH_SIZE, RM);
        GPUMatrix<__half> fused_intermediate(network.num_forward_activations() * WIDTH, BATCH_SIZE);
        
        std::cout << "Running fused forward pass..." << std::endl;
        mlp_fused_forward<WIDTH, __half, ACTIVATION, false>(
            hipStreamDefault,
            Activation::None,
            weights,
            input,
            fused_intermediate,
            &fused_output,
            N_HIDDEN_LAYERS - 1
        );

        // --- 2. Unfused Path ---
        GPUMatrixDynamic<__half> unfused_output(WIDTH, BATCH_SIZE, RM);
        GPUMatrix<__half> unfused_intermediate(WIDTH, BATCH_SIZE);

        std::cout << "Running unfused forward pass..." << std::endl;
        
        // Unfused Layer 1
        GPUMatrix<__half, RM> weights_layer1(weights.data(), WIDTH, WIDTH);
        fc_multiply(hipStreamDefault, weights_layer1.transposed(), input.rm(), unfused_intermediate, ACTIVATION);

        // Unfused Layer 2
        GPUMatrix<__half, RM> weights_layer2(weights.data() + weights_layer1.n_elements(), WIDTH, WIDTH);
        fc_multiply(hipStreamDefault, weights_layer2.transposed(), unfused_intermediate, unfused_output.rm(), Activation::None);

        // --- 3. Compare Results ---
        std::cout << "Comparing outputs..." << std::endl;
        bool success = check_matrices_equal(fused_output.rm(), unfused_output.rm());

        if (success) {
            std::cout << "SUCCESS: Fused and unfused outputs match!" << std::endl;
            return 0;
        } else {
            std::cerr << "FAILURE: Fused and unfused outputs differ!" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstring>

// Load binary file
template <typename T>
std::vector<T> load_binary(const char* filename, size_t expected_size) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Failed to open %s\n", filename);
        exit(1);
    }
    
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    size_t num_elements = file_size / sizeof(T);
    if (num_elements != expected_size) {
        printf("Warning: %s size mismatch. Expected %zu, got %zu\n", 
               filename, expected_size, num_elements);
    }
    
    std::vector<T> buf(num_elements);
    fread(buf.data(), sizeof(T), num_elements, f);
    fclose(f);
    
    printf("Loaded %s: %zu elements\n", filename, num_elements);
    return buf;
}

// Parse meta file
struct GemmMeta {
    int m, n, k;
    char op_a[8], op_b[8];
    int lda, ldb, ldc;
    float alpha, beta;
    int algo;
};

GemmMeta load_meta(const char* filename) {
    GemmMeta meta = {};
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Failed to open %s\n", filename);
        exit(1);
    }
    
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "m=%d n=%d k=%d", &meta.m, &meta.n, &meta.k) == 3) continue;
        if (sscanf(line, "op_a=%s op_b=%s", meta.op_a, meta.op_b) == 2) continue;
        if (sscanf(line, "lda=%d ldb=%d ldc=%d", &meta.lda, &meta.ldb, &meta.ldc) == 3) continue;
        if (sscanf(line, "alpha=%f beta=%f", &meta.alpha, &meta.beta) == 2) continue;
        if (sscanf(line, "algo=%d", &meta.algo) == 1) continue;
    }
    fclose(f);
    
    printf("Meta: m=%d n=%d k=%d, op_a=%s op_b=%s, lda=%d ldb=%d ldc=%d, alpha=%.4f beta=%.4f\n",
           meta.m, meta.n, meta.k, meta.op_a, meta.op_b, 
           meta.lda, meta.ldb, meta.ldc, meta.alpha, meta.beta);
    return meta;
}

// Compute stats
template <typename T>
void compute_stats(const std::vector<T>& buf, const char* name) {
    size_t nan_count = 0, inf_count = 0;
    float max_abs = 0.0f, min_val = 1e10f, max_val = -1e10f;
    
    for (size_t i = 0; i < buf.size(); ++i) {
        float v = (float)buf[i];
        if (std::isnan(v)) { ++nan_count; continue; }
        if (std::isinf(v)) { ++inf_count; continue; }
        float a = std::fabs(v);
        if (a > max_abs) max_abs = a;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    
    printf("%s stats: NaN=%zu Inf=%zu range=[%.6f, %.6f] max_abs=%.6f\n",
           name, nan_count, inf_count, min_val, max_val, max_abs);
    
    // Print first 16 values
    printf("%s[0:16]: ", name);
    for (int i = 0; i < 16 && i < (int)buf.size(); ++i) {
        printf("%.4f ", (float)buf[i]);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <dump_prefix>\n", argv[0]);
        printf("Example: %s mixed_rm_first_\n", argv[0]);
        printf("Will load: <prefix>A_before.bin, <prefix>B_before.bin, <prefix>C_before.bin, <prefix>meta.txt\n");
        return 1;
    }
    
    const char* prefix = argv[1];
    char path_a[512], path_b[512], path_c[512], path_meta[512];
    snprintf(path_a, sizeof(path_a), "%sA_before.bin", prefix);
    snprintf(path_b, sizeof(path_b), "%sB_before.bin", prefix);
    snprintf(path_c, sizeof(path_c), "%sC_before.bin", prefix);
    snprintf(path_meta, sizeof(path_meta), "%smeta.txt", prefix);
    
    // Load meta
    GemmMeta meta = load_meta(path_meta);
    
    // Load matrices
    auto A_host = load_binary<__half>(path_a, (size_t)meta.m * meta.k);
    auto B_host = load_binary<__half>(path_b, (size_t)meta.k * meta.n);
    auto C_host = load_binary<__half>(path_c, (size_t)meta.m * meta.n);
    
    // Compute stats
    compute_stats(A_host, "A_before");
    compute_stats(B_host, "B_before");
    compute_stats(C_host, "C_before");
    
    // Allocate GPU memory
    __half *A_dev, *B_dev, *C_dev;
    hipMalloc(&A_dev, A_host.size() * sizeof(__half));
    hipMalloc(&B_dev, B_host.size() * sizeof(__half));
    hipMalloc(&C_dev, C_host.size() * sizeof(__half));
    
    // Copy to GPU
    hipMemcpy(A_dev, A_host.data(), A_host.size() * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(B_dev, B_host.data(), B_host.size() * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(C_dev, C_host.data(), C_host.size() * sizeof(__half), hipMemcpyHostToDevice);
    
    // Create hipBLAS handle
    hipblasHandle_t handle;
    hipblasCreate(&handle);
    
    // Parse op flags
    hipblasOperation_t op_a = (strcmp(meta.op_a, "N") == 0) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    hipblasOperation_t op_b = (strcmp(meta.op_b, "N") == 0) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    
    printf("\n=== Replaying GEMM ===\n");
    printf("hipblasGemmEx(handle, op_b=%s, op_a=%s, n=%d, m=%d, k=%d,\n",
           meta.op_b, meta.op_a, meta.n, meta.m, meta.k);
    printf("              alpha=%.4f, B, ldb=%d, A, lda=%d,\n", meta.alpha, meta.ldb, meta.lda);
    printf("              beta=%.4f, C, ldc=%d, FP32_COMPUTE)\n", meta.beta, meta.ldc);
    
    // Execute GEMM
    hipblasStatus_t status = hipblasGemmEx(
        handle,
        op_b, op_a,
        meta.n, meta.m, meta.k,
        &meta.alpha,
        B_dev, HIPBLAS_R_16F, meta.ldb,
        A_dev, HIPBLAS_R_16F, meta.lda,
        &meta.beta,
        C_dev, HIPBLAS_R_16F, meta.ldc,
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT
    );
    
    if (status != HIPBLAS_STATUS_SUCCESS) {
        printf("hipblasGemmEx failed: %d\n", (int)status);
        return 1;
    }
    
    hipDeviceSynchronize();
    printf("GEMM completed successfully\n");
    
    // Copy result back
    std::vector<__half> C_after(C_host.size());
    hipMemcpy(C_after.data(), C_dev, C_after.size() * sizeof(__half), hipMemcpyDeviceToHost);
    
    // Compute stats
    compute_stats(C_after, "C_after");
    
    // Compare with expected C_after if available
    char path_c_after[512];
    snprintf(path_c_after, sizeof(path_c_after), "%sC_after.bin", prefix);
    FILE* f_check = fopen(path_c_after, "rb");
    if (f_check) {
        fclose(f_check);
        auto C_expected = load_binary<__half>(path_c_after, C_after.size());
        compute_stats(C_expected, "C_expected");
        
        // Compare
        size_t diff_count = 0;
        float max_diff = 0.0f;
        for (size_t i = 0; i < C_after.size(); ++i) {
            float v1 = (float)C_after[i];
            float v2 = (float)C_expected[i];
            float diff = std::fabs(v1 - v2);
            if (diff > 1e-4f) {
                ++diff_count;
                if (diff > max_diff) max_diff = diff;
            }
        }
        printf("\nComparison: diff_count=%zu/%zu max_diff=%.6f\n", 
               diff_count, C_after.size(), max_diff);
        
        if (diff_count == 0) {
            printf("✓ MATCH: Offline result matches training C_after\n");
        } else {
            printf("✗ MISMATCH: Offline result differs from training\n");
        }
    }
    
    // Cleanup
    hipFree(A_dev);
    hipFree(B_dev);
    hipFree(C_dev);
    hipblasDestroy(handle);
    
    return 0;
}
# v010 训练不收敛问题分析

## 问题现状

- ✓ v010 没有 cuBLAS 运行时错误
- ✓ `transposed()` 方法经过验证是正确的
- ✗ 训练不收敛：loss 停留在 8.7-8.9

## 配置信息

从 `samples/mlp_learning_an_image.cu` 和 `data/config.json`:
- `n_input_dims = 2` (图像坐标)
- `n_output_dims = 3` (RGB)
- `m_padded_output_width = 16` (padding 后)
- `n_neurons = 64` (隐藏层宽度)
- `n_hidden_layers = 4`
- `activation = ReLU`
- `output_activation = None`

从 v005.log 可以看到：
```
[DEBUG backward_impl] tmp_dL_doutput: 16x256 layout=CM
[DEBUG backward_impl] forward.hidden.at(3).transposed(): 256x64
[DEBUG backward_impl] output_gradient_matrix(): 16x64
```

## 关键发现：CUTLASS vs cuBLAS 的区别

### CUTLASS (官方 tiny-cuda-nn)

**Epilogue 融合**:
- GEMM + 激活函数在一个 kernel 中完成
- 反向传播中，`transfer=true` 允许激活函数反向传播与 GEMM 融合
- 高效且数值稳定

**在反向传播中**:
```cpp
// CUTLASS 可以这样调用
fc_multiply(stream, W^T, dL_dout, dL_dhidden, dL_dhidden,
            activation,  // ReLU
            true,        // transfer=true: 使用前向激活值
            true);       // sum_source=true
```

### cuBLAS (tiny-rocm-nn)

**无 Epilogue 融合**:
- cuBLAS 只做 GEMM
- 激活函数必须单独执行
- 当前实现在 `transfer=true` 时**直接抛异常**

**当前实现**:
```cpp
if (transfer) {
    throw std::runtime_error("cuBLAS fc_multiply does not support transfer=true...");
}

// 激活函数处理（如果有）
if (activation != Activation::None) {
    kernel_activation<T, 1><<<...>>>(num_elements, activation, D.data(), D.data());
}
```

## 执行路径分析

### 前向传播

由于 `m_output_width = 3 < 16`，前向传播完全在融合 kernel 中完成：
- ✓ 不使用 `fc_multiply`
- ✓ 所有层（包括输出层）都在 `kernel_mlp_fused` 中处理

### 反向传播

从 `fully_fused_mlp.cu` 的 `backward_impl`:

```cpp
// 1. 输出层激活函数反向传播（如果有）
if (m_output_activation != Activation::None) {
    activation_backward_output_gpu(...);  // 单独处理
}

// 2. 输出层权重梯度
fc_multiply_split_k(stream, tmp_dL_doutput, forward.hidden.at(tmp_idx).transposed(),
                    output_gradient_matrix(), split_k_factor, param_gradient_beta);
// 注意：这里没有 activation 参数！

// 3. 如果 m_output_width > 16，计算对隐藏层的梯度
if (m_output_width > 16) {
    fc_multiply(stream, output_weight_matrix(...).transposed(), tmp_dL_doutput,
                backward_tmp.at(...), backward_tmp.at(...),
                m_activation, true, true);  // 这里有 transfer=true!
}

// 4. 融合 kernel 处理隐藏层反向传播
mlp_fused_backward<...>(...);  // 处理所有隐藏层

// 5. 隐藏层权重梯度
for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
    fc_multiply_split_k(stream, backward_tmp.at(...), forward.hidden.at(...).transposed(),
                        gradient_matrix_at(...), split_k_factor, param_gradient_beta);
    // 注意：这里也没有 activation 参数！
}

// 6. 输入层权重梯度
fc_multiply_split_k(stream, backward_tmp.at(...), input.transposed(),
                    input_gradient_matrix(), split_k_factor, param_gradient_beta);
```

## 关键问题识别

### 问题 1: m_output_width <= 16 的情况

由于 `m_padded_output_width = 16`，**不会**执行：
```cpp
if (m_output_width > 16) {
    fc_multiply(..., m_activation, true, true);  // 这段代码不会执行！
}
```

**这意味着**：
- 融合 kernel `mlp_fused_backward` 直接处理了从输出层到输入层的所有激活梯度
- `fc_multiply_split_k` 只用于计算**权重梯度**，不涉及激活函数

### 问题 2: fc_multiply_split_k 不需要激活函数

权重梯度的计算公式：
```
dL/dW = dL/doutput * input^T
```

这里的 `dL/doutput` **已经**包含了激活函数的反向传播！

**示例**：
```
Layer: input -> [W] -> pre_activation -> [ReLU] -> activation -> output

反向传播：
1. dL/d(activation) = 从上一层传来的梯度
2. dL/d(pre_activation) = dL/d(activation) * ReLU'(pre_activation)  <- 融合 kernel 完成
3. dL/dW = dL/d(pre_activation) * input^T  <- fc_multiply_split_k 完成
```

**结论**: `fc_multiply_split_k` 接收的 `dL_doutput` 应该已经是经过激活函数反向传播处理的梯度！

## 可能的根本原因

### 假设 1: 融合 kernel 的激活反向传播有问题

`mlp_fused_backward` 在处理激活梯度时可能有 bug，导致传给 `fc_multiply_split_k` 的梯度不正确。

### 假设 2: fc_multiply_split_k 被错误调用

可能在某处，`fc_multiply_split_k` 被传入了**未经激活反向传播处理**的梯度。

### 假设 3: cuBLAS 矩阵乘法本身的问题

虽然没有运行时错误，但混合布局的 GEMM 计算结果可能不正确。

## 调试策略

### 新增的调试功能

1. **矩阵内容采样打印** (`print_matrix_sample`)
   - 打印矩阵的前 4x4 元素
   - 支持 CM 和 RM 布局

2. **混合布局 gemm 详细日志**
   - 前 3 次调用的参数和矩阵内容
   - cuBLAS 操作类型
   - 输入输出矩阵采样

3. **fc_multiply_split_k 调用追踪**
   - 前 5 次调用的参数信息

### 需要添加的调试

1. **在 fc_multiply 中添加**:
   ```cpp
   static int fc_multiply_call_count = 0;
   fc_multiply_call_count++;
   
   if (fc_multiply_call_count <= 3) {
       std::cout << "\n[fc_multiply Call #" << fc_multiply_call_count << "]" << std::endl;
       std::cout << "  activation=" << to_string(activation) << std::endl;
       std::cout << "  transfer=" << transfer << std::endl;
       std::cout << "  sum_source=" << sum_source << std::endl;
   }
   ```

2. **验证梯度值的合理性**:
   - 检查 `dL_doutput` 的数值范围
   - 检查权重梯度的数值范围
   - 对比 CUTLASS 版本的相同位置的值

## 下一步行动

1. **添加 fc_multiply 调试输出**
2. **重新编译运行**，收集详细日志
3. **手动验证**几个矩阵乘法的结果
4. **对比 CUTLASS 版本**的中间结果（如果可能）

## 预期发现

通过详细日志，我们应该能够确定：
1. cuBLAS GEMM 的计算结果是否正确
2. 梯度的数值范围是否合理
3. 是否有某个特定的矩阵乘法出错

---

## v011 调试增强 (Oct 30, 2025)

### 实施的更改

**文件**: `include/tiny-cuda-nn/cublas_matmul.h`
**Patch**: `claude/claude_v011.patch`
**Commit**: c6f2db7

#### 1. 新增矩阵内容采样函数

```cpp
template <typename T>
void print_matrix_sample(const char* name, const T* data, int m, int n, 
                        int stride, MatrixLayout layout, cudaStream_t stream = 0)
```

功能：
- 打印矩阵前 4x4 元素
- 支持 CM 和 RM 布局的正确解释
- 自动同步 CUDA stream

#### 2. cublas_gemm 混合布局详细日志

在混合布局版本中添加（前 3 次调用）：
- 矩阵维度、布局、stride 信息
- cuBLAS 操作类型（op_a, op_b）
- 输入矩阵 A、B 的内容采样
- 输出矩阵 C 的内容采样
- cuBLAS 调用参数（m, n, k）

#### 3. fc_multiply 调用追踪

添加（前 3 次调用）：
- 所有矩阵的维度和布局
- activation, transfer, sum_source 参数
- 激活函数应用前后的矩阵内容对比

#### 4. fc_multiply_split_k 增强日志

添加（前 5 次调用）：
- 矩阵维度、布局、stride
- split_k 和 beta 参数

### 调试目标

通过这些详细日志，验证：

1. **GEMM 计算正确性**
   - 手动验证 C[0,0] = Σ(A[0,i] * B[i,0])
   - 检查输出矩阵的数值范围

2. **cuBLAS 参数正确性**
   - op_a, op_b 的选择
   - 维度参数 (m, n, k)
   - stride 值

3. **激活函数处理**
   - 确认是否被调用
   - 验证激活前后的数值变化

### 预期日志格式

```
[fc_multiply_split_k Call #1]
  A: 16x256 (CM, stride=16)
  B: 256x64 (RM, stride=64)
  C: 16x64 (RM, stride=64)
  split_k=1, beta=0

[cublas_gemm MIXED LAYOUT Call #1]
  A: 16x256 (CM, stride=16)
  B: 256x64 (RM, stride=64)
  C: 16x64 (RM, stride=64)
  alpha=1, beta=0
  Output is RM, using C^T = B^T * A^T
  op_b=N, op_a=T
  cuBLAS call: gemm(op_b, op_a, n=64, m=16, k=256)
  Input A (16x256, CM, stride=16) sample:
    [4x4 matrix values]
  Input B (256x64, RM, stride=64) sample:
    [4x4 matrix values]
  Output C (16x64, RM, stride=64) sample:
    [4x4 matrix values]
```

### v012 错误分析

重新审视 v11.log，发现一个关键问题：

```cpp
// 当前代码
float alpha = 1.0f;
float beta = 0.0f;

cublasGemmEx(..., &alpha, ..., &beta, ..., 
             CUDA_R_16F,  // 数据类型是 __half
             ...);
```

**问题**：alpha 和 beta 是 `float` 类型，但传递给处理 `__half` 数据的 cuBLAS！

根据 cuBLAS 文档，当使用 `CUDA_R_16F` 时，alpha 和 beta 也应该是 `__half` 类型或者使用正确的 compute type。

### 下一步

1. 检查 alpha/beta 的类型转换
2. 验证 compute_type 的设置
3. 可能需要根据数据类型正确设置 alpha/beta

---

## v011 运行结果分析 (Oct 30, 2025)

### 🔴 关键问题发现：输出矩阵全为零！

从 `debug_logs/v11.log` 发现**所有 cuBLAS 矩阵乘法的输出都是 0**！

#### 第一次调用详细分析

```
[cublas_gemm MIXED LAYOUT Call #1]
  A: 16x256 (CM, stride=16)
  B: 256x64 (RM, stride=64)
  C: 16x64 (RM, stride=64)
  alpha=1, beta=0
  op_b=N, op_a=T
  cuBLAS call: gemm(op_b, op_a, n=64, m=16, k=256)
  
  Input A: 有非零值 ✓ (-14.555, -9.977, -19.984, ...)
  Input B: 有非零值 ✓ (0.059, 0.046, 0.011, ...)
  Output C: 全是 0.000 ✗✗✗
```

**预期**: C 应该 = A * B，有非零值
**实际**: C 全是 0

### 问题根源分析

#### cuBLAS 调用参数检查

当前调用：
```cpp
cublasGemmEx(
    handle,
    CUBLAS_OP_N,     // op_b (B 不转置)
    CUBLAS_OP_T,     // op_a (A 转置)
    64,              // n (输出列数)
    16,              // m (输出行数)
    256,             // k (内积维度)
    &alpha,          // 1.0
    B.data(),        // B: 256x64 RM, stride=64
    CUDA_R_16F,
    64,              // ldb = B.stride()
    A.data(),        // A: 16x256 CM, stride=16
    CUDA_R_16F,
    16,              // lda = A.stride()
    &beta,           // 0.0
    C.data(),
    CUDA_R_16F,
    64,              // ldc = C.stride()
    ...
);
```

#### cuBLAS 的解释

cuBLAS 假设所有矩阵都是列主序：

1. **B (op_b=N)**: 
   - cuBLAS 认为 B 是 (64×256) 列主序矩阵
   - ldb=64 表示列高
   - 但实际 B 是 (256×64) 行主序！
   - **问题**: B 的物理数据是 256×64 RM，cuBLAS 把它当成 64×256 CM 来读取

2. **A (op_a=T)**:
   - cuBLAS 认为 A 是 (16×256) 列主序矩阵，需要转置成 (256×16)
   - lda=16 表示列高
   - 实际 A 确实是 (16×256) CM ✓
   - **这个是对的**

3. **C**:
   - cuBLAS 输出 (64×16) 列主序矩阵
   - ldc=64 表示列高
   - 但我们期望的是 (16×64) 行主序
   - **问题**: 维度不匹配！

#### 根本问题

**cuBLAS 期望计算**: C_cm (64×16) = B_cm (64×256) * A_cm^T (256×16)

**我们想要计算**: C_rm (16×64) = A_cm (16×256) * B_rm (256×64)

**转换关系**: C_rm = A * B ⟺ C_cm^T = B^T * A^T

但是：
- B 是 RM (256×64)，作为 CM 解释时是 (64×256) - **维度错了！**
- 应该是 B^T，即 (64×256)，但 B 本身是 (256×64)

### 错误的根源

**问题在于对 RM 矩阵的理解**：

当 B 是 (256×64, RM, stride=64):
- 物理内存: 按行存储，每行 64 个元素
- cuBLAS 解释为 CM 时: 会认为是 (64×256) 矩阵，stride=64 是列高
- **但这是错的！** RM 矩阵 (256×64) 作为 CM 解释应该是 (64×256)，但 stride 应该是 64（列高），这恰好匹配
- **真正的问题**: 我们在用 B 的数据，但告诉 cuBLAS 它是 (64×256)，而实际数据布局是 (256×64) 的行主序

### 正确的理解

对于 RM 矩阵 B (m×n, RM):
- 物理存储: row0, row1, ..., row_{m-1}
- 要让 cuBLAS 正确使用，需要告诉它这是一个**转置的** CM 矩阵
- 即: B_rm (m×n) = B_cm^T (n×m)
- cuBLAS 看到的应该是 (n×m) CM，使用 CUBLAS_OP_T 后变成 (m×n)

**当前代码的问题**:
```cpp
cublasOperation_t op_b = LB == RM ? CUBLAS_OP_N : CUBLAS_OP_T;
```

对于 B (256×64, RM):
- op_b = CUBLAS_OP_N
- cuBLAS 认为 B 是 (64×256) CM，不转置
- **错误！** 应该认为 B 是 (64×256) CM，然后转置成 (256×64)

**正确的应该是**:
```cpp
cublasOperation_t op_b = LB == RM ? CUBLAS_OP_T : CUBLAS_OP_N;
```

### 修复方案

在混合布局 `cublas_gemm` 的 RM 输出分支中：

```cpp
if (LC == RM) {
    // 当前错误的实现
    cublasOperation_t op_a = LA == RM ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t op_b = LB == RM ? CUBLAS_OP_N : CUBLAS_OP_T;
    
    // 应该改为
    cublasOperation_t op_a = LA == RM ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = LB == RM ? CUBLAS_OP_T : CUBLAS_OP_N;
}
```

**原因**:
- RM 矩阵作为 CM 解释时，维度是转置的
- 需要使用 CUBLAS_OP_T 来"转置回来"
- CM 矩阵已经是 CM，使用 CUBLAS_OP_N

### 下一步

1. 修复 op_a 和 op_b 的逻辑
2. 重新测试
3. 验证输出矩阵不再是全零

---

## v011 最终修复方案 (Oct 30, 2025)

### 🎯 根本原因确认

经过深入分析 v11.log 和 cuBLAS 文档，确认问题的根本原因是：

**cuBLAS compute_type 与 alpha/beta 类型不匹配**

在 v011 的实现中，对于 FP16 (`__half`) 数据类型：

```cpp
cudaDataType_t cuda_data_type = CUDA_R_16F;
cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;  // ❌ 错误！

float alpha = 1.0f;  // float 类型
float beta = 0.0f;   // float 类型

cublasGemmEx(..., &alpha, ..., &beta, ..., compute_type, ...);
```

**问题分析**：
1. 当 `compute_type = CUBLAS_COMPUTE_16F` 时，cuBLAS 期望 alpha 和 beta 是 `__half*` 类型
2. 但代码传入的是 `float*` 类型
3. cuBLAS 将 `float*` 按 `__half*` 解释，读取到错误的值（很可能是 0）
4. 导致所有矩阵乘法结果为 0：`C = 0 * A * B + 0 * C = 0`

### cuBLAS 文档说明

根据 cuBLAS 文档：
- `CUBLAS_COMPUTE_16F`: 使用 FP16 计算，alpha/beta 必须是 `__half*`
- `CUBLAS_COMPUTE_32F`: 使用 FP32 计算，alpha/beta 必须是 `float*`
- `CUBLAS_COMPUTE_32F_FAST_16F`: 使用 FP32 累加但允许 FP16 输入，alpha/beta 是 `float*` ✓

### 修复实施

**文件**: `tiny-rocm-nn/include/tiny-cuda-nn/cublas_matmul.h`

修改了 3 个 `cublas_gemm` 函数：

#### 1. 更改 compute_type

```cpp
// 修复前
cublasComputeType_t compute_type = std::is_same<T, float>::value 
    ? CUBLAS_COMPUTE_32F 
    : CUBLAS_COMPUTE_16F;  // ❌

// 修复后
cublasComputeType_t compute_type = std::is_same<T, float>::value 
    ? CUBLAS_COMPUTE_32F 
    : CUBLAS_COMPUTE_32F_FAST_16F;  // ✓
```

#### 2. 更改 algorithm

```cpp
// 修复前
CUBLAS_GEMM_DEFAULT

// 修复后  
CUBLAS_GEMM_DEFAULT_TENSOR_OP
```

### 修复的优势

1. **类型匹配**：`float` 类型的 alpha/beta 与 `CUBLAS_COMPUTE_32F_FAST_16F` 匹配
2. **数值稳定性**：使用 FP32 累加，避免 FP16 累加的精度损失
3. **性能优化**：仍然使用 FP16 输入/输出，利用 Tensor Core 加速
4. **兼容性**：与 CUTLASS 的行为更接近（CUTLASS 也使用 FP32 累加）

### 技术细节

#### CUTLASS vs cuBLAS 的区别

**CUTLASS (官方 tiny-cuda-nn)**:
```cpp
using TypeAccumulator = cutlass::half_t;  // FP32 for accumulation
using TypeCompute = cutlass::half_t;      // FP32 for compute

// CUTLASS 内部使用 FP32 累加，即使数据类型是 FP16
```

**cuBLAS (tiny-rocm-nn 修复后)**:
```cpp
cudaDataType_t cuda_data_type = CUDA_R_16F;           // FP16 data
cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_16F;  // FP32 accumulation
```

两者现在行为一致：**FP16 输入/输出 + FP32 累加**

#### 为什么没有报错但结果错误？

1. cuBLAS 没有类型检查，不会报错
2. 将 `float*` 按 `__half*` 解释：
   - `float 1.0f` 的二进制：`0x3F800000`
   - 按 `__half` 解释前 16 位：`0x3F80` ≈ 0（或很小的值）
3. 导致 `alpha ≈ 0`，所以 `C = 0 * A * B = 0`

### 预期效果

修复后：
- ✓ cuBLAS 矩阵乘法输出非零值
- ✓ 梯度正确传播
- ✓ Loss 正常下降
- ✓ 训练收敛

### 总结

v011 的问题不是矩阵布局或转置逻辑的问题，而是 **cuBLAS API 使用不当**导致的类型不匹配。修复方法很简单：使用正确的 compute_type 以匹配 alpha/beta 的数据类型。

这个问题很隐蔽，因为：
1. 没有运行时错误
2. 参数检查通过
3. 只有通过查看实际输出值才能发现

---

## v012 训练收敛缓慢问题分析 (Oct 30, 2025)

### 观察到的现象

从 `debug_logs/v12.log`:
- ✓ Step 0: loss=8.768
- ✓ Step 10: loss=0.770 (下降 91%)
- ✓ Step 100: loss=0.149 (下降 81%)
- ✗ Step 1000: loss=0.126 (仅下降 15%)

**问题**: 训练在 step 100 之后几乎停滞

### cuBLAS 输出验证

从 v12.log 的矩阵输出：

#### Call #1 (输出层权重梯度)
```
Input A (16x256, CM): 有合理的非零值 (-14.555, -10.234, ...)
Input B (256x64, RM): 有合理的非零值 (0.059, 0.046, ...)
Output C (16x64, RM): 有合理的非零值 (-23.281, -18.469, ...)
```
✓ 矩阵乘法计算正确

#### Call #2 & #3 (隐藏层权重梯度)
```
Output C 样本值: (-2.484, -3.016, 7.594, ...) 和 (3.670, 3.229, ...)
```
✓ 数值范围合理

### 可能的原因分析

#### 1. 学习率问题 ❓

当前配置：
```json
"learning_rate": 1e-2  // 0.01
```

**分析**：
- 早期训练有效说明学习率不是太小
- 但后期停滞可能是学习率太大，导致在最优点附近震荡
- 或者需要学习率衰减

#### 2. 优化器状态问题 ⚠️

使用 Adam 优化器：
```json
"beta1": 0.9,
"beta2": 0.99
```

**潜在问题**：
- Adam 的动量累积可能在 cuBLAS 实现中有问题
- 需要检查优化器的梯度更新是否正确

#### 3. 梯度裁剪缺失 ⚠️

观察到的梯度值范围：
- 输出层：-37.5 到 -18.4
- 隐藏层：-7.9 到 12.2

**问题**：
- 梯度值偏大，可能导致参数更新过大
- CUTLASS 版本可能有隐式的梯度裁剪或归一化

#### 4. 数值精度问题 🔍

**关键发现**：使用 `CUBLAS_COMPUTE_32F_FAST_16F`

```cpp
cudaDataType_t cuda_data_type = CUDA_R_16F;           // FP16 输入/输出
cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_16F;  // FP32 累加
```

**可能的问题**：
- `FAST_16F` 模式可能使用 Tensor Core 的快速路径
- 这可能与 CUTLASS 的精确 FP32 累加有细微差异
- 建议尝试 `CUBLAS_COMPUTE_32F` 以获得完全的 FP32 精度

#### 5. 批次大小影响 ⚠️

当前：`batch_size = 256`

**分析**：
- 小批次可能导致梯度估计不稳定
- CUTLASS 版本可能使用更大的批次

### 调试建议

#### 优先级 1: 检查数值精度

修改 `cublas_matmul.h`：
```cpp
// 尝试完全 FP32 计算
cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;  // 而不是 FAST_16F
```

#### 优先级 2: 添加梯度监控

在 `fully_fused_mlp.cu` 的 `backward_impl` 中添加：
```cpp
// 检查梯度的范数
float grad_norm = compute_l2_norm(gradient_matrix);
if (step % 100 == 0) {
    std::cout << "Gradient norm: " << grad_norm << std::endl;
}
```

#### 优先级 3: 对比权重更新

添加调试代码检查：
1. 权重在每步的变化量
2. Adam 优化器的一阶和二阶动量
3. 实际的参数更新步长

#### 优先级 4: 验证反向传播

检查 `mlp_fused_backward` kernel：
- 激活函数梯度计算是否正确
- ReLU 的梯度（0 或 1）是否正确传播

### 与 CUTLASS 的关键差异

#### CUTLASS 优势

1. **Epilogue 融合**：
   - GEMM + 激活函数在一个 kernel
   - 减少内存访问，提高数值稳定性

2. **精确的 FP32 累加**：
   - 使用标准 FP32，不是 FAST_16F
   - 可能有更好的数值精度

3. **优化的内存布局**：
   - 可能有更好的缓存利用
   - 减少数据移动

#### cuBLAS 当前实现

1. **分离的操作**：
   - GEMM 和激活函数分开
   - 更多内存访问

2. **FAST_16F 模式**：
   - 可能牺牲一些精度换取速度
   - 在累加大量值时可能累积误差

### 下一步行动

1. **立即尝试**：
   - 将 `CUBLAS_COMPUTE_32F_FAST_16F` 改为 `CUBLAS_COMPUTE_32F`
   - 重新训练，观察 loss 曲线

2. **如果问题持续**：
   - 添加梯度范数监控
   - 检查权重更新的实际步长
   - 对比 CUTLASS 版本的中间值

3. **长期优化**：
   - 考虑实现自定义 CUDA kernel 融合 GEMM + 激活
   - 或者使用 cuDNN 的融合操作

### 预期结果

如果是精度问题，改用 `CUBLAS_COMPUTE_32F` 后应该看到：
- Loss 在 step 100-1000 之间继续显著下降
- 最终 loss 接近 CUTLASS 版本的结果

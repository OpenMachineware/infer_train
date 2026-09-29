# GPU 算子清单 (llama.cpp Metal)

## 核心算子

| 算子 | 文件 | 行数 | 实现方式 | 模板类型 |
|------|------|------|----------|----------|
| **MV (decode)** | mul_mv.metal | 3398 | 模板 | `r1ptg`, `block_q`, `dequantize_func` |
| **GEMM (batch)** | mul_mm.metal | 967 | 模板 | `block_q`, `dequantize_func`, SIMD矩阵类型 |
| **Flash Attention** | fa.metal | 2476 | 模板 | `block_t`, `dequantize_func`, seq_len并行 |
| **RMS Norm** | norm.metal | 318 | 模板 | 行并行 |
| **ROPE** | rope.metal | 333 | 模板 | 行并行 |
| **Softmax** | softmax.metal | 223 | 模板 | 行并行 |
| **Quantize** | quantize.metal | 480 | 模板 | 行并行 |
| **ArgSort** | argsort.metal | 479 | 模板 | 行并行 |
| **Binary Broadcast** | binbcast.metal | 228 | 模板 | 广播操作 |
| **Reduce** | reduce.metal | 228 | 模板 | 归约操作 |
| **Conv** | conv.metal | 724 | 模板 | 卷积 |
| **Pool** | pool.metal | 148 | 模板 | 池化 |
| **Upscale** | upscale.metal | 179 | 模板 | 上采样 |
| **Unary** | unary.metal | 400 | 模板 | 一元操作 |
| **WKV** | wkv.metal | 179 | 模板 | RWKV |
| **SSM** | ssm.metal | 476 | 模板 | Mamba |
| **Gated Delta Net** | gated_delta_net.metal | 264 | 模板 | |

## 模板架构

### MV (decode, 小batch N=1-5)
```cpp
template<short r1ptg, typename q_t, short chpb, void (*deq_t4)(...)>
kernel void kernel_mul_mv_ext_q4_f32_disp(...)
```
- `r1ptg`: 每个threadgroup处理的行数 (1-5)
- 不同格式通过不同的反量化函数特化

### GEMM (batch, N≥8)
```cpp
template<typename S0, typename S0_4x4, typename S0_8x8,
         typename S1, typename S1_2x4, typename S1_8x8,
         typename block_q, short nl, void (*dequantize_func)(...)>
kernel void kernel_mul_mm(...)
```
- 使用 SIMD matrix (simdgroup_half8x8)
- 通过反量化函数支持所有量化格式

### Flash Attention
```cpp
template<typename block_t, short QK, void (*deq_t4x4)(...)>
kernel void kernel_flash_attn_ext_kv_f16(...)
```
- 支持量化 KV cache
- seq_len 维度并行 (32 workgroups)

## 支持的量化格式

### MV 格式
- F32, F16, BF16
- Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
- IQ 系列: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS
- TQ 系列: TQ1_0, TQ2_0

### GEMM 格式
- 同 MV，加上 K-quant: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K

## 实现策略

llama.cpp 使用**统一模板**：
1. 一个模板内核
2. 通过函数常量/模板特化支持所有格式
3. 编译时生成所有变体

## 我们需要的实现

### 优先级 P0（核心）
1. MV 模板 - decode 路径
2. GEMM 模板 - batch 路径
3. Flash Attention 模板 - attention

### 优先级 P1（辅助）
4. RMS Norm 模板
5. ROPE 模板
6. Softmax 模板

### 优先级 P2（可选）
7. Quantize 模板
8. ArgSort 模板
9. Binary Broadcast 模板

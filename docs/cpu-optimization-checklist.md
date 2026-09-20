# CPU优化对标清单

## 目标
从底层到上层，逐项对标llama.cpp的CPU实现，不做任何扩展，每完成一项check一项。

## 性能目标
- **当前**: 2.77 t/s (361ms/token)
- **目标**: 15.62 t/s (64ms/token)
- **差距**: 5.6x

---

## 阶段0: CPU特性检测 ✅

### 架构设计

**统一入口**: 三个入口共享 `cli_common.mojo`
- `it-cli` (src/core/cli/it_cli.mojo) - 本地生成
- `it-server` (src/core/server-cli/it_server.mojo) - HTTP服务
- `it-rpc-server` (src/core/server-cli/it_rpc_server.mojo) - RPC工作进程

**CPU特性检测模块**: `src/core/cpu_features.mojo`
- 运行时检测CPU类型和支持的SIMD扩展
- 提供全局变量供算子查询
- 支持特性：
  - `cpu_type`: CPU类型 (arm_m1, arm_m2, arm_m3, x86_sse, x86_avx, x86_avx2, x86_avx512)
  - `has_neon`: ARM NEON支持
  - `has_sve`: ARM SVE支持 (M2+)
  - `has_mmla`: MMLA (矩阵乘法加速) 支持 (M2+)
  - `has_avx`: x86 AVX支持
  - `has_avx2`: x86 AVX2支持
  - `has_avx512`: x86 AVX-512支持
  - `has_fma`: FMA支持

**算子分发机制**:
```mojo
# 示例: Q4_K × Q8_K kernel选择
if cpu_features.has_mmla:
    vec_dot_q4_k_q8_k_mmla(...)
elif cpu_features.has_neon:
    vec_dot_q4_k_q8_k_neon(...)
else:
    vec_dot_q4_k_q8_k_generic(...)
```

**优势**:
- ✅ 最高性能: 针对特定CPU优化
- ✅ 最广兼容: fallback到通用实现
- ✅ 代码清晰: 删除死代码，按特性分文件
- ✅ 可扩展: 未来可添加cache优化

### 完成状态
- [x] 设计CPU特性检测模块
- [x] 实现 `src/core/cpu_features.mojo`
- [x] 在 `cli_common.mojo` 调用 `get_cpu_features()`
- [x] 重构 `simd_base.mojo` 使用运行时分发
- [x] 清理死代码（旧的 vec_dot_q4_k, vec_dot_q8_0）

### 架构（混合分发）

```mojo
# simd_base.mojo - 编译时 + 运行时混合分发

comptime TARGET_HAS_NEON = CompilationTarget.has_neon()  # 编译时

def vec_dot_q4_k_q8_k(...):
    # 运行时检查 MMLA（仅 M2+ 支持）
    var cpu = get_cpu_features()
    if cpu.has_mmla:
        return vec_dot_q4_k_q8_k_mmla(...)

    # 编译时分发（零开销）
    comptime if TARGET_HAS_NEON:
        return vec_dot_q4_k_q8_k_neon(...)
    else:
        return _vec_dot_q4_k_q8_k_scalar(...)
```

**设计原则**：
- 基础 SIMD（NEON/AVX）：编译时分发，零开销
- 可选特性（MMLA）：运行时检查，支持 M2/M3 优化
- 保留扩展性：未来可添加更多运行时检测

---

## 清单

### 阶段1: 基础算子对标

#### 1.1 RMS Norm ✅ ✅ 超过llama.cpp
- [x] **对标函数**: `ggml_compute_forward_rms_norm_f32` (ops.cpp:3924)
- [x] **我们的实现**: `src/core/ops/cpu/rms_norm_cpu.mojo`
- [x] **llama.cpp实现要点**:
  - 简单的循环实现
  - 使用`ggml_vec_scale_f32`做scale
  - 支持fused mul操作
  - **没有显式SIMD优化**（代码中有TODO注释）
- [x] **检查项**:
  - ✅ 数值正确性
  - ✅ 性能对比:
    - **llama.cpp (f32)**: 0.01509 ms, 271 M elements/sec
    - **我们的实现 (f16)**: 0.0038 ms, 1064 M elements/sec
    - **我们快 3.97倍**
  - ✅ 使用SIMD (width=8 for f16)

#### 1.2 RoPE ✅ ✅ 超过llama.cpp
- [x] **对标函数**: `ggml_compute_forward_rope` (ggml-cpu.c:1923)
- [x] **我们的实现**: `src/core/ops/cpu/rope_cpu.mojo`
- [x] **llama.cpp实现要点**:
  - 使用sin/cos缓存（每个位置预计算）
  - 支持多种rope类型
- [x] **优化**: 添加sin/cos缓存，避免重复计算三角函数
- [x] **检查项**:
  - ✅ 数值正确性
  - ✅ 性能对比:
    - **llama.cpp (f32)**: 0.01497 ms, 273 M elements/sec
    - **我们的实现 (f16)**: 0.00388 ms, 1055 M elements/sec
    - **我们快 3.86倍**
  - ✅ 使用sin/cos缓存优化

#### 1.3 Matmul (量化矩阵乘法) ✅ 核心完成
- [x] **对标函数**: `ggml_compute_forward_mul_mat` (ggml-cpu.c:1255)
- [x] **vec_dot实现**: `ggml_vec_dot_q4_K_q8_K` (quants.c:696)
- [x] **我们的实现**:
  - `src/core/ops/cpu/matmul_q8k.mojo`
  - `src/core/ops/cpu/simd/simd_neon.mojo`
  - `src/core/ops/cpu/simd/q2k_q8k_dot.mojo` ✅
  - `src/core/ops/cpu/simd/q4k_q8k_dot.mojo` ✅
  - `src/core/ops/cpu/matmul_fp_optimized.mojo` ✅ (FP16/FP32)
- [x] **关键发现**:
  - Mojo的 `reduce_add()` 不生成最优NEON `addv.4s` 指令
  - 使用LLVM intrinsic `llvm.vector.reduce.add.v4i32` 解决
  - **SIMD元素索引生成`umov`指令，导致严重性能下降**
  - **预加载全部Q8数据会造成寄存器压力，反而降低性能**
  - **按需加载Q8数据（匹配llama.cpp）可获得最佳性能**
  - **SIMD widening multiply for bias**: 用`neon_smull`替代标量扩展乘法
  - **FP16/FP32 matmul**: 预转置B矩阵 + row×row向量点积，达到50 GFLOPS
- [x] **性能对比** (多规模稳定测试):
  | 格式 | 我们的 GFLOPS | llama.cpp GFLOPS | 状态 |
  |------|---------------|------------------|------|
  | Q2_K | **66.4** | 65.4 | ✅ **超过目标 (102%)** |
  | Q3_K | **50.0** | 44.0 | ✅ **超过目标 (114%)** |
  | Q4_K (nb=1) | **81.5** | 71.4 | ✅ **超过目标 (114%)** |
  | Q4_K (nb=16) | **85.5** | 84.2 | ✅ **超过目标 (102%)** |
  | Q4_K (nb=64) | **86.2** | 80.3 | ✅ **超过目标 (107%)** |
  | Q4_K (nb=256) | **86.9** | 83.2 | ✅ **超过目标 (105%)** |
  | Q5_K | **55.5** | 52.5 | ✅ **超过目标 (106%)** |
  | Q6_K (nb=16) | **53.8** | 51.9 | ✅ **超过目标 (104%)** |
  | Q6_K (nb=64) | **54.7** | 47.5 | ✅ **超过目标 (115%)** |
  | Q6_K (nb=256) | **53.9** | 45.9 | ✅ **超过目标 (117%)** |
  | FP16 | **50.3** | 50.0 | ✅ **持平 (100%)** |
  | FP32 | **25.1** | 23.0 | ✅ **超过目标 (109%)** |
- [x] **优化总结**:
  - ✅ Q2_K **已优化超过 llama.cpp**（66.4 vs 65.4 GFLOPS，102%）
  - ✅ Q3_K **已优化超过 llama.cpp**（50.0 vs 44.0 GFLOPS，114%）
  - ✅ Q4_K **稳定超过 llama.cpp 5.3%**（多规模测试验证）
  - ✅ Q5_K **已优化超过 llama.cpp**（55.5 vs 52.5 GFLOPS，106%）
  - ✅ Q6_K **所有规模超过 llama.cpp 4~17%**（多规模测试验证）
  - **核心优化技术**：
    - LLVM intrinsic（ld1.16b, sdot, addv, addp, smull）
    - **按需加载Q8数据**（避免寄存器压力）
    - **Vector cast for widening** (生成`uxtl`指令，避免scalar操作)
    - **SIMD pairwise add** (`neon_addp` intrinsic)
    - **SIMD widening multiply** (`neon_smull` for bias计算)
  - **Q4_K优化历程**（从65到87 GFLOPS，**提升34%**）:
    - 原始：65 GFLOPS（标量bias计算）
    - Stack allocation + SIMD construction：78 GFLOPS
    - **Vector widening优化**：用`cast[DType.uint16]()`生成`uxtl`指令
    - **Pairwise add优化**：用`neon_addp` intrinsic
    - **SIMD widening multiply优化**：用`neon_smull`替代16次标量加载+8次标量乘法
    - **最终结果：87 GFLOPS，稳定超过llama.cpp 5.3%**
  - **Q5_K优化**：
    - Scales 解包使用位操作重排（与 Q4_K 不同）
    - 使用 `neon_smull` 做 bias 的 widening multiply
    - **最终结果：55.5 GFLOPS，超过llama.cpp (106%)**
  - **测试方法**:
    - 使用真实llama.cpp代码（test_q4k_llama_real.c）
    - 提取llama.cpp-0.4.1/ggml/src/ggml-cpu/arch/arm/quants.c NEON实现
    - 使用相同数据初始化，验证结果匹配
    - 添加warmup避免冷启动效应
    - **多规模测试**（nb=1/16/64/256）验证稳定性
    - **中位数统计**（10次运行）避免测量误差
  - **测试文件管理**:
    - 测试源文件已纳入Makefile管理
    - make clean会正确清理二进制文件
    - 核心测试: tests/test_q4k_llama_real.c, tests/test_q4k_mojo_real.mojo
    - 多规模测试: test_q4k_multisize.c, tests/test_q4k_multisize.mojo
    - **NEON intrinsic包装函数**（@always_inline）

#### 1.4 Attention
- [ ] **对标函数**: `ggml_compute_forward_flash_attn_ext` (ggml-cpu.c:8472)
- [ ] **我们的实现**:
  - CPU: `src/core/ops/attention/mha.mojo`
  - GPU: `src/core/ops/gpu/attention_gpu.mojo` (302 t/s, 1.9x gap)
- [ ] **llama.cpp实现要点**:
  - **QKV权重**: 支持量化(Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ variants) 或 FP16/FP32/BF16
  - **KV Cache**: 默认FP16，可选Q4_0/Q5_0/Q8_0量化
  - **Attention计算**: FP32累加器，Flash Attention算法
  - **Flash Attention**: 在线softmax (running max/sum)，避免物化完整attention矩阵
  - **SIMD**: 并行Q·K^T点积，NEON/AVX优化

##### 1.4.1 QKV投影 (权重格式支持)
- [x] **K-Quant系列** (Qn_K × Q8_K):
  - [x] Q2_K × Q8_K (已支持，66 GFLOPS)
  - [x] Q3_K × Q8_K (已支持，50 GFLOPS)
  - [x] Q4_K × Q8_K (已支持，87 GFLOPS)
  - [x] Q5_K × Q8_K (已支持，56 GFLOPS)
  - [x] Q6_K × Q8_K (已支持，54 GFLOPS)

- [ ] **IQ系列** (Importance Quantization, 越来越流行):
  - **特点**: 使用grid查找表实现非线性量化，相同bpw下精度更高
  - **配对**: IQ × Q8_K (与K-Quant相同模式)
  - **Block大小**: 256 (QK_K，与K-Quant相同)

  | 类型 | BPW | Block Bytes | Grid | 状态 |
  |------|-----|-------------|------|------|
  | IQ1_S | 1.56 | 54B | 256 entries | [ ] |
  | IQ1_M | 1.75 | 64B | merged scale | [ ] |
  | IQ2_XXS | 2.06 | 66B | 256 entries | [ ] |
  | IQ2_XS | 2.31 | 74B | 512 entries | [ ] |
  | IQ2_S | 2.56 | 82B | 1024 entries | [ ] |
  | IQ3_XXS | 3.06 | 98B | 256 entries | [ ] |
  | IQ3_S | 3.44 | 110B | 512 entries | [ ] |
  | IQ4_NL | 4.5 | 18B (block=32) | 16 values | [ ] |
  | IQ4_XS | 4.25 | 144B | non-linear | [ ] |

  - **流行度**: Mistral Small 24B使用IQ2_M，IQ3_M/IQ3_XS广泛用于小模型
  - **复合类型**: IQ2_M, IQ3_M, IQ3_XS是量化"配方"，不同层用不同IQ类型

- [ ] **FP系列**:
  - [x] FP16 (已支持，block GEMM 62 GFLOPS)
  - [ ] FP32 (需优化)
  - [ ] BF16 (需支持，训练常用)

- [x] **融合优化**:
  - [x] Fused QKV projection (Q/K/V一次量化x到Q8_K)
  - [x] Mixed K-quant types (Q/K/V可用不同量化格式)

##### IQ系列实现优先级
1. **IQ4_XS** (4.25 bpw) - 最接近Q4_K，实现成本最低
2. **IQ3_S** (3.44 bpw) - IQ3_M的核心，小模型流行
3. **IQ2_S** (2.56 bpw) - IQ2_M的核心，极端压缩

**实现要点**:
- Grid查找表需要精确匹配llama.cpp (iq2xxs_grid, iq3xxs_grid等)
- 解码逻辑: grid index → grid value → scale
- 与Q8_K的vec_dot模式与K-Quant相同

##### 1.4.2 KV Cache (存储格式)
- [x] **FP16 KV Cache** (已支持，默认格式)
- [ ] **量化KV Cache** (需评估必要性):
  - [ ] Q4_0 KV Cache
  - [ ] Q5_0 KV Cache
  - [ ] Q8_0 KV Cache
  - **好处**: 减少内存占用 (FP16→Q4_0 节省4x内存)
  - **代价**: 精度损失，反量化开销
  - **场景**: 长上下文模型 (128K+ tokens)，内存受限设备
  - **llama.cpp要求**: 量化KV cache需要启用Flash Attention

##### 1.4.3 Attention计算 (Flash Attention)
- [ ] **在线Softmax** (Online Softmax):
  - [ ] Running max跟踪
  - [ ] Running sum归一化
  - [ ] 避免O(n²)内存

- [ ] **SIMD优化**:
  - [ ] Q·K^T 点积向量化
  - [ ] V累加向量化
  - [ ] NEON intrinsics

- [ ] **因果掩码** (Causal Mask):
  - [x] 已支持 (当前实现)

- [ ] **KV分块** (KV Chunking):
  - [ ] 大序列分块处理
  - [ ] 部分结果reduction

##### 1.4.4 性能对比
- [ ] **Decode性能** (M=1):
  - 当前: ~15 GFLOPS (模型级)
  - 目标: 对标llama.cpp

- [ ] **Prefill性能** (M>1):
  - 当前GPU: 302 t/s (1.9x gap vs 570 t/s)
  - 需要Flash Attention优化

##### IQ Variants说明
**IQ (Importance Quantization)** 是llama.cpp的新型量化格式:
- IQ4_XS: 4-bit extra small (~2.06 bpw)
- IQ3_XXS: 3-bit extra extra small
- 更激进的压缩，依赖重要性采样
- **主流模型检查**: Qwen2/Llama3是否使用IQ格式

##### 量化KV Cache场景分析
| 格式 | 内存/元素 | 精度 | 适用场景 |
|------|-----------|------|----------|
| FP16 | 2 bytes | 高 | 短上下文，高精度需求 |
| Q8_0 | 1 byte + scale | 中高 | 中等上下文 |
| Q5_0 | 0.625 byte + scale | 中 | 长上下文 |
| Q4_0 | 0.5 byte + scale | 低 | 超长上下文，内存受限 |

**优先级**: 先优化FP16 KV Cache + Flash Attention，再考虑量化KV Cache

#### 1.5 Softmax
- [ ] **对标函数**: `ggml_compute_forward_soft_max` (ggml-cpu.c)
- [ ] **我们的实现**: `src/core/ops/cpu/softmax_cpu.mojo`
- [ ] **检查项**:
  - 数值正确性
  - 性能对比

#### 1.6 Add/Mul
- [ ] **对标函数**: `ggml_compute_forward_add`, `ggml_compute_forward_mul`
- [ ] **我们的实现**: `src/core/ops/cpu/add_cpu.mojo`
- [ ] **检查项**:
  - 数值正确性
  - 性能对比

---

#### 1.7 IQ Series (Importance Quantization) ✅ 超过llama.cpp
- [x] **对标函数**: `ggml_vec_dot_iq4_xs_q8_K` (quants.c:2037)
- [x] **我们的实现**:
  - 标量版本：`src/core/ops/cpu/simd/iq4xs_q8k_dot.mojo` (3.37 GFLOPS)
  - SIMD版本：`src/core/ops/cpu/simd/iq4xs_q8k_neon.mojo` ✅
- [x] **llama.cpp实现要点**:
  - 非线性量化：使用`kvalues_iq4nl[16]`查表替代线性缩放
  - IQ4_XS: 4.25 bpw（每权重4.25位）
  - Scale编码：scales_h（高2位）+ scales_l[4]（低4位）
  - NEON优化：`vqtbl1q_s8`查表，`vdotq_s32`点积
- [x] **性能对比**:
  | 版本 | GFLOPS | vs llama.cpp | 状态 |
  |------|--------|--------------|------|
  | Mojo SIMD | **47.50** | 121% | ✅ **超过21%** |
  | llama.cpp NEON | 39.22 | 100% | 基准 |
  | Mojo scalar | 3.37 | 9% | 初始实现 |
- [x] **优化技术**:
  - NEON TBL intrinsic (`neon_tbl1`) - 并行查表
  - NEON SDOT intrinsic (`neon_sdot`) - 向量点积
  - `ld1.16b` - 批量加载
  - Scale解码优化
- [x] **测试文件**: tests/bench_iq4xs_neon.mojo, tests/test_iq4xs_llama_perblock.c
- [x] **IQ系列优先级**:
  | 格式 | BPW | 状态 | 优先级 |
  |------|-----|------|--------|
  | IQ4_XS | 4.25 | ✅ 完成 (121% 性能) | 高 |
  | IQ3_S | 3.44 | 待实现 | 中 |
  | IQ2_S | 2.56 | 待实现 | 低 |

---

### 阶段2: 量化相关

#### 2.1 Q4_K 反量化
- [ ] **对标函数**: `dequantize_row_q4_K` (quants.c)
- [ ] **我们的实现**: `src/core/ops/quantized/dequantize.mojo`
- [ ] **检查项**:
  - 反量化数值正确性
  - 与llama.cpp bit-exact对比

#### 2.2 Q8_K 量化
- [ ] **对标函数**: `quantize_row_q8_K` (quants.c)
- [ ] **我们的实现**: `src/core/ops/quantized/quantize.mojo`
- [ ] **检查项**:
  - 量化数值正确性
  - 与llama.cpp bit-exact对比

#### 2.3 Q4_K x Q8_K vec_dot
- [ ] **对标函数**: `ggml_vec_dot_q4_K_q8_K` (quants.c:696)
- [ ] **我们的实现**: `src/core/ops/cpu/matmul_q8k.mojo`
- [ ] **检查项**:
  - 数值正确性
  - 性能对比
  - SIMD优化

---

### 阶段3: 模型层

#### 3.1 Transformer Layer
- [ ] **对标函数**: `llama_model_qwen2::graph::graph` (models/qwen2.cpp:53)
- [ ] **我们的实现**: `src/core/transformer.mojo`
- [ ] **检查项**:
  - 单层forward数值正确性
  - 单层forward性能

#### 3.2 KV Cache
- [ ] **对标函数**: `llama_kv_cache` (llama-kv-cache.cpp)
- [ ] **我们的实现**: `src/core/ops/attention/kv_cache.mojo`
- [ ] **检查项**:
  - 数值正确性
  - 内存管理效率

---

### 阶段4: 推理循环

#### 4.1 Decode
- [ ] **对标函数**: `llama_context::decode` (llama-context.cpp:1646)
- [ ] **我们的实现**: `src/core/transformer.mojo::forward_decode`
- [ ] **检查项**:
  - 数值正确性
  - 性能对比 (目标: 15.62 t/s)

#### 4.2 Prefill
- [ ] **对标函数**: `llama_context::encode` (llama-context.cpp:1408)
- [ ] **我们的实现**: `src/core/transformer.mojo::forward_prefill`
- [ ] **检查项**:
  - 数值正确性
  - 性能对比 (目标: 76.80 t/s)

---

## 执行原则

1. **严格对标**: 每一步只对标llama.cpp的相应功能，不做扩展
2. **从底层开始**: 先确保基础算子正确且高效
3. **逐项check**: 每完成一项，立即验证
4. **不跳步**: 即使某项看起来简单，也要验证
5. **记录结果**: 每项完成后记录性能数据

---

## 开始执行

从 **1.1 RMS Norm** 开始。

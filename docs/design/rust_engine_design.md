# Rust Inference Engine 设计文稿

## 背景

Mojo CPU路径因循环开销问题（11-16x vs C）放弃，决定用Rust重写CPU推理引擎。

**商业目标**：
- CPU和GPU性能都超过llama.cpp（≥100%，不是95%）
- 扩展vLLM功能（多请求批处理）
- 找熟人推广，在他们的公司尝试使用

**平台范围**：macOS only（Metal），其他平台CPU fallback

## 核心架构

```
┌─────────────────────────────────────────┐
│        InferenceEngine (Rust)           │
├─────────────────────────────────────────┤
│  CPU Kernels (NEON/AVX2/Scalar)         │
│  GPU Kernels (Metal/CUDA)               │
│  Unified Memory Manager                 │
├─────────────────────────────────────────┤
│  Single-User Fast Path                  │ ← 本地部署：零开销
│  Multi-User Batch Path                  │ ← 服务器：vLLM特性
├─────────────────────────────────────────┤
│  Pipeline Parallel (Phase 3)            │ ← 异构扩展
│  Heterogeneous Memory Pool (Phase 3)    │
└─────────────────────────────────────────┘
```

## Phase 1: MVP（本地单机）

### 目标
- CPU NEON kernels性能 ≥ llama.cpp
- Metal GPU kernels可用
- 单请求推理可用

### CPU Kernels

**量化格式支持**：
- K-quant: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
- IQ系列: IQ1_S, IQ1_M, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_XS, IQ4_NL
- 量化权重 + FP16激活

**关键优化**（从Mojo经验）：
1. **Q8_K量化复用**：
   - 激活量化到Q8_K一次，多投影复用（gate+up, QKV）
   - 量化开销：~10µs vs 多次量化~100µs

2. **Threaded kernel**：
   - 阈值：N >= 2048（threading才有效益）
   - Batch prefill: M > 1时用threaded kernel
   - Decode: M = 1时用单线程kernel

3. **IQ4_XS处理**：
   - 非K-quant格式，但用Q8_K vec_dot kernel
   - Block size: 136 bytes
   - Dispatch: ggml_type == 23

**性能目标**（参考Mojo benchmark）：
| 格式 | GFLOPS | vs llama.cpp |
|------|--------|-------------|
| Q4_K | 117-121 | +5-21% |
| Q5_K | ~95 | +10% |
| Q6_K | ~85 | +10% |
| IQ4_XS | ~70 | +21% |
| IQ3_S | ~20 | +11% |

**验证方法论**：
- **自底向上**：每个kernel ≥100% llama.cpp再向上构建
- **汇编对比**：确保生成相同的SIMD指令
- **测试数据**：使用llama.cpp verbatim的测试数据（不简化）

### GPU Kernels (Metal)

**技术栈**：Rust + `metal-rs`

**Kernel列表**：
- Matmul（量化权重 + FP16激活）
- Flash Attention（prefill优化）
- RMSNorm, Softmax
- RoPE

**性能目标**：
- Prefill: 接近llama.cpp（Metal路径）
- Decode: GPU比CPU慢是正常的（小batch时），但可接受

**Unified Memory**：
- CPU和GPU共享内存，零拷贝
- macOS独有优势

### 执行模式

**Single-User Fast Path**：
```rust
fn forward_fast(&mut self, tokens: &[Token]) -> Logits {
    // 无调度器，无锁，无caching检查
    // 连续KV Cache（非paged）
    // 直接执行，接近llama.cpp开销
}
```

**设计原则**：
- 本地部署性能 ≥ llama.cpp
- 不为vLLM特性牺牲单用户性能

## Phase 2: 服务器部署

### Multi-User Batch Path

**核心特性**：
1. **PagedAttention + Block Manager**
   - KV Cache分页管理，类似OS虚拟内存
   - 共享前缀（system prompt）自动去重
   - 避免预分配 `max_context × n_requests` 浪费

2. **Prefix Caching**
   - 多用户共享system prompt → KV Cache复用
   - 吞吐量翻倍（实际场景）
   - 实现：KV block引用计数 + LRU淘汰

3. **Continuous Batching**
   - 不等所有请求完成，有空位就插入新请求
   - 避免"长prompt阻塞所有短请求"

4. **快慢路径分离**
   ```rust
   enum ExecutionMode {
       SingleUser,      // 本地部署：无调度开销
       MultiUser,       // 服务器：完整vLLM特性
   }
   ```

**单用户开销分析**：
| 特性 | 单用户开销 | 原因 |
|------|-----------|------|
| PagedAttention | 2-5% | 指针跳转 vs 连续内存 |
| Continuous Batching | <1% | if分支，CPU预测准 |
| Prefix Caching | <1% | LRU查找 |

**目标**：单用户模式性能 ≥ llama.cpp

### 异步架构

**技术栈**：Rust + tokio

**设计**：
- async/await + tokio运行时
- 单请求时零开销（快路径）
- 多请求时自动调度

**vs Mojo**：
- Mojo pthread: ms级调度开销
- Rust tokio: µs级调度开销
- 解决"单请求时线程池吃掉收益"问题

### 其他vLLM特性（可选）

**Chunked Prefill**（Phase 2.5）：
- 长prompt分块执行（每块32 tokens）
- 插空处理decode请求
- 避免长prompt阻塞

**KV Cache Quantization**（Phase 2.5）：
- Q8/Q6/Q4量化KV Cache
- 内存带宽敏感场景收益大
- 精度损失需权衡

**Speculative Decoding**（Phase 3）：
- 小模型预测，大模型验证
- 解码速度×3（高正确率场景）
- 需要：两模型 + 验证逻辑

**Attention Sink**（Phase 3）：
- 流式生成时保留前几个token的KV
- 解决滚动窗口attention崩坏

## Phase 3: 分布式/异构

### Heterogeneous Architecture（AMD AI Max + NVIDIA 5090）

**场景**：
- AI Max: 192GB统一内存（放得下整个大模型）
- 5090: 24-32GB显存，但计算快10倍
- 组合：AI Max做内存池，5090做计算

**Pipeline Parallel + Local Weights**：
```
AI Max (内存服务器)          5090 (计算节点)
┌──────────────────┐         ┌──────────────────┐
│ 192GB 内存池      │ ←RDMA──→│ 24GB 显存         │
│ 激活数据/KV Cache │ 零拷贝  │ 所有计算          │
│ 模型权重(可选)    │         │ 从AI Max直接读取  │
└──────────────────┘         └──────────────────┘
```

**通信方式**：
1. **同机PCIe**（最优）：
   - AI Max主板插5090
   - 统一内存 + PCIe = 零拷贝

2. **RDMA网络**：
   - GPU Direct RDMA
   - 带宽：~12.5GB/s（100Gbps）
   - 延迟：<10µs

**调度策略**：
```rust
struct PipelineSchedule {
    layers_on_device_a: Range<usize>,  // AI Max: 0-15
    layers_on_device_b: Range<usize>,  // 5090: 16-31
    activation_channel: Channel<Tensor>,  // KB级
}
```

**性能分析**：
- 激活传输：8KB per token
- 网络延迟：可忽略（µs级）
- 计算：主导因素（ms级）

### 同构GPU集群

**对标vLLM**：
- Tensor Parallel（层内切分）
- Pipeline Parallel（层间切分）
- RDMA + GPU Direct通信

**RPC模式**（不推荐）：
- 传统网络传输权重太慢
- 40GB × 10Gbps = 32秒
- 只适合偶发调用，不适合持续推理

## 技术选型

### Rust GPU Libraries

| 平台 | Library | 性能 vs Mojo/C++ | 成熟度 |
|------|---------|------------------|--------|
| Metal | metal-rs | = Mojo | 社区库，够用 |
| CUDA | cust, rust-cuda | = C++ | 成熟 |
| ROCm | rocm-rs | = C++ | 较弱，可能需FFI |

**关键点**：
- GPU Kernel性能与主机语言无关（同样的shader/PTX）
- API调用开销可忽略（µs级 vs ms级kernel）
- 性能不是问题，生态成熟度是

### 量化Kernel实现

**参考**：llama.cpp NEON kernels

**验证方法**：
1. 复制llama.cpp的测试数据（不简化）
2. 逐函数对比输出
3. 汇编级对比SIMD指令

**关键代码路径**（从Mojo移植）：
- `ggml_vec_dot_q2_k_q8_k` → Rust NEON
- `ggml_vec_dot_q4_k_q8_k` → Rust NEON
- `quantize_row_q8_k` → Rust NEON

## 性能目标总结

### CPU Kernels

| 格式 | Mojo GFLOPS | 目标Rust GFLOPS | vs llama.cpp |
|------|-------------|----------------|-------------|
| Q4_K | 117-121 | ≥120 | ≥105% |
| Q5_K | ~95 | ≥95 | ≥110% |
| Q6_K | ~85 | ≥85 | ≥110% |
| IQ4_XS | ~70 | ≥70 | ≥121% |

### 端到端

| 场景 | 目标 | vs llama.cpp |
|------|------|-------------|
| 本地单请求（7B Q4_K） | ≥10 t/s | ≥100% |
| 服务器多请求 | 吞吐量优先 | vLLM级别 |

### GPU

| 场景 | 目标 | vs llama.cpp |
|------|------|-------------|
| Prefill | 300+ t/s | ~100% |
| Decode | CPU的70% | 可接受（小batch） |

## 开发优先级

**Phase 1**（2-4周）：
- [ ] CPU NEON kernels（量化格式）
- [ ] Metal GPU kernels
- [ ] Single-user fast path
- [ ] 性能验证 ≥ llama.cpp

**Phase 2**（2-4周）：
- [ ] Multi-user batch path
- [ ] PagedAttention
- [ ] Prefix Caching
- [ ] Continuous Batching

**Phase 3**（按需）：
- [ ] Pipeline Parallel
- [ ] Heterogeneous memory pool
- [ ] RDMA通信

## 参考资料

### Code

- `src/core/ops/quantized/qweight.mojo` - 量化投影dispatch
- `src/core/ops/cpu/matmul_q8k_threaded.mojo` - 线程化kernel，fused projection
- `src/core/ops/attention/mha.mojo` - Attention实现，QKV fused
- `llama.cpp/src/ggml-quants.c` - NEON kernel参考

### Memory

- `Q4_K Final Performance` - 117-121 GFLOPS，+5-21% vs llama.cpp
- `IQ Series Final Status` - 5个格式超过llama.cpp 11-67%
- `Mojo Loop Overhead` - 11-16x vs C，放弃CPU路径
- `vLLM Design Strategy` - vLLM特性优先级
- `Heterogeneous Distributed Architecture` - 异构架构设计

### Methodology

- **自底向上验证**：每个layer ≥100% llama.cpp再向上
- **汇编对比**：确保SIMD指令相同
- **测试数据**：llama.cpp verbatim数据，不简化

---

**创建日期**：2026-09-22
**状态**：设计完成，待实现
**下一步**：Phase 1 CPU kernels

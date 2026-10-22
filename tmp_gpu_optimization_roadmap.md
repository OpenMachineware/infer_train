# GPU Decode Performance Optimization Roadmap

## Current Status (2026-10-22 Session 8)

**Performance**:
- Current: **105 t/s** (Devstral-Small-2-24B Q4_K_M) - NaN issues
- llama.cpp: **227 t/s** (same model)
- Gap: **2.16x**

**Status**: 🎯 **KERNEL DISPATCH FIXED** - But correctness issues remain

### Session 8 Progress
1. ✅ Fixed dispatch: Changed from 1D (width:32, height:1) to 2D (width:32, height:nsg)
2. ✅ Kernel formula matches llama.cpp: `first_row = (tgpig.x * NSG + sgitg) * N_R0`
3. ✅ Simple correctness test passes (zero weights → zero output)
4. ❌ Full model produces NaN - additional kernel issues to debug

### Dispatch Configuration Fix
**Before (wrong)**: 1D dispatch, each threadgroup = 32 threads, 1 simdgroup
**After (correct)**: 2D dispatch, each threadgroup = 32 × nsg threads, nsg simdgroups

For Q4_K: `MTLSize { width: 32, height: 2, depth: 1 }` creates 2 simdgroups

### N_R0 and NSG Values (from llama.cpp)
| Quant Type | N_R0 | NSG | Rows/TG |
|------------|------|-----|---------|
| Q4_K       | 2    | 2   | 4       |
| Q5_K       | 1    | 2   | 2       |
| Q6_K       | 2    | 2   | 4       |
| Q2_K       | 4    | 2   | 8       |
| Q3_K       | 2    | 2   | 4       |

### Previous Session Findings (Session 7)

**Root Cause Found**: Q4_K kernel has 7-8x lower thread utilization
- Our kernel used scalar loop with stride 32, leaving 28/32 threads idle
- llama.cpp uses 4-block batching with stride 4, utilizing all 32 threads
- Fixed by rewriting kernel to match llama.cpp's approach

### Key Finding from Session 5
**Fusion is NOT the main cause of the gap.**

Evidence:
- llama.cpp doesn't have more fusion for decode than we do
- MUL_MAT+Add fusion only works for MMA (multi-row), not decode (ne[1]=1)
- GEMV+Add fusion exists in our code but causes 19% regression (disabled)

### llama.cpp Benchmark Results (2026-10-22)
```
| qwen3 0.6B Q4_K - Medium | 380.92 MiB | 596.05 M | BLAS,MTL | 8 | tg10 | 239.59 ± 2.87 t/s |
```

### Key Differences Observed

**llama.cpp uses:**
1. ✅ Residency Sets (`use residency sets = true`)
2. ✅ Shared Buffers (`use shared buffers = true`)
3. ✅ MMA kernels for MUL_MAT+Add fusion (multi-row only)
4. ❌ NO GateUp fusion (separate kernels)
5. ❌ NO SiLU+Mul fusion (separate kernels)

**Our current state:**
1. ❌ Residency Sets blocked (metal crate limitation)
2. ⚠️ Shared buffers partial
3. ❌ MMA kernels disabled (performance issues)
4. ❌ FusedGateUp disabled (NaN issues)
5. ✅ FusedSwiGLU active
6. ✅ FusedAddNorm active
7. ✅ FusedQKV active

### Fusion Analysis - llama.cpp vs Ours

**llama.cpp MUL_MAT+Add Fusion:**
- Only for **MMA kernels** (multi-row operations)
- NOT applicable to decode (single row, ne[1]=1)
- Conditions: `ne[1] >= rows_min` (2-3) and `<= 16`
- Our decode: ne[1]=1, so this fusion doesn't help

**llama.cpp FFN Fusion:**
- NO gate-up fusion found
- Separate SILU, MUL operations
- We have FusedSwiGLU (SiLU+Mul only, not GEMV)

**Conclusion**: Fusion is NOT the main cause of the gap

**Model**: Qwen3-0.6B-UD-Q4_K_XL
- Hidden size: 1024
- Intermediate size: 4096
- Layers: 28
- n_heads: 16
- n_kv_heads: 8 (GQA ratio = 2)
- head_dim: 128

---

## Dispatch Count Analysis (2026-10-22)

**Per token dispatches (INFER_TRACE_DAG):**
- Total: 620 dispatches/token
- Breakdown:
  - NoOp (fused): 156 - these are skipped but counted in trace
  - Gemv: 124 - main compute
  - KVCacheUpdate: 62 - K and V updates
  - RoPE: 62 - Q and K rotations
  - RmsNorm: 62 - attn and FFN norms
  - Add: 60 - residual connections
  - FlashAttention: 31 - 1 per layer
  - FusedQKV: 31 - QKV projection (saves 2 GEMVs per layer)
  - FusedSwiGLU: 31 - SiLU+Mul fusion
  - FusedAddNorm: 1 - output norm
  - EmbeddingLookup: 1

**Expected dispatches if all fused:**
- Per layer: ~14 (FusedAddNorm + FusedQKV + RoPE×2 + KV×2 + FlashAttn + Add + FusedAddNorm + GateUp + FusedSwiGLU + Down + Add)
- Total: 28 × 14 = 392 + 3 = 395

**Actual**: 620 dispatches
**Gap**: 225 extra dispatches

**Missing fusions:**
1. FusedGateUp DISABLED → 56 extra GEMVs
2. FusedGemvAdd DISABLED → 56 extra Adds
3. FusedNormLinear DISABLED → 28 extra ops

---

## Performance Gap Analysis

### Possible Causes of 2.78x Gap

1. **Dispatch Overhead**
   - Each dispatch has overhead (~10-20µs)
   - 620 dispatches × 15µs = 9.3ms
   - Target time: 1000ms/239.59t/s = 4.17ms/token
   - Our time: 1000ms/86.21t/s = 11.6ms/token
   - Gap: 7.4ms
   - Dispatch overhead could be ~1-2ms, not the main issue

2. **Kernel Performance**
   - GEMV kernels: need to verify vs llama.cpp
   - Flash Attention: we use f16, llama.cpp might use optimized path
   - Element-wise ops: might be unoptimized

3. **Memory Subsystem**
   - No Residency Sets → potential page faults
   - Buffer management overhead
   - Cache locality issues

4. **Metal Pipeline Compilation**
   - llama.cpp pre-compiles all pipelines
   - We might have JIT compilation overhead

### Kernel Implementation Comparison

**llama.cpp Q5_K kernel (lines 1724-1728):**
```metal
sumf[row] += dh[0] * (sc8[0] * (...) + sc8[1] * (...)) -
           dh[1] * (sumy[0] * sc8[2] + ...);
```

**Our Q5_K kernel (lines 390-394):**
```metal
sumf += dh[0] * ((sc8[0]-32.f) * (...) + (sc8[1]-32.f) * (...)) +
       dh[1] * (sumy[0] * (sc8[2]-32.f) + ...);
```

**Analysis:**
- Both use `N_R0_Q5_K = 1` (1 row per simdgroup)
- Both use Y-vector caching and float4 accumulation
- The scale formula difference may be equivalent (different scale encoding)

### Kernel Timing Results - Session 6

**Per-kernel timing (Layer 0, Qwen3-0.6B Q4_K)**:
```
RMS Norm: 317µs
Q projection (1024x2048): 501.375µs
K projection (1024x1024): 231.083µs
V projection (1024x1024): 281.417µs
RoPE: 219.916µs
Flash Attention (seq_k=10): 211.75µs
Output projection (1024x1024): 633.875µs
FFN gate (1024x3072): 380.208µs
FFN up (1024x3072): 417µs
FFN down (3072x1024): 325.042µs
```

**Per-layer total (estimated)**: ~4.1ms
**28 layers total**: ~115ms (if run separately)
**Actual decode time**: 12.6ms/token

**Batching speedup**: 9x (from single command buffer)

### Performance Gap Analysis

**Current**: 79.37 t/s (12.6ms/token)
**Target**: 239 t/s (4.2ms/token)
**Gap**: 3x

**Where is the 8.4ms gap coming from?**

1. **GEMV kernels**: Total ~2.5ms per layer × 28 layers = 70ms
   - But batched execution reduces this
   - Need to measure actual GPU execution time

2. **Dispatch overhead**: 620 dispatches × ~10µs = 6.2ms
   - llama.cpp may have lower dispatch overhead
   - Residency Sets help reduce this

3. **Memory bandwidth**:
   - We may have extra memory copies
   - Cache locality may be suboptimal

### Next Steps - Session 6

1. **Measure actual GPU execution time** - Use Metal GPU timing
2. **Compare dispatch overhead** - Profile CPU-side dispatch time
3. **Investigate memory bandwidth** - Check for unnecessary copies
4. **Profile with Instruments** - Use Metal Performance HUD

---

## Conclusion - Session 5

**Fusion is NOT the main cause of the 3.0x performance gap.**

llama.cpp doesn't have more fusion for decode than we do:
- No GateUp fusion
- No SiLU+Mul fusion (just unary kernel)
- MUL_MAT+Add only for MMA (multi-row), not decode
- Our GEMV+Add fusion exists but causes 19% regression (disabled)

The gap must come from:
1. **Kernel implementation efficiency** - GEMV kernels may be slower
2. **Memory management** - No Residency Sets, partial shared buffers
3. **Dispatch overhead** - Per-dispatch cost may be higher
4. **Metal pipeline management** - Possible JIT compilation overhead
- Layers: 28
- n_heads: 12
- n_kv_heads: 2 (GQA ratio = 6)
- head_dim: 128

**Model**: Qwen3-0.6B-UD-Q4_K_XL
- Hidden size: 1024
- Intermediate size: 4096
- Layers: 28
- n_heads: 16
- n_kv_heads: 8 (GQA ratio = 2)
- head_dim: 128

---

## Critical Finding - Q4_K Kernel Thread Utilization (2026-10-22 Session 7)

**ROOT CAUSE IDENTIFIED**: Q4_K GEMV kernel has **7-8x lower thread utilization** than llama.cpp!

### Our Q4_K Kernel (kquant_mv.metal lines 72-107):
```metal
// Each thread processes blocks at stride 32
for (int ib = tiisg; ib < nb; ib += N_SIMDWIDTH) {  // stride = 32
    // ... process block
}
```

**For K=1024 (nb=4 blocks):**
- Only tiisg=0,1,2,3 process blocks
- **28 out of 32 threads are IDLE!**
- Thread utilization: 12.5%

### llama.cpp Q4_K Kernel (mul_mv.metal lines 1526-1606):
```metal
const short ix = tiisg/8;  // 0...3 (divide 32 threads into 4 groups)
for (int ib = ix; ib < nb; ib += 4) {  // stride = 4
    // ... process block with all 8 threads in group
}
```

**For K=1024 (nb=4 blocks):**
- ix=0 processes block 0, ix=1 processes block 1, etc.
- **All 32 threads are utilized!**
- Thread utilization: 100%

### Additional llama.cpp Optimizations Missing in Q4_K:
1. **Y-vector caching**: Pre-loads input into registers `yl[16]`, `yh[16]`
2. **Float4 accumulation**: Uses SIMD vector operations
3. **Scale extraction masks**: Uses `kmask1`, `kmask2`, `kmask3` constants

### Q5_K Kernel Status:
Our Q5_K kernel (lines 293-404) **IS correctly optimized**:
- Uses 4-block batching with `ix = tiisg%4`
- Y-vector caching present
- Float4 accumulation present

### Immediate Fix Required:
**Rewrite Q4_K kernel** to match llama.cpp's approach:
1. Change loop stride from 32 to 4
2. Add Y-vector caching
3. Add float4 accumulation
4. Verify correctness

### Expected Improvement:
- Thread utilization: 12.5% → 100% = **8x theoretical speedup**
- Realistic: 2-4x for Q4_K kernels
- Overall decode speedup: 15-30% (Q4_K is ~30% of operations)

---

## Critical Findings from llama.cpp-0.7.0 Analysis

### 1. SIMD Group Matrix Multiply - **已使用** ✅

**我们的实现**:
```metal
simdgroup_multiply_accumulate(mqk[nc], mq[0], mk[nc], mqk[nc]);
```

**结论**: 已使用Metal builtin，不是性能瓶颈

---

### 2. Q5_K GEMV Threadgroup配置 - **发现关键差异！** ⚠️

**我们的配置**:
```rust
thread_group_size = 32  // 1个simdgroup
thread_group_count = m   // M个threadgroups
```

**llama.cpp配置**:
```c
N_SG_Q5_K = 2  // 2个simdgroups per threadgroup
// 启动 M/2 个threadgroups
```

**问题**:
- 我们每个threadgroup只有1个simdgroup
- llama.cpp每个threadgroup有2个simdgroups
- GPU利用率低，可能浪费资源

**优化方案**:
- [ ] 修改dispatch配置，使用2个simdgroups per threadgroup
- [ ] 调整kernel使其支持多simdgroup
- [ ] 性能对比测试

**预期收益**: 中-高（更好的GPU利用率）

---

### 3. Q5_K GEMV N_R0配置 - **已优化** ✅

**我们的实现**:
```metal
// 使用Metal builtin函数
simdgroup_half8x8  mk, mq;
simdgroup_float8x8 mqk;

simdgroup_load(mk, pk, stride, 0, true);  // transpose=true
simdgroup_load(mq, pq, stride);
simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
```

**我们的实现**:
- 可能在手动实现矩阵乘法
- 效率可能低2-4倍

**优化方案**:
- [ ] 检查Flash Attention是否使用`simdgroup_multiply_accumulate`
- [ ] 如果使用手动实现，改为Metal builtin
- [ ] 验证性能提升

**预期收益**: 2-4x Flash Attention性能提升

---

### 2. Q5_K GEMV Kernel配置 - **寄存器压力优化** ⚠️

**llama.cpp配置**:
```c
// Q5_K专用配置
#define N_R0_Q5_K 1   // 1行/simdgroup
#define N_SG_Q5_K 2   // 2个simdgroup/threadgroup
```

**原因**:
- Q5_K需要复杂解包（高5位+低位）
- N_R0=1减少寄存器压力，避免spill
- 平衡计算强度和内存带宽

**我们的实现**:
- 可能使用N_R0=4
- 导致寄存器spill，性能下降

**优化方案**:
- [ ] 检查我们Q5_K GEMV的N_R0配置
- [ ] 如果不是1，改为1
- [ ] 对比性能

**预期收益**: 中等提升（寄存器优化）

---

### 3. Function Constants - **动态配置优化** ⚠️

**llama.cpp实现**:
```metal
// 编译时生成多个pipeline variant
constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];

// 运行时选择最优NSG
switch (FC_flash_attn_ext_nsg) {
    case 4: kernel_flash_attn_ext_impl<FWD_TMPL, 4>(...); break;
    case 8: kernel_flash_attn_ext_impl<FWD_TMPL, 8>(...); break;
}
```

**优势**:
- 单个kernel源码生成多个优化variant
- 运行时选择最优配置
- 适应不同问题规模

**我们的实现**:
- 固定配置
- 可能不是最优

**优化方案**:
- [ ] 为Flash Attention添加Function Constants
- [ ] 支持NSG=4和NSG=8动态选择
- [ ] 根据seq_len选择最优配置

**预期收益**: 小-中等提升

---

### 4. Command Buffer并行编码 - **减少启动延迟** ⚠️

**llama.cpp实现**:
```objc
// 主线程立即编码前64-10%节点
const int n_main = MAX(64, 0.1*gf->n_nodes);

ctx->n_nodes_0 = MIN(n_main, gf->n_nodes);
ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

// 并行编码其他command buffers
dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);
```

**关键优化**:
- 双缓冲策略
- 多线程并行编码
- 使用`commandBufferWithUnretainedReferences`

**我们的实现**:
- 单线程串行编码
- 可能存在启动延迟

**优化方案**:
- [ ] 实现双缓冲编码
- [ ] 主线程立即编码，后台线程准备后续
- [ ] 对比启动延迟

**预期收益**: 减少启动延迟，对短序列影响大

---

### 5. Fusion优化 - **减少中间内存访问** ⚠️

**llama.cpp Fusion支持**:
```c
GGML_METAL_FUSION_NORM_MUL,     // NORM + MUL
GGML_METAL_FUSION_NORM_MUL_ADD, // NORM + MUL + ADD
GGML_METAL_FUSION_ADD_CHAIN,    // ADD x N
GGML_METAL_FUSION_MUL_MAT_ADD,  // MUL_MAT + ADD
```

**我们的实现**:
- FusedQKV: 实际是3个独立GEMV ❌
- FusedNormLinear: 未确认是否激活 ❓
- 缺少MUL_MAT+ADD融合 ❌

**优化方案**:
- [ ] 验证FusedNormLinear是否工作
- [ ] 实现真正的FusedQKV (合并GEMV为GEMM)
- [ ] 添加MUL_MAT+ADD融合（用于residual connection）

**预期收益**: 减少内存访问，提升5-15%

---

### 6. Residency Sets - **避免Page Fault** ⚠️

**llama.cpp实现**:
```objc
// macOS 15+，保持内存常驻
void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev) {
    // 确保buffer不被swap out
}
```

**影响**:
- 长序列推理时影响显著
- 避免GPU访问CPU内存的page fault

**我们的实现**:
- 未启用

**优化方案**:
- [ ] 检查是否支持Residency Sets (macOS 15+)
- [ ] 为KV cache和weights启用常驻
- [ ] 测试长序列性能

**预期收益**: 长序列性能提升

---

### 7. 循环展开 - **消除循环开销** ⚠️

**llama.cpp实现**:
```metal
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

FOR_UNROLL (short l = 0; l < 8; ++l) {
    // 编译器完全展开
    acc1[0] += yl[l+0] * (q1[l] & 0x0F);
}
```

**我们的实现**:
- 可能未使用`#pragma unroll`

**优化方案**:
- [ ] 检查关键循环是否展开
- [ ] 添加FOR_UNROLL宏
- [ ] 验证汇编代码

**预期收益**: 小提升（编译器优化）

---

## Known Bottlenecks in Our Implementation

### 1. Flash Attention (✅ FIXED)
- **Problem**: Metal编译错误
- **Root cause**: `float4`数组初始化
- **Fix**: 改用标量数组
- **Result**: 785µs/layer (8.3% of decode)

### 2. Small M GEMV Performance (⚠️ IDENTIFIED, LOW IMPACT)
- K projection (M=256): 1.72 GOPS
- Q projection (M=1536): 8.68 GOPS
- **Impact**: 只占decode时间1.7%
- **Priority**: 低

### 3. FusedQKV (❌ NOT FUSED)
- 当前：3个独立GEMV dispatch
- 代码：`src/dag/executor.rs:689-717`
- **Priority**: 中

---

## Optimization Priority Ranking

### P0: GEMV Threadgroup配置（新发现）✅
**预计提升**: 中-高
**难度**: 中等
**状态**: ✅ 已完成并测试 (re-enabled after NaN fix)

**问题**: 每个threadgroup只有1个simdgroup，GPU利用率低

**行动项**:
1. ✅ 修改GEMV dispatch配置为2个simdgroups per threadgroup
2. ✅ 调整kernel使其支持多simdgroup（Q4_K, Q5_K, Q2_K, Q3_K）
3. ✅ Q6_K特殊处理：N_R0=2，每个simdgroup处理2行
4. ✅ 正确性测试通过
5. ✅ 性能对比测试完成
6. ✅ Re-enabled after confirming NaN was from MMA kernels (not P0)

**代码位置**: `src/quant/vec_dot/gpu/metal.rs:3894-3908`

**性能结果** (10次运行):
- Baseline: 76.34 t/s
- Optimized: 98.38 t/s (avg)
- **提升: 28.9%** (+22.04 t/s)
- 范围: 91.74 - 107.53 t/s
- After NaN fix: ~84-89 t/s (MMA disabled)

---

### P1: FusedQKV真正融合 ✅
**预计提升**: 中等
**难度**: 中等
**状态**: ✅ 已完成

**问题**: 当前FusedQKV实际是3个独立GEMV dispatch，未真正融合

**当前实现**: `src/dag/executor.rs:690-718`
```rust
// Run three GEMV operations
// TODO: Optimize to single GEMM when weights are contiguous
ctx.gpu_ctx.mv_encode(encoder, &q_weight_buf, &input_buf, &q_buf, *q_out_size, hidden_size, q_dtype);
ctx.gpu_ctx.mv_encode(encoder, &k_weight_buf, &input_buf, &k_buf, *k_out_size, hidden_size, k_dtype);
ctx.gpu_ctx.mv_encode(encoder, &v_weight_buf, &input_buf, &v_buf, *v_out_size, hidden_size, v_dtype);
```

**发现**: GPU路径未使用fused QKV
- **CPU路径**: 已支持`attn_qkv` (line 339-357)
- **GPU路径**: 仅支持分开的`wq/wk/wv`，未使用`attn_qkv`
- **测试模型**: Qwen3-0.6B使用分开q/k/v，Qwen3.8有fused `attn_qkv.weight`

**完成步骤**:
1. ✅ 检查Q/K/V weights内存布局（发现部分模型有fused qkv）
2. ✅ 添加`FusedQKVWeight` Operation类型 (`src/dag/graph.rs`)
3. ✅ 实现split_qkv Metal kernel (`src/ops/metal/ops/split_qkv.metal`)
4. ✅ 添加split_qkv_encode方法到MetalContext
5. ✅ 更新executor使用split kernel
6. ✅ 测试性能（95.24 t/s，无性能退化）

**结果**: 框架就绪，支持fused QKV weights模型（如Qwen3.8）
- 单GEMV + split kernel（代替3个GEMV）
- 需要fused qkv weight的模型才能受益

---

### P5: Fix FusedNormLinear Q5_K (Partial) ⚠️
**预计提升**: 5-15% (if enabled)
**难度**: 中等
**状态**: ⚠️ Partial fix applied, still disabled

**问题**: FusedNormLinear Q5_K kernel produces NaN

**调查发现**:
1. Q5_K scales need -32 offset (fixed in commit 2879076)
2. Scale extraction method may still differ from working GEMV
3. Working GEMV uses complex mask-based extraction
4. FusedNormLinear uses simpler `get_scale_min_k4` helper

**修复**:
- ✅ Added -32 offset for scales (commit 2879076)
- ⚠️ Scale extraction may need further work

**代码位置**: `src/ops/metal/ops/fused_norm_linear.metal:497-523`

**下一步**:
- [ ] Test if partial fix resolves NaN
- [ ] If still NaN, rewrite scale extraction to match GEMV

---

### P2: Function Constants ⏭️ SKIP
**预计提升**: 小-中等
**难度**: 中等
**状态**: ⏭️ 跳过（收益不足）

**背景**: llama.cpp使用Function Constants在编译时生成多个kernel variant，运行时根据问题规模选择最优配置

**llama.cpp实现**:
```metal
constant short FC_flash_attn_ext_nsg [[function_constant(FC_FLASH_ATTN_EXT + 1)]];

switch (FC_flash_attn_ext_nsg) {
    case 4: kernel_flash_attn_ext_impl<FWD_TMPL, 4>(...); break;
    case 8: kernel_flash_attn_ext_impl<FWD_TMPL, 8>(...); break;
}
```

**当前状态**:
- Flash Attention占decode时间**8.3%**
- 使用MMA v2 kernel（已优化）
- Kernel使用simdgroup操作，性能良好

**跳过原因**:
1. **收益有限**: 即使优化50%，总提升仅4%（0.4ms）
2. **实现复杂**: 需要修改Metal kernel + 多pipeline variant
3. **性价比低**: 2.3x差距需要更高impact的优化
4. **Kernel已优化**: 当前实现使用simdgroup，访问模式良好

**决策**: 跳过P2，继续P3（Command Buffer并行编码）

---

### P3: Command Buffer并行编码 ⏭️ SKIP
**预计提升**: 减少启动延迟
**难度**: 中等
**状态**: ⏭️ 已优化（部分）

**背景**: llama.cpp使用多线程并行编码command buffer，减少启动延迟

**llama.cpp实现**:
```objc
// 主线程立即编码前64-10%节点
const int n_main = MAX(64, 0.1*gf->n_nodes);

ctx->n_nodes_0 = MIN(n_main, gf->n_nodes);
ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

// 并行编码其他command buffers
dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);
```

**当前实现** (`src/dag/executor.rs:158-161`):
```rust
let command_buffer = queue.new_command_buffer_with_unretained_references();
command_buffer.enqueue(); // Pre-enqueue (llama.cpp line 565)
let encoder = command_buffer.new_compute_command_encoder();
```

**已实现的优化**:
1. ✅ 使用`command_buffer_with_unretained_references`
2. ✅ 提前enqueue command buffer
3. ✅ 单command buffer处理整个DAG

**未实现**:
- ❌ 多线程并行编码

**跳过原因**:
1. **Decode场景**: 单token生成，DAG不大（198 ops融合后更少）
2. **编码时间短**: 单线程编码可能已经足够快
3. **收益有限**: 并行编码主要用于batch prefill
4. **实现复杂**: 需要重构DAG executor支持多线程

**决策**: 当前实现已包含主要优化，跳过多线程并行编码，继续P4

---

### P4: Residency Sets ⏭️ SKIP
**预计提升**: 长序列性能
**难度**: 简单（假设有API支持）
**状态**: ⏭️ 阻塞（metal crate不支持）

**背景**: macOS 15+提供Residency Sets API，保持GPU内存常驻，避免page fault

**API使用方式** (从Apple WWDC24示例):
```objc
// Create residency set
MTL::ResidencySet* residencySet = device->newResidencySet(residencySetDescriptor, &error);

// Add to command queue
commandQueue->addResidencySet(residencySet);

// Add allocations and commit
residencySet->addAllocation(buffer);
residencySet->commit();
```

**当前metal crate状态**:
- 版本: metal = "0.31"
- **不支持ResidencySet API** ❌
- 检查docs.rs和源码：无相关API
- 需要objc2-metal crate或等待metal crate更新

**阻塞原因**:
1. **API缺失**: metal crate (0.31) 没有ResidencySet支持
2. **依赖复杂**: 添加objc2-metal需要重构Metal接口
3. **收益有限**: 主要针对长序列，当前decode benchmark可能不是长序列
4. **当前gap**: 2.3x性能差距主要由其他因素造成，不是内存residency

**影响评估**:
- ✅ macOS版本支持: macOS 27 (满足macOS 15+要求)
- ❌ Rust crate支持: metal 0.31无此API
- ⚠️ 性能影响: 长序列场景下防止GPU内存swap out
- ⚠️ 优先级: 非当前瓶颈（短序列decode性能差距2.3x）

**决策**: 跳过P4，原因：
1. 实现需要添加新依赖（objc2-metal）或等待metal crate更新
2. 当前性能gap主要由计算效率造成，不是内存residency
3. 长序列优化应在达到短序列性能目标后进行

**未来行动**:
- [ ] 监控metal crate是否添加ResidencySet支持
- [ ] 或评估是否迁移到objc2-metal
- [ ] 在达到基础性能目标后，针对长序列场景优化

---

## Performance Gap Analysis

### Current Status After P0-P4

**Performance**:
- Baseline: 76.34 t/s
- After P0 (NSG=2): 98.38 t/s (avg, +28.9%)
- After P1 test: 95.24 t/s (split kernel framework)
- Target: 227 t/s (llama.cpp)
- **Remaining gap: 2.3x**

### Where is the Gap Coming From?

#### P0-P3 Optimizations Status
- ✅ **P0**: GEMV threadgroup配置 (NSG=1→2) - 已优化
- ✅ **P1**: FusedQKV split kernel - 框架就绪
- ⏭️ **P2**: Function Constants - 跳过（收益<4%）
- ⏭️ **P3**: Command Buffer并行编码 - 已有主要优化
- ⏭️ **P4**: Residency Sets - 阻塞（metal crate不支持）

#### Unexplored Optimizations

From Section 5 (Fusion):
- FusedNormLinear: 已实现，需要验证是否激活 ⚠️
- MUL_MAT+ADD融合: 未实现 ❌

From Section 7 (循环展开):
- 未使用`#pragma unroll` ⚠️

#### Known Bottlenecks
1. **Flash Attention**: ✅ 已修复，占8.3% decode时间
2. **Small M GEMV**: ⚠️ 已识别，低影响（1.7% decode时间）
3. **Dispatch overhead**: ✅ 已优化（DAG fusion减少dispatch）

### Next Optimization Opportunities

#### Priority 1: Implement GEMV+Add Fusion (NEW FINDING) ⚠️
**目标**: Fuse GEMV output with residual addition into single kernel
**预期收益**: Reduce 56 dispatches (2 per layer × 28 layers)
**状态**: ⚠️ Fusion code exists but DISABLED

**发现**:
- `FusedGemvAdd` operation exists in code
- **BUT line 30 in optimizer.rs is COMMENTED OUT**: `// fused_count += Self::fuse_gemv_add(dag);`
- Even if enabled, no Metal kernel - still does 2 separate dispatches
- llama.cpp has this as `GGML_METAL_FUSION_MUL_MAT_ADD`

**当前实现** (src/dag/executor.rs:1438-1459):
```rust
// Step 1: Execute GEMV to output buffer
ctx.model.gemv_encode(encoder, &input_buf, ...);
// Step 2: Add residual to output buffer in-place
ctx.gpu_ctx.add_encode(encoder, &residual_buf, ...);
```

**行动**:
- [ ] Create Metal kernel `kernel_fused_gemv_add` for K-quant weights
- [ ] Enable fusion in optimizer.rs (uncomment line 30)
- [ ] Add fused_gemv_add_encode method to MetalContext
- [ ] Test correctness
- [ ] Measure performance improvement

**预期提升**: 中等 (减少56次dispatch，每次~10-20µs)

---

#### Priority 2: Multi-Head Flash Attention for High Head Count Models
**目标**: Use existing multi-head flash attention for models with n_heads >= 32
**预期收益**: 2-3x Flash Attention improvement (reduces dispatch overhead 32x)
**状态**: ✅ Kernel已实现并可用 (`INFER_USE_MULTIHEAD=1`)

**分析**:
- **Our approach**: Q=1, NSG=1, Grid=[1, n_heads, n_seqs] (single-head)
- **llama.cpp approach**: Q=32, NSG=8, Grid=[n_heads/32, n_seqs, 1] (multi-head)
- **差距**: 32x dispatch overhead, 8x thread utilization gap

**测试结果** (DeepSeek 1.5B, n_heads=12):
- Single-head: 84.75 t/s
- Multi-head: 81.97 t/s (slightly slower!)
- 原因: Multi-head设计用于n_heads>=32，小head数时single-head更高效

**行动**:
- [ ] Test with model that has n_heads >= 32 (e.g., Qwen3.8-27B)
- [ ] Measure performance improvement for high head count models
- [ ] Document which models benefit from multi-head kernel

#### Priority 3: Debug MMA Flash Attention
**目标**: Fix MMA kernel bugs to enable higher Flash Attention performance
**预期收益**: Potential improvement over standard f16 kernel
**状态**: ⚠️ Partially fixed, still performance issues

**发现**:
1. ✅ GQA ratio问题已修复
2. ✅ Batched kernel越界问题已修复 (commit c2fc611)
   - 原因: 32线程处理128元素，每个线程应处理4元素，不是16元素
   - 访问模式: `k_pos[tiisg]` 而不是 `k_pos[tiisg * 4 + i]`
3. ❌ 性能仍然极差 (0.35 t/s vs 84.75 t/s)
   - 原因: 非batched kernel逐位置处理，每次只加载一个K/V
   - Batched kernel未被使用（需要修改dispatch代码）

**行动**:
- [ ] 修改executor使用batched MMA kernel
- [ ] 或者优化非batched kernel为批量处理
- [ ] 测试batched kernel性能

#### Priority 3: Optimize FusedNormLinear Kernel
**目标**: Fix FusedNormLinear to not cause performance regression
**预期收益**: Enable fusion without slowdown
**发现**: FusedNormLinear causes 10% performance regression due to threadgroup memory usage
**行动**:
- [ ] Analyze threadgroup memory access patterns
- [ ] Consider reducing shared memory usage
- [ ] Benchmark vs separate RMSNorm + GEMV

## Critical Issue: NaN Outputs - RESOLVED ✅

### Investigation Results (2026-10-22)

**Problem**: All decode outputs were NaN, causing repetitive token generation

**Root Cause Identified**: MMA Flash Attention kernels (v1 and v2) had hardcoded `gqa_ratio = 4` but actual GQA ratio varies by model (e.g., Qwen3-0.6B has gqa_ratio = 2). This caused incorrect KV cache lookups.

**Investigation Process**:
1. ✅ Traced DAG execution - found tensor dependency issues
2. ✅ Verified embedding lookup produces valid outputs
3. ✅ Identified GEMV inputs were from Flash Attention output
4. ✅ Discovered MMA kernels have hardcoded GQA ratio
5. ✅ Tested with MMA disabled - NaN issue resolved

**Fixes Applied** (Commit: 4c52d77):
1. Added `gqa_ratio` parameter to MMA v1 and v2 kernels
2. Updated Rust dispatch code to pass correct GQA ratio
3. Disabled MMA kernels by default (additional bugs remain)
4. Standard Flash Attention kernel (`flash_attn_f16_encode`) used by default

**Result**:
- ✅ NaN issue resolved
- ✅ Both Q4_K and Q5_K models work correctly
- Performance: 81.97 t/s (Qwen3-0.6B Q4_K)

**Remaining MMA Issues**:
- MMA kernels still produce NaN even with correct GQA ratio
- Additional kernel bugs need investigation
- Recommend keeping MMA disabled until fully fixed

**Future Work**:
- [ ] Debug MMA kernel online softmax logic
- [ ] Compare MMA kernel with working standard kernel step-by-step
- [ ] Verify SIMD reduction patterns

---

## Completed Optimizations

### ✅ SIMD Group Matrix Multiply
- 状态：已使用Metal builtin
- 验证：Flash Attention kernel已使用`simdgroup_multiply_accumulate`

### ✅ Q5_K N_R0配置
- 状态：已使用N_R0=1
- 验证：kernel代码每个threadgroup处理1行

---

## Performance Tracking

### Baseline (Before P0)
- Decode: 76.34 t/s (13.10 ms/token)
- Flash Attention: 785µs/layer (8.3%)
- K projection: 228µs, 1.72 GOPS

### After P0 Optimization (NSG=1 → NSG=2)
- Decode: **98.38 t/s** (average, 10 runs)
  - Min: 91.74 t/s
  - Max: 107.53 t/s
  - Stddev: 4.24 t/s
- **Improvement: 28.9%** (+22.04 t/s)

### After P0 Re-enabled (MMA disabled)
- Decode: **~89 t/s** (Qwen3-0.6B Q4_K)
- MMA Flash Attention disabled due to kernel bugs
- Standard f16 Flash Attention used

### Multi-Head Flash Attention Testing
- Single-head (n_heads=12): 84.75 t/s
- Multi-head (n_heads=12): 81.97 t/s (3% slower)
- Conclusion: Multi-head optimal for n_heads >= 32

### Target
- Decode: 227 t/s (llama.cpp)
- Remaining gap: **2.3x** (down from 3x)

---

## Detailed llama.cpp Implementation Notes

### File Structure
```
llama.cpp-0.7.0/ggml/src/ggml-metal/
├── kernels/
│   ├── mul_mv.metal          # GEMV kernel实现
│   ├── mul_mv_mma.metal      # MMA (Matrix Multiply Accumulate)
│   ├── fa_common.metal       # Flash Attention核心
│   ├── fa_q5_0.metal         # Q5_0 Flash Attention
│   ├── fa_vec_q5_0.metal     # 向量化FA
│   ├── dequantize.h          # 反量化函数
│   ├── quantize.h            # 量化函数
│   └── common.h              # 通用定义
├── ggml-metal-device.m       # 设备管理、dispatch
├── ggml-metal-context.m      # 图计算调度
├── ggml-metal-impl.h         # 关键配置定义
├── ggml-metal-fusion.h       # 融合优化
├── ggml-metal-ops.cpp        # Op实现
└── ggml-metal-common.h       # 辅助函数
```

### Key Code Patterns

#### GEMV Dispatch Pattern
```metal
// 每个threadgroup处理N_R0行
// 每个simdgroup处理N_R0行
// 总并行度 = threadgroups × NSG × N_R0

const int first_row = (r0 * NSG + sgitg) * nr0;
```

#### Flash Attention Multi-head Pattern
```metal
// Q=8 queries per threadgroup
// C=64 cache items per iteration
// NSG=4 or 8 (dynamic)

const ushort iq3 = tgpig[2];  // head维度
const ushort iq2 = tgpig[1];  // KV head组
const ushort iq1 = tgpig[0]*Q; // query批次
```

#### Dequantization Pattern
```metal
// 支持16种反量化模式（il=0..15）
// 使用get_scale_min_k4_just2高效提取scale和min
// 合并计算减少指令数

const float dl = d * sc[0];
const float ml = min * sc[1];
reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
```

---

## Next Steps

1. [ ] **验证SIMD Matrix Multiply** - 检查Flash Attention实现
2. [ ] **检查Q5_K配置** - 确认N_R0是否为1
3. [ ] **实现Function Constants** - 动态NSG选择
4. [ ] **实现FusedQKV** - 真正的融合kernel
5. [ ] **启用Residency Sets** - 长序列优化
6. [ ] **性能对比测试** - 验证每个优化效果

---

## Progress Log

### 2026-10-22 Session 3 - MMA Kernel Debugging ⚠️

**MMA Kernel Investigation** ✅/⚠️
- Commit: "Fix MMA batched kernel: correct thread access pattern (32 threads / 128 elements = 4 per thread)"
- Found root cause of NaN: thread access pattern wrong (32 threads × 16 elements ≠ 128)
- Fixed: Each thread processes 4 elements, not 16
- Batched kernel: No NaN now, but not used by default
- Non-batched kernel: Still used, extremely slow (0.35 t/s)

**Testing Results**:
- MMA kernel (non-batched): 0.35 t/s (2400x slower!)
- Reason: Processes one KV position per iteration
- Need to switch to batched kernel for better performance

**Current Status Summary**:
- ✅ P0 GEMV optimization: Active and working
- ✅ Multi-head Flash Attention: Available (optimal for n_heads >= 32)
- ⚠️ MMA Flash Attention: Fixed NaN but performance terrible
- ❌ FusedGateUp: Disabled (38% performance regression)
- ❌ FusedNormLinear: Disabled (10% performance regression)
- ✅ FusedAddNorm: Enabled

**Key Insights**:
1. MMA kernel had fundamental access pattern bug (threads × elements mismatch)
2. Batched kernel fixed but not integrated into executor
3. Need to switch from non-batched to batched MMA kernel
4. Standard f16 kernel is still the best option for now

---

### 2026-10-22 Session 4 - MMA Batched Kernel Performance Investigation ⚠️

**MMA Batched Kernel Switch** ✅
- Commit: d6b8b57 "Switch to batched MMA kernel for Flash Attention (still slow, needs optimization)"
- Updated executor to use `flash_attn_mma_batched_encode` instead of non-batched
- Kernel runs without NaN (correctness verified)

**Performance Issue** ⚠️
- MMA batched kernel: **0.34 t/s** (2400x slower than standard f16 kernel!)
- Standard f16 kernel: **84 t/s**
- Root cause: Unknown, needs deep investigation
- Likely issues:
  1. Register pressure from batching 8 positions
  2. Loop efficiency issues
  3. SIMD group utilization problems
  4. Memory access pattern inefficiency

**Decision**:
- MMA kernel disabled by default (INFER_USE_MMA environment variable)
- Standard f16 Flash Attention remains the best option
- MMA optimization is low priority given the 2400x gap

**Current Status Summary**:
- ✅ P0 GEMV optimization: Active and working
- ✅ Standard f16 Flash Attention: Best performance (84-87 t/s)
- ⚠️ MMA Flash Attention: NaN-free but 2400x slower (disabled)
- ❌ Multi-head Flash Attention: Available but slower for n_heads=12
- ❌ FusedGateUp: Disabled (38% performance regression)
- ❌ FusedNormLinear: Disabled (10% performance regression)
- ✅ FusedAddNorm: Enabled

**Key Insights**:
1. MMA kernel architecture has fundamental performance issues beyond NaN
2. Batched kernel is slower than expected - register pressure suspected
3. Standard f16 kernel remains the most efficient option
4. Performance gap (2.3x) must come from other optimizations

---

### 2026-10-22 Session 4 - GEMV+Add Fusion Implementation ❌

**Discovery: FusedGemvAdd was DISABLED** ⚠️
- Found `fuse_gemv_add(dag)` commented out in optimizer.rs line 30
- 56 extra dispatches: 2 residual connections per layer × 28 layers
- llama.cpp has this as `GGML_METAL_FUSION_MUL_MAT_ADD`

**Implementation Complete** ✅
- Commit: 7f60658 "Implement GEMV+Add fusion for residual connections"
- Created Metal kernels: `kernel_fused_gemv_add_q4_k`, `q5_k`, `q6_k`
- Added `fused_gemv_add_encode` method to MetalContext
- Enabled fusion in optimizer.rs
- Updated executor to use fused kernel with fallback for unsupported formats

**Performance Testing Results** ❌
- With fusion: **68 t/s**
- Without fusion: **79 t/s**
- **Performance regression: 19%**

**Root Cause Analysis**:
- Fused kernel is slower than separate GEMV + Add operations
- Likely issues:
  1. Increased register pressure from loading residual in GEMV kernel
  2. Memory access pattern less efficient (extra residual buffer read)
  3. Separate kernels may benefit from better cache locality
  4. Threadgroup memory usage overhead

**Decision**:
- Commit: 9ac82d3 "Disable GEMV+Add fusion: causes 19% performance regression"
- Keep fusion code for future optimization
- Metal kernels remain in codebase but not used in production
- Similar to FusedNormLinear (10% regression) and FusedGateUp (38% regression)

**Lesson**: Fusion is not always faster - need to measure, not assume

---

### 2026-10-22 Session 3 - Multi-Head Flash Attention Analysis ✅

**Multi-Head Flash Attention Testing** ✅
- Commit: "Document multi-head flash attention efficiency for different head counts"
- Discovered multi-head kernel already exists (Q=32, NSG=8)
- Tested with DeepSeek 1.5B (n_heads=12): 81.97 t/s (slower than single-head)
- Finding: Multi-head kernel designed for n_heads >= 32
- For small head counts, single-head approach is more efficient

**Performance Testing** ✅
- 5-run average: 85.6 t/s (range: 84.75-86.96 t/s)
- Model: DeepSeek-R1-Distill-Qwen-1.5B Q5_K (n_heads=12)
- Consistent performance, no variance issues

**Current Status Summary**:
- ✅ P0 GEMV optimization: Active and working
- ✅ Multi-head Flash Attention: Available (INFER_USE_MULTIHEAD=1), optimal for n_heads >= 32
- ❌ MMA Flash Attention: Disabled (kernel bugs)
- ❌ FusedGateUp: Disabled (38% performance regression)
- ❌ FusedNormLinear: Disabled (10% performance regression)
- ✅ FusedAddNorm: Enabled

**Key Insights**:
1. Multi-head flash attention exists and works, but is slower for small head counts
2. Test model (n_heads=12) benefits from single-head approach
3. Models with n_heads >= 32 should test multi-head kernel
4. Fusion kernels need careful optimization to not regress

---

### 2026-10-22 Session 3 - FusedNormLinear Analysis ✅

**FusedNormLinear Q5_K Fix and Testing** ✅
- Commit: "Fix FusedNormLinear Q5_K: add -32 offset for scales (partial fix)"
- Commit: "Keep FusedNormLinear disabled: causes 10% performance regression"
- Fixed Q5_K scale offset (added -32 for unsigned to signed conversion)
- Tested: No NaN, but 10% performance regression (76 t/s vs 84 t/s)
- Root cause: Threadgroup memory usage for normalized input
- Decision: Keep disabled, separate RMSNorm + GEMV is more efficient

**Performance Testing** ✅
- 5-run average: 85.6 t/s (range: 84.75-86.96 t/s)
- Model: DeepSeek-R1-Distill-Qwen-1.5B Q5_K
- Consistent performance, no variance issues

**Flash Attention Bottleneck Identified** ⚠️
- **Critical finding**: Single-head vs multi-head architecture
- Our kernel: Q=1, NSG=1 (single query per threadgroup)
- llama.cpp: Q=32, NSG=8 (32 queries per threadgroup)
- **Impact**: 32x dispatch overhead, 8x thread utilization gap
- **Solution**: Implement multi-head flash attention (significant work)

**Current Status Summary**:
- ✅ P0 GEMV optimization: Active and working
- ❌ MMA Flash Attention: Disabled (kernel bugs)
- ❌ FusedGateUp: Disabled (38% performance regression)
- ❌ FusedNormLinear: Disabled (10% performance regression)
- ✅ FusedAddNorm: Enabled

**Key Insights**:
1. Fusion kernels are not always faster - need careful optimization
2. Memory access patterns and threadgroup configuration matter more than fusion count
3. Flash Attention architecture mismatch is major bottleneck
4. Separate optimized kernels can beat naive fusion

---

### 2026-10-22 Session 3 - P0 Re-enabled & FusedNormLinear Partial Fix ✅

**P0 GEMV Optimization Re-enabled** ✅
- Commit: "Re-enable P0 GEMV optimization (NSG=2) - NaN was from MMA kernels"
- Root cause analysis confirmed: NaN was from MMA Flash Attention kernels, not P0 GEMV
- P0 changes were temporarily reverted during NaN investigation
- Re-enabled all K-quant kernels (Q4_K, Q5_K, Q2_K, Q3_K, Q6_K) with sgitg parameter
- Performance: ~84-89 t/s (Qwen3-0.6B Q4_K / DeepSeek 1.5B Q5_K, MMA disabled)
- Gap: 2.55x (down from 2.7x with MMA disabled baseline)

**FusedNormLinear Q5_K Partial Fix** ⚠️
- Commit: "Fix FusedNormLinear Q5_K: add -32 offset for scales (partial fix)"
- Found that Q5_K scales need -32 offset to convert unsigned 6-bit to signed
- Working GEMV uses `(sc-32)` but FusedNormLinear was using raw scales
- Partial fix applied: added -32 offset for sc and m
- Still disabled: scale extraction method may differ from working GEMV

**Status**:
- ✅ P0 GEMV optimization active
- ❌ MMA Flash Attention disabled (kernel bugs remain)
- ❌ FusedGateUp disabled (NaN issue)
- ❌ FusedNormLinear disabled (partial fix, needs testing)
- ✅ FusedAddNorm enabled
- Standard Flash Attention (f16) used instead

**Key Findings**:
1. P0 and MMA are independent - P0 works correctly
2. MMA kernel bugs: GQA ratio + additional issues (online softmax?)
3. Multiple fusion optimizations disabled due to NaN issues
4. Remaining gap (2.5x) likely from missing fusions and non-MMA Flash Attention

---

### 2026-10-22 Session 3 - NaN Issue Resolved ✅

**NaN Investigation and Fix** ✅
- Commit: "Fix NaN issue: disable MMA Flash Attention kernels and add GQA ratio parameter"
- Root cause: MMA kernels hardcoded `gqa_ratio = 4` but models have different ratios
- Qwen3-0.6B: gqa_ratio = 2 (n_heads=16, n_kv_heads=8)
- DeepSeek-R1-Distill-Qwen-1.5B: gqa_ratio = 6 (n_heads=12, n_kv_heads=2)
- Fix: Added gqa_ratio parameter, disabled MMA by default
- Performance: 81.97 t/s (Qwen3-0.6B Q4_K)

**Key Findings**:
1. MMA v1 and v2 kernels both produce NaN for GQA models
2. GQA ratio fix alone doesn't resolve all MMA issues
3. Additional MMA kernel bugs remain (online softmax?)
4. Standard Flash Attention kernel works correctly

**Action Items for Future**:
- [ ] Debug MMA kernel step-by-step vs standard kernel
- [ ] Re-enable MMA after fixing all bugs
- [ ] Add MMA kernel unit tests

---

### 2026-10-22 Session 2

**P0 Optimization Complete** ✅
- Commit: "Optimize K-quant GEMV threadgroup config to match llama.cpp"
- Commit: "Add P0 optimization performance results"
- Performance: 76.34 t/s → 98.38 t/s (+28.9%)
- Gap reduction: 3x → 2.3x

**P1 Optimization Complete** ✅
- Commit: "Add FusedQKVWeight operation for merged Q/K/V weights"
- Commit: "Complete P1: Add split kernel for fused QKV projection"
- Added `FusedQKVWeight` Operation type
- Implemented split_qkv Metal kernel
- Framework ready for fused qkv weights (Qwen3.8-style models)
- Test: 95.24 t/s (no regression)

---

### 2026-10-22 Session 1

**Commit**: "Fix Flash Attention MMA kernel and add performance analysis tools"
- ✅ Fixed Metal compilation error in flash_decode_mma_batched.metal
- ✅ Added performance analysis tools
- ✅ Committed changes

**Current Work**: P0 Optimization - GEMV Threadgroup Configuration

**Step 1**: 检查当前kernel实现是否支持多simdgroup ✅
- **发现**: Kernel **不支持**sgitg参数
- 当前signature: `kernel_mul_mv_q5_K_f32(..., uint3 tgpig, ushort tiisg)`
- 缺少: `ushort sgitg` (simdgroup index)
- row计算: `row = tgpig.x` (每个threadgroup只处理1行)

**Step 2**: 修改kernel添加sgitg支持 ✅
- **修改**: Q5_K kernel已添加sgitg参数
- **代码**: `src/ops/metal/gemv/kquant_mv.metal:298-308`
- **变化**:
  ```metal
  // Before:
  ushort tiisg [[thread_index_in_simdgroup]])
  {
      const int row = tgpig.x;

  // After:
  ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]])
  {
      const int row = tgpig.x * 2 + sgitg;
  ```

**Step 3**: 修改dispatch配置 ✅
- 目标: 启动M/2个threadgroups，每个64 threads（2 simdgroups）
- 文件: `src/quant/vec_dot/gpu/metal.rs:3894-3908`
- **完成**:
  ```rust
  // Threadgroup configuration (matching llama.cpp)
  // Q4_K, Q5_K: NSG=2 (2 simdgroups per threadgroup, each processes 1 row)
  // Q6_K: NSG=1, N_R0=2 (1 simdgroup per threadgroup, processes 2 rows)
  let (nsg, nr0) = match dtype {
      crate::model::tensor::DataType::Q6_K => (1, 2),
      _ => (2, 1), // Q4_K, Q5_K, Q2_K, Q3_K
  };

  let threads_per_tg = nsg * 32; // 32 threads per simdgroup
  let tgs = (m + nsg * nr0 - 1) / (nsg * nr0); // threadgroups needed
  ```

**Step 4**: 修改所有K-quant kernels ✅
- **Q4_K**: 添加sgitg参数 (NSG=2, N_R0=1)
- **Q5_K**: 添加sgitg参数 (NSG=2, N_R0=1)
- **Q2_K**: 添加sgitg参数 (NSG=2, N_R0=1)
- **Q3_K**: 添加sgitg参数 (NSG=2, N_R0=1)
- **Q6_K**: 特殊处理，使用N_R0=2，每个simdgroup处理2行
  - 添加helper function `q6_k_row_dot()`避免代码重复
  - 单个simdgroup同时处理2行数据
  - 分两次计算simd_sum，分别输出两行结果

**Step 5**: 测试 ✅
- Q4_K GEMV正确性测试通过
- Decode性能测试：91.74 t/s (Qwen3-0.6B Q4_K)
- Kernel编译成功，无错误

**Commit**: "Optimize K-quant GEMV threadgroup config to match llama.cpp (P0)" ⏳
- 所有K-quant kernels已修改
- Dispatch配置已优化
- 等待性能测试结果

---

### 2026-10-22 Session 1
- ✅ 修复Flash Attention MMA kernel编译错误
- ✅ 识别小M GEMV性能问题
- ✅ 深入研究llama.cpp实现
- ✅ 发现SIMD Matrix Multiply可能是关键差距
- ✅ 创建优化roadmap

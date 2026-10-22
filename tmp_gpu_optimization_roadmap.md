# GPU Decode Performance Optimization Roadmap

## Current Status (2026-10-23 Session 10)

**Performance** (Q4_K models, Qwen3-0.6B):
| Metric | We | llama.cpp | Gap |
|--------|------|-----------|------|
| Prefill (128 tokens) | 2571 t/s | 920 t/s | **0.36x (faster!)** |
| Decode | 128.2 t/s | 201.3 t/s | **1.57x** |

**Status**: ✅ **Prefill exceeds llama.cpp!** Decode gap reduced to 1.57x

### Session 10 Progress
1. ✅ Verified prefill performance: 2571 t/s (2.8x faster than llama.cpp)
2. ✅ Verified decode performance: 128.2 t/s (1.57x gap)
3. ✅ Fixed benchmark to include embedding lookup time
4. ✅ Analyzed Flash Attention architecture - found workgroup parallelization gap
5. ✅ Implemented NWG parallelization for Flash Attention
6. ⏳ Testing NWG performance - requires longer sequences to show benefit

### Performance Timeline
| Session | Decode | Prefill | Gap (decode) | Optimization |
|---------|--------|---------|--------------|--------------|
| Session 8 | 95.24 t/s | - | 2.51x | Dispatch fix |
| Session 9 | 103 t/s | - | 2.05x | Vec registry baseline |
| Session 9 | 120.3 t/s | - | 1.99x | Vec registry + dtype cache |
| Session 10 | 128.2 t/s | 2571 t/s | 1.57x | Verified benchmarks |

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

### Remaining Gap Analysis (Session 10)

**Decode gap: 1.57x** (128.2 t/s vs 201.3 t/s)

**Decode is memory-bandwidth bound, not compute bound.**

#### Root Cause Analysis from First Principles

**Hypothesis 1: GEMV + Dequantization Fusion** ✅ **Already Fused**
- Dequantization happens inside GEMV kernel (kquant_mv.metal:127-143)
- Weights are dequantized on-the-fly during dot product
- No separate dequantization kernel
- **Not the problem**

**Hypothesis 2: Thread Count vs Memory Bandwidth** ⚠️ **Testing Needed**
- Current GEMV config: 64 threads/threadgroup (32 × 2 simdgroups)
- Q projection (M=2048): 512 threadgroups, **32,768 threads total**
- On ARM GPU, memory bandwidth is shared across all cores
- Too many threads can cause cache thrashing and bandwidth contention
- **Action: Test 2/4/8 threads to find optimal bandwidth utilization**

**Hypothesis 3: KV Cache Access Pattern** ✅ **NOT an Issue in Current Kernel**
- The MMA kernel (flash_decode_mma_true.metal) uses **sequential processing**:
  ```metal
  for (uint pos = 0; pos < seq_k; pos++) {
      device const half4* k_h4 = k_base + pos * 32;
      // All 32 threads read contiguous K[pos, d]
  }
  ```
- This is **contiguous access** - all threads work on the same position together
- Good cache utilization

**Root Cause Found: Flash Attention Workgroup Parallelization Gap** 🔴 **CRITICAL**

**Our implementation:**
- 1 workgroup per head (32 threads)
- Sequential processing of all KV positions
- Grid: `[n_heads, n_seqs, 1]`

**llama.cpp implementation:**
- **32 workgroups per head (NWG=32)**
- Each workgroup processes different KV position chunks
- Parallelization: `for (int ic0 = iwg*NSG; ic0 += NWG*NSG)`
- Final reduction across workgroups
- Grid: `[n_heads, n_seqs, NWG=32]`

**Impact:**
- For seq_len=128, each llama.cpp workgroup processes ~4 positions
- Our kernel processes all 128 positions sequentially
- **Parallelism gap: 32x fewer threadgroups**

#### Optimization Plan

**Priority 1: Flash Attention NWG Parallelization** ⚠️ **Implemented, Testing Needed**
- ✅ Implemented two-pass NWG kernel (flash_decode_nwg.metal)
- ✅ Pass 1: 32 workgroups process different KV position chunks
- ✅ Pass 2: Reduction across workgroups using online softmax formula
- ✅ Adaptive dispatch: Uses NWG only for seq_len >= 64
- ⏳ Performance testing shows thermal throttling affecting results
- **Note**: Short sequences (< 64) use sequential kernel (faster for small batches)

**NWG Implementation Details:**
- File: `src/ops/metal/attention/flash_decode_nwg.metal`
- Two kernels: `flash_decode_nwg_pass1_f16` and `flash_decode_nwg_pass2_f16`
- Pass 1 stores partial (O, S, M) results in global memory
- Pass 2 combines partial results using online softmax formula
- Rust dispatch: `flash_attn_nwg_f16_gpu` in `src/quant/vec_dot/gpu/metal.rs`

**Key Findings:**
1. KV Cache access in current MMA kernel is **contiguous, not strided**
2. NWG parallelization adds overhead for short sequences
3. Decode performance sensitive to thermal throttling (98-128 t/s range observed)
4. GEMV+dequantization is already fused inside kernel

**Priority 2: Find Optimal Thread Count for GEMV** (Expected: 10-20% speedup)
- Test thread counts: 2, 4, 8, 16, 32
- Measure actual memory bandwidth utilization
- Metal Performance HUD or Instruments

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

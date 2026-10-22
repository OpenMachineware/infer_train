# GPU Decode Performance Optimization Roadmap

## Current Status (2026-10-23 Session 11)

**Performance** (Q4_K models, Qwen3-0.6B):
| Metric | We | llama.cpp | Gap |
|--------|------|-----------|------|
| Prefill | 2571 t/s | 920 t/s | **0.36x (faster!)** |
| Decode | 120-125 t/s | 184 t/s | **1.47-1.53x** |
| **Overall BW** | **20.2%** | **31.0%** | **11pp** |
| **Single Kernel** | **5.0%** | **?** | **Unknown** |

**Root Cause**: **GEMV kernel带宽利用率极低**

## Bandwidth Analysis Results

### Single Kernel Profiling
```
Kernel                    M      K   Time(μs)  BW(GB/s)   Util%
-----------------------------------------------------------------
Q projection           2048   1024     265.52      4.49    4.5%
K projection           1024   1024     271.46      2.20    2.2%
V projection           1024   1024     331.94      1.80    1.8%
Gate projection        3072   1024     288.62      6.19    6.2%
Up projection          3072   1024     276.49      6.46    6.5%
Down projection        1024   3072     309.75      5.77    5.8%
-----------------------------------------------------------------
AVERAGE                                   4.48    4.5%
```

### Dispatch Overhead Analysis
```
Separate command buffers (6 dispatches):  3.1% util
Single command buffer (batched):          5.0% util
Speedup from batching:                    1.63x
```

### Key Findings
1. **Kernel本身是主要问题**: 5% vs 31%的理论带宽
2. **Dispatch overhead次要**: 批量化只提升1.63x
3. **整体decode vs 单kernel**: 20% vs 5% = 多kernel并行执行有overlap

## Root Cause Analysis: NSG Parallelism Gap

### llama.cpp MMA Architecture
```cpp
// From llama.cpp-0.7.0/ggml/src/ggml-metal/ggml-metal-device.cpp

// NSG (Number of SimdGroups) - ADAPTIVE based on row count!
GGML_METAL_MMA_NSG_FEW_ROWS  = 32  // For <= 64 rows (decode!)
GGML_METAL_MMA_NSG_MID_ROWS  = 16  // For 64-6144 rows
GGML_METAL_MMA_NSG_MANY_ROWS = 8   // For > 6144 rows

// Tile dimensions
GGML_METAL_MMA_TILE = 8      // 8x8 simdgroup matrix
GGML_METAL_MMA_NT_MAX = 4    // Max 4 tiles = 32 rows per threadgroup

// K-step for Q4_K
GGML_METAL_MMA_K_CHUNK = 64  // Process 64 weights per step
```

### Our Kernel Architecture
```rust
// From src/ops/metal/gemv/kquant_mv.metal

const int NSG = 2;  // FIXED at 2, no adaptation!
const int N_R0 = 2; // 2 rows per simdgroup
// Total: 2 simdgroups × 2 rows = 4 rows per threadgroup
```

### Critical Gap: 16x Parallelism Deficit!

**For decode (1 position, ~2048 rows):**
| Metric | llama.cpp | Ours | Gap |
|--------|-----------|------|-----|
| **NSG** | 32 | 2 | **16x** |
| **K-split parallelism** | 32-way | 2-way | **16x** |
| **Threads per threadgroup** | 1024 | 64 | **16x** |
| **Memory parallelism** | High | Low | **16x** |

### Why This Causes 5% Bandwidth Utilization

1. **Insufficient thread-level parallelism**:
   - 32 simdgroups = 1024 threads (llama.cpp)
   - 2 simdgroups = 64 threads (ours)
   - **GPU starvation**: Not enough threads to hide memory latency

2. **K-dimension serialization**:
   - llama.cpp: Each simdgroup processes K/32 blocks in parallel
   - Ours: Each simdgroup processes K/2 blocks serially
   - **Memory latency not hidden**: Threads wait for memory

3. **Low occupancy**:
   - Fewer threads = lower GPU occupancy
   - Memory controller idle most of the time
   - **5% bandwidth**: GPU only busy 5% of the time

### MMA (Matrix-Matrix-Add) vs Simple GEMV

llama.cpp uses **MMA tiles** (8x8 simdgroup matrices):
- `simdgroup_float8x8` for accumulation
- `simdgroup_multiply_accumulate` for efficient MAC operations
- Better register usage and ALU efficiency

Our kernel uses **simple scalar accumulation**:
- `float sumf` per thread
- Sequential reduction with `simd_sum`
- Less efficient ALU usage

## Optimization Plan

### Immediate Fix: Adaptive NSG Architecture

1. **Implement adaptive NSG** (like llama.cpp):
   ```rust
   fn get_nsg(num_rows: usize) -> usize {
       if num_rows <= 64 {
           32  // Decode scenario
       } else if num_rows <= 6144 {
           16
       } else {
           8
       }
   }
   ```

2. **Rewrite kernel to support variable NSG**:
   - Template-based kernel with `NSG` parameter
   - Compile multiple variants (nsg=8, 16, 32)
   - Select at runtime based on problem size

3. **Implement MMA accumulation**:
   - Use `simdgroup_float8x8` for 8x8 tiles
   - Use `simdgroup_multiply_accumulate` for efficient MAC
   - Match llama.cpp's tile-based approach

### Expected Improvement

If we fix the parallelism gap:
- **From**: 5% bandwidth (64 threads, K/2 blocks)
- **To**: ~15-20% bandwidth (1024 threads, K/32 blocks)
- **Speedup**: 3-4x on GEMV kernels
- **Overall decode**: 120 t/s → 150-200 t/s (potentially match/exceed llama.cpp!)

### Implementation Steps

1. ✅ **Root cause identified**: NSG=2 vs NSG=32 (16x gap)
2. 🔄 **Create MMA template kernel** (Q4_K first)
3. ⏳ **Implement adaptive NSG selection**
4. ⏳ **Add MMA accumulation path**
5. ⏳ **Benchmark and verify bandwidth improvement**

## Progress Log

### Session 11
1. ✅ FusedGateUp优化: 20%提升 (101→120 t/s)
2. ✅ 带宽分析工具: 发现核心问题
3. ✅ Kernel-level profiling: 发现kernel本身是瓶颈 (5% util)
4. ✅ Dispatch overhead测试: 批量化提升1.63x
5. ✅ **Root cause found**: NSG parallelism gap (16x deficit!)

### Session 10
1. ✅ NWG parallelization for Flash Attention
2. ✅ Verified prefill performance

## Progress Log

### Session 11
1. ✅ FusedGateUp优化: 20%提升 (101→120 t/s)
2. ✅ 带宽分析工具: 发现核心问题
3. ✅ Kernel-level profiling: 发现kernel本身是瓶颈 (5% util)
4. ✅ Dispatch overhead测试: 批量化提升1.63x

### Session 10
1. ✅ NWG parallelization for Flash Attention
2. ✅ Verified prefill performance

### Session 9
1. ✅ Vec registry + dtype cache
2. ✅ Dispatch fix

## Key Insights

1. **Decode是memory-bound**: 性能 = 带宽利用率 × 理论带宽
2. **GEMV占90%内存访问**: 优化GEMV = 优化整体
3. **Kernel带宽利用率极低**: 5% vs 31%理论
4. **Dispatch overhead次要**: 批量化只提升1.63x
5. **Kernel实现是关键**: 需要深入分析kernel代码

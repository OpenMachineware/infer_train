# Flash Decode Performance Analysis

## Executive Summary

Our flash decode kernels are **3-4x slower** than llama.cpp at long sequences. The root cause is a fundamental architectural difference in how SIMD groups are utilized.

## Benchmark Results

### Our Implementation (GFLOPS)
| Format | seq_k=256 | seq_k=512 | seq_k=1024 | seq_k=2048 | seq_k=4096 |
|--------|-----------|-----------|------------|------------|------------|
| Q4_0   | 10.62     | 16.07     | 26.73      | 37.32      | 47.26      |
| Q4_1   | 11.18     | 15.09     | 26.31      | 34.47      | 43.37      |
| Q5_0   | 10.31     | 15.08     | 23.53      | 34.57      | 37.03      |
| Q5_1   | 8.61      | 15.40     | 23.57      | 31.67      | 36.24      |
| Q8_0   | 8.13      | 14.81     | 23.71      | 33.18      | 32.21      |
| BF16   | 7.98      | 11.90     | 16.29      | 20.23      | 21.26      |

### llama.cpp Implementation (GFLOPS)
| Format | seq_k=256 | seq_k=512 | seq_k=1024 | seq_k=2048 | seq_k=4096 |
|--------|-----------|-----------|------------|------------|------------|
| F16    | 15.31     | 36.08     | 59.26      | 116.73     | 191.82     |
| BF16   | 12.73     | 30.58     | 47.49      | 82.76      | 147.25     |
| Q4_0   | 14.50     | 30.58     | 55.19      | 95.82      | 173.10     |
| Q4_1   | 13.66     | 27.47     | 54.38      | 95.28      | 177.45     |
| Q5_0   | 13.77     | 23.15     | 43.80      | 79.65      | 130.09     |
| Q5_1   | 14.08     | 23.02     | 45.09      | 78.59      | 128.69     |
| Q8_0   | 16.71     | 26.41     | 56.68      | 96.70      | 169.78     |

### Performance Ratio (Our / llama.cpp)
| Format | seq_k=256 | seq_k=512 | seq_k=1024 | seq_k=2048 | seq_k=4096 |
|--------|-----------|-----------|------------|------------|------------|
| Q4_0   | 73%       | 53%       | 48%        | 39%        | 27%        |
| Q4_1   | 82%       | 55%       | 48%        | 36%        | 24%        |
| Q5_0   | 75%       | 65%       | 54%        | 43%        | 28%        |
| Q5_1   | 61%       | 67%       | 52%        | 40%        | 28%        |
| Q8_0   | 49%       | 56%       | 42%        | 34%        | 19%        |
| BF16   | 63%       | 39%       | 34%        | 24%        | 14%        |

## Architectural Analysis

### llama.cpp Architecture

**SIMD Group Strategy:**
- seq_k ≤ 2048: **NSG = 1**
- seq_k ≤ 4096: **NSG = 2**
- seq_k > 4096: **NSG = 4**

**Processing Pattern:**
```
- C = 32 cache items per iteration
- Each SIMD group processes different chunks of KV cache
- For NSG=1: single SIMD group processes entire KV cache sequentially
- For NSG=2: 2 SIMD groups split KV cache, process in parallel
- Uses hierarchical tree reduction for efficiency
```

**Key Optimizations:**
1. **Minimal SIMD Groups:** Avoids synchronization overhead
2. **Large Chunk Size:** 32 cache items per iteration (we use 4)
3. **Tree Reduction:** Efficient SIMD operations for reduction
4. **Threadgroup Memory:** Shared memory per SIMD group for softmax

### Our Architecture

**SIMD Group Strategy:**
- All seq_k: **NSG = 8**

**Processing Pattern:**
```
- CHUNK = 4 cache items per iteration
- 8 SIMD groups split KV cache
- Each group processes seq_k/8 items
- Requires final reduction across 8 groups
```

**Performance Bottlenecks:**
1. **Too Many SIMD Groups:** 8 groups means:
   - 8x synchronization overhead
   - 8x threadgroup memory usage (512 + nsg*16 + nsg*512 bytes)
   - Less cache locality per group
   - Final reduction overhead across 8 groups

2. **Small Chunk Size:** 4 items vs llama.cpp's 32:
   - More iterations needed (seq_k/4 vs seq_k/32)
   - More loop overhead
   - Less amortized memory access cost

3. **Inefficient Reduction:** Our reduction across 8 SIMD groups requires:
   - Multiple threadgroup barriers
   - Sequential write/read from shared memory
   - No hierarchical tree reduction optimization

## Why More SIMD Groups = Slower

For decode (seq_q=1):
- Total work = seq_k * 128 FLOPs per cache item
- With NSG=1: 1 SIMD group does all work, minimal overhead
- With NSG=8: 8 SIMD groups each do 1/8 work, but:
  - 8x threadgroup memory usage
  - 8x synchronization points
  - Final reduction across 8 groups
  - Less L1 cache hit rate per group

**Critical Insight:** For small batches, overhead dominates. llama.cpp optimizes for minimal overhead (NSG=1-2), we maximize parallelism (NSG=8) which backfires.

## Recommendations

### Immediate Fix: Adaptive NSG Strategy

Match llama.cpp's strategy:
```rust
fn get_optimal_nsg(seq_k: usize) -> usize {
    if seq_k <= 2048 { 1 }
    else if seq_k <= 4096 { 2 }
    else { 4 }
}
```

Expected improvement:
- seq_k ≤ 2048: **2-3x speedup** (NSG 8→1)
- seq_k = 4096: **1.5-2x speedup** (NSG 8→2)

### Medium-term: Increase Chunk Size

Change CHUNK from 4 to 32:
- 8x fewer iterations
- Better memory access patterns
- Amortize loop overhead

Expected improvement: **20-30% additional speedup**

### Long-term: Rewrite Using llama.cpp Template

Port llama.cpp's `fa_vec_common.metal` template exactly:
- Tree reduction pattern
- Optimal threadgroup memory layout
- Proven architecture

Expected result: **Match or exceed llama.cpp performance**

## Action Plan

1. **Phase 1:** Implement adaptive NSG selection (1-2 hours)
   - Modify benchmark to test g1/g2/g4 variants
   - Create adaptive dispatch logic

2. **Phase 2:** Increase chunk size and optimize (4-6 hours)
   - Modify kernel to use CHUNK=32
   - Optimize threadgroup memory layout
   - Test correctness and performance

3. **Phase 3:** Port llama.cpp template (8-12 hours)
   - Copy template structure exactly
   - Implement quantized format variants
   - Verify performance matches llama.cpp

## Conclusion

Our kernels are fundamentally inefficient for decode workloads. The root cause is using too many SIMD groups (8 vs llama.cpp's 1-2), which creates massive overhead that outweighs parallelism benefits. The fix is straightforward: adopt llama.cpp's adaptive NSG strategy.

**Goal:** "每种格式都稳定超过对方" requires matching llama.cpp's architecture, not just optimizing our current approach.

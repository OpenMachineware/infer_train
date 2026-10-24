# GPU Inference Optimization Roadmap

## Current Performance Status

Based on profiling data:
- Encode time: 2.3 ms (19% of total)
- GPU time: 10 ms (81% of total)
- Performance: 123.4 t/s vs llama.cpp 157.6 t/s (2x bandwidth gap)
- **Critical bottleneck**: Memory bandwidth utilization (8.8% vs llama.cpp 18.1%)

## Root Cause Analysis

The performance gap is NOT due to:
- Encode overhead (only 19% of total time)
- Kernel efficiency (slightly better than llama.cpp)
- Buffer allocation (reuse pool already in place)

The gap IS due to:
- **Memory bandwidth underutilization** - kernels not executing concurrently enough
- **Lack of fine-grained GPU barriers** - memory range tracking implemented but barriers disabled
- **Serial execution within single CB** - even with MTLDispatchType::Concurrent

## Optimization Priorities (Reordered)

### 1. Fine-Grained Kernel Parallelism (Priority: CRITICAL)

**Goal**: Maximize concurrent kernel execution to saturate memory bandwidth.

**Why**:
- Memory bandwidth is 8.8% vs llama.cpp 18.1% - this is the root cause
- MTLDispatchType::Concurrent is enabled but not effective enough
- Memory range tracking is implemented but GPU barriers are disabled
- Need explicit dependency-based dispatch to enable true parallelism

**Implementation**:

Phase 1: Enable GPU barriers using `memoryBarrierWithScope`
```rust
// src/dag/memory_ranges.rs
fn insert_gpu_barrier(
    encoder: &ComputeCommandEncoderRef,
    conflicting_buffers: &[Buffer],
) {
    // Option 1: Use Objective-C bridge for memoryBarrierWithScope
    // Option 2: Use memory_barrier_with_resources (Rust metal crate)
    // Option 3: Use MTLFence for fine-grained synchronization

    // Current barrier implementation (needs testing):
    unsafe {
        let encoder_ptr = encoder.as_ptr();
        msg_send![encoder_ptr, memoryBarrierWithScope: MTLMemoryScopeDevice];
    }
}
```

Phase 2: Dependency-based dispatch
```rust
// src/dag/executor.rs
// Build dependency graph from DAG
// Dispatch independent nodes concurrently
// Insert barriers only when needed (conflict detection from memory ranges)
```

**Expected improvement**:
- 2x memory bandwidth utilization (8.8% -> 18%)
- Total improvement: 50-100% (bandwidth is critical path)

**Files to modify**:
- `src/dag/memory_ranges.rs`: Enable GPU barriers
- `src/dag/executor.rs`: Dependency-based dispatch
- Add Objective-C bridge if needed

**Status**: Memory range tracking implemented, GPU barriers disabled

---

### 2. Multi-Threaded Encoding (Priority: MEDIUM)

**Goal**: Reduce encode overhead and enable parallel encoding.

**Why**:
- Encode is 19% of total time - significant but not critical
- llama.cpp uses `dispatch_apply` with `n_cb=1` or `n_cb=2` for parallel encoding
- Current single-threaded encoding is a bottleneck

**Challenge**: ExecutionContext has lifetime constraints and mutable references, making cross-thread transfer difficult.

**Solution Options**:
1. **Multiple command buffers** (llama.cpp approach):
   - Create 2-3 command buffers in parallel
   - Split DAG by layer groups
   - Wait for all CBs to complete

2. **Thread-safe ExecutionContext**:
   - Use Arc<Mutex<>> for shared state
   - Limit mutation to single writer
   - More complex but keeps single CB

**Expected improvement**:
- 10-20% encode time reduction
- Total improvement: 5-10%

**Status**: Architecture redesign needed, deferred

---

### 3. Buffer Pre-Allocation (Priority: LOW)

**Goal**: Eliminate runtime buffer allocation overhead.

**Current state**:
- Reuse pool already implemented (ForwardBufferPool)
- Weight buffers cached at model load
- Intermediate tensor buffers reused across forward passes

**Why LOW priority**:
- Buffer reuse pool is already effective
- First forward pass has allocation overhead, but subsequent passes reuse
- Impact is minimal compared to bandwidth utilization

**Potential improvement**: 2-5% on first pass, negligible on subsequent passes

**Status**: Already partially implemented via reuse pool

## Success Metrics

- Target: Match or exceed llama.cpp performance (157.6 t/s)
- Current: 123.4 t/s
- Gap: 2x bandwidth utilization

Expected total improvement: 20-30% combined

## References

- [Multi-Thread Encoding Purpose](memory/multithread-encoding-purpose.md)
- [Memory Range Tracking Impl](memory/memory-range-tracking-impl.md)
- [GPU Decode Performance Comparison](memory/gpu-decode-performance-comparison.md)
- [Bandwidth Utilization Approach](memory/bandwidth-utilization-approach.md)

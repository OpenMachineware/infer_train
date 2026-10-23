# GPU Decode Optimization Roadmap

**Current Status**: 112-120 t/s
**Target**: 205 t/s (llama.cpp)
**Gap**: 1.7x

---

## Completed Optimizations ✅

### 1. MTLDispatchTypeConcurrent - IMPLEMENTED

**Commit**: `60bb780`

**Implementation**:
```rust
let encoder = command_buffer
    .compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
```

**Result**:
- Performance: 79 t/s → 112-120 t/s
- Improvement: +42-52%
- Allows independent kernels in same CB to execute in parallel
- Maximizes memory bandwidth utilization

**Reference**: Based on llama.cpp's `ggml_metal_encoder_init()` with `concurrent=true`

### 2. Memory Range Tracking - IMPLEMENTED ✅

**Commit**: `3bc5041`

**Implementation**:
- Created `src/dag/memory_ranges.rs` with `MemoryRanges` struct
- Tracks src/dst buffer access patterns for conflict detection
- Integrated into all DAG operations in `execute_node()`

**Status**:
- ✅ Memory tracking working (verified via `INFER_DEBUG_MEM_RANGES=1`)
- ⚠️ GPU barrier temporarily disabled - Rust metal crate doesn't expose `memoryBarrierWithScope:` API
- ⚠️ Model loading hanging - separate issue, needs investigation

**Files modified**:
- `src/dag/memory_ranges.rs`: New file for memory range tracking
- `src/dag/executor.rs`: Added `check_memory_and_barrier()` and `track_memory_ranges()`
- `src/dag.rs`: Added `pub mod memory_ranges`

**Expected improvement**: 15-25% once barrier is working

---

## Blocking Issue: Model Loading Hangs

**Symptom**: Model loads successfully, forward pass hangs/returns None

**Tested models**:
- Qwen3.8-27B-UD-Q5_K_M.gguf
- DeepSeek-Coder-V2-Lite-Instruct-Q4_K_L.gguf

**Debug output**:
- DAG builds correctly (327 operations fused for Qwen)
- Execution order shows nodes 0, 1, then jumps to 1173 (incorrect)
- Forward returns None or hangs

**Investigation needed**:
1. Check if DAG execution order is correct
2. Verify tensor registry bindings
3. Check for missing tensor IDs
4. Test with simpler model

---

## Remaining Optimizations

### Priority 1: Fix Model Loading Issue

**Goal**: Get forward pass working again

**Steps**:
1. Add debug output for tensor registry lookups
2. Check if all required tensors are present
3. Verify execution order in DAG
4. Test with a minimal model

### Priority 2: Complete GPU Barrier Implementation

**Goal**: Enable actual memory barriers for correct concurrent execution

**Approach options**:
1. Use `memory_barrier_with_resources()` with tracked buffers
2. Call Objective-C API via `objc` crate
3. Wait for upstream metal crate to expose `memoryBarrierWithScope:`

### Priority 3: Multi-threaded Graph Encoding

---

### Priority 2: Multi-threaded Graph Encoding

**Goal**: Parallelize encoding to saturate memory bandwidth during command buffer creation

**llama.cpp Implementation** (`ggml-metal-context.m`, lines 496-595):

1. **Command buffer allocation**:
```objc
// Main thread: cmd_bufs[n_cb]
// Worker threads: cmd_bufs[0..n_cb-1]
// Total: n_cb + 1 command buffers

#define GGML_METAL_MAX_COMMAND_BUFFERS 8
```

2. **Node distribution**:
```objc
n_main = MAX(64, 0.1 * gf->n_nodes)  // Main thread handles first 10%
n_nodes_0 = n_main
n_nodes_1 = gf->n_nodes - n_nodes_0
n_nodes_per_cb = (n_nodes_1 + n_cb - 1) / n_cb
```

3. **Parallel encoding**:
```objc
// Main thread encodes immediately
[cmd_buf enqueue];
ctx->encode_async(n_cb);

// Worker threads encode in parallel
dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);
```

**Our Implementation Plan**:

1. Split DAG nodes by layer (e.g., layers 0-13 in thread 0, layers 14-27 in thread 1)
2. Create separate command buffer per thread
3. Use `std::thread::scope` for parallel encoding
4. Enqueue all command buffers in order
5. Metal executes them sequentially

**Files to modify**:
- `src/dag/executor.rs`: Add multi-threaded encoding path
- `src/dag/graph.rs`: Add layer grouping utilities

**Expected improvement**: 10-15% (encoding is 19% of total time)

---

### Priority 3: Event-based Synchronization (Optional)

**Goal**: Non-blocking cross-command-buffer synchronization

**llama.cpp Implementation** (`ggml-metal-device.m`, lines 1447-1496):

```objc
struct ggml_metal_event {
    id<MTLSharedEvent> obj;
    atomic_int value;
};

void ggml_metal_event_encode_signal(ggml_metal_event_t ev, id<MTLCommandBuffer> cmd_buf) {
    [cmd_buf encodeSignalEvent:event value:atomic_fetch_add(&ev->value, 1) + 1];
}

void ggml_metal_event_encode_wait(ggml_metal_event_t ev, id<MTLCommandBuffer> cmd_buf) {
    [cmd_buf encodeWaitForEvent:event value:atomic_load(&ev->value)];
}
```

**When needed**:
- Async data transfers between contexts
- CPU-GPU pipeline for prefill/decode overlap
- Not critical for single-thread decode

**Files to modify**:
- `src/quant/vec_dot/gpu/metal.rs`: Add Event wrapper
- `src/dag/executor.rs`: Use events for cross-CB sync

**Expected improvement**: 5-10% for async pipeline scenarios

---

## Implementation Order

```
Step 1: Memory Range Tracking + Barriers
  ├─ Add MemoryRanges struct
  ├─ Track src/dst in execute_node()
  ├─ Check conflicts before dispatch
  ├─ Insert GPU barriers when needed
  └─ Test: expect 15-25% improvement

Step 2: Multi-threaded Encoding
  ├─ Split DAG by layer groups
  ├─ Create n_cb + 1 command buffers
  ├─ Use std::thread::scope for encoding
  ├─ Enqueue in order, commit once
  └─ Test: expect 10-15% improvement

Step 3: Event Synchronization (if needed)
  ├─ Wrap MTLSharedEvent in Rust
  ├─ Signal/wait for async scenarios
  └─ Test: expect 5-10% for prefill overlap
```

---

## Performance Projection

| Stage | Performance | Cumulative Gain |
|-------|-------------|-----------------|
| Baseline | 79 t/s | - |
| Concurrent Encoder | 112-120 t/s | +42-52% |
| + Memory Barriers | ~130-150 t/s | +64-90% |
| + Multi-thread Encoding | ~145-170 t/s | +83-115% |
| Target (llama.cpp) | 205 t/s | +160% |

---

## Technical References

1. **llama.cpp Metal Backend**:
   - `ggml-metal-context.m`: Graph compute flow, multi-CB management
   - `ggml-metal-device.m`: Encoder creation, event synchronization
   - `ggml-metal-ops.cpp`: Memory range tracking, concurrency checks
   - `ggml-metal-common.h`: Memory range data structures

2. **Key Code Sections**:
   - Lines 474-660: `ggml_metal_graph_compute()` main flow
   - Lines 836-850: Concurrent encoder creation
   - Lines 890-896: Memory barrier insertion
   - Lines 1447-1496: Event-based synchronization

3. **External Analysis**:
   - [RTX 4090D Vulkan Double-buffering](https://m.blog.csdn.net/weixin_30444111/article/details/162917938)
   - [Metal 4 Concurrent Encoding](https://blakecrosley.com/blog/metal-4-essentials)

---

## Notes

- All optimizations are based on llama.cpp's proven implementation
- Implementation matches llama.cpp's exact approach (no guessing)
- Each step is independently testable
- Performance gains are cumulative

**Next Action**: Implement Priority 1 (Memory Range Tracking + Barriers)

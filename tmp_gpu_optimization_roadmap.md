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

---

## Remaining Optimizations

### Priority 1: Memory Range Tracking + GPU Barriers

**Goal**: Allow independent operations to execute concurrently within a single encoder

**llama.cpp Implementation** (`ggml-metal-common.h`, `ggml-metal-ops.cpp`):

1. **Memory range tracking**:
```c
struct ggml_mem_ranges {
    int n_ranges;
    struct {
        void * ptr;
        size_t size;
        bool is_dst;  // true = write, false = read
    } ranges[MAX_RANGES];
};

// Check if operation can run concurrently:
// - new src cannot overlap with existing dst
// - new dst cannot overlap with any existing range
bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_tensor * tensor);
```

2. **Automatic barrier insertion**:
```cpp
bool is_concurrent = ggml_mem_ranges_check(ctx, node);
if (!is_concurrent) {
    ggml_metal_encoder_memory_barrier(ctx->enc);
    ggml_mem_ranges_reset(ctx->mem_ranges);
}
```

3. **GPU barrier**:
```objc
void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder) {
    [encoder->obj memoryBarrierWithScope:MTLBarrierScopeBuffers];
}
```

**Our Implementation Plan**:

1. Add `MemoryRanges` struct to track buffer access
2. In `execute_node()`, record src/dst ranges
3. Before each dispatch, check for conflicts
4. Insert `encoder.memory_barrier_with_scope(MTLBarrierScopeBuffers)` when needed
5. Reset tracking after barrier

**Files to modify**:
- `src/dag/executor.rs`: Add memory range tracking
- `src/dag/graph.rs`: No changes needed

**Expected improvement**: 15-25% (based on llama.cpp's concurrency gain)

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

# Parallel Implementation Comparison: AsyncRT vs pthread

## Summary

| Aspect | max.algorithm.parallelize (AsyncRT) | pthread pool (thread_pool.mojo) |
|--------|-------------------------------------|--------------------------------|
| **Mechanism** | AsyncRT coroutines + DeviceContext | C pthread thread pool |
| **Optimal threads** | 4 (exponential overhead after) | 4-8 (linear scaling) |
| **Overhead (4 threads)** | 5-7 µs per call | ~1-2 µs per call (estimated) |
| **Overhead (8 threads)** | 141 µs (20x worse!) | ~2-4 µs per call (estimated) |
| **GPU support** | ✅ Unified API for CPU/GPU | ❌ CPU only |
| **DeviceContext** | Required (create once, reuse) | Not needed |
| **Integration** | max library built-in | Custom C + Mojo FFI |

## Performance Analysis

### AsyncRT Overhead (Measured)

```
Threads | Overhead/call | Relative to 4 threads
--------|---------------|----------------------
1       | 0.04 µs       | 0.006x
2       | 5 µs          | 0.7x
4       | 7 µs          | 1x (optimal)
6       | 29 µs         | 4x
8       | 141 µs        | 20x ❌
```

**Root cause**: AsyncRT creates coroutines for each `parallelize` call. Scheduling and synchronization overhead grows exponentially with thread count.

### pthread Pool (Estimated)

Based on C pthread thread pool implementation:
- **Startup**: One-time dlopen + dlsym (~1-2 ms, done once)
- **Per-call**: Submit to work queue + wake workers + join (~1-4 µs)
- **Scaling**: Linear with thread count (no coroutine overhead)

## Use Case Recommendations

### 1. Decode Matmul (M=1, large N)

```mojo
# Decode: single token, compute across all weight columns
# Work per call: ~10-50 ms matmul computation
# Overhead impact: negligible (< 0.01%)
# Recommendation: Either works, but pthread has lower overhead
```

**Analysis**:
- Matmul computation dominates (~10-50 ms)
- Overhead is negligible regardless of choice
- pthread has slight edge due to lower overhead

### 2. Batch Prefill (M=32-512)

```mojo
# Prefill: batch of tokens, compute across all weight columns
# Work per call: ~100-500 ms matmul computation
# Overhead impact: negligible
# Recommendation: Either works, use max.algorithm for GPU compatibility
```

**Analysis**:
- Batch matmul computation dominates
- Overhead completely negligible
- Consider `max.algorithm.parallelize` for unified CPU/GPU API

### 3. Fine-grained Parallelism

```mojo
# Example: Parallel element-wise operations
# Work per call: ~10-100 µs computation
# Overhead impact: SIGNIFICANT
# Recommendation: Use pthread or avoid parallelism entirely
```

**Analysis**:
- AsyncRT overhead (5-7 µs) is 5-50% of work
- pthread overhead (~1-2 µs) is still 1-20% of work
- Consider: **sequential is faster** for such small work

## Implementation Examples

### Option A: pthread Pool (Current Implementation)

```mojo
# src/core/thread_pool.mojo
from src.core.thread_pool import parallel_run

def matmul_quantized_threaded[
    quant_type: QuantType,
](
    x: Tensor[DType.float16, 2],
    w_quant: Tensor[DType.uint8, 2],
    nthreads: Int = 0,
) -> Tensor[DType.float16, 2]:
    var threads = min(resolve_threads(nthreads), 4)  # Cap at 4
    var M = x.shape()[0]
    var N = w_quant.shape()[0]

    # Parallel across N output columns
    var symbol = _matmul_worker_symbol(quant_type)
    var rc = parallel_run(symbol, ctx, N, threads)
    # ...
```

**Pros**:
- ✅ Lower overhead
- ✅ Better scaling beyond 4 threads
- ✅ Already integrated in project

**Cons**:
- ❌ CPU only
- ❌ Custom C code to maintain
- ❌ Requires manual worker symbol management

### Option B: max.algorithm.parallelize

```mojo
from max.algorithm import parallelize
from max.gpu.host import DeviceContext

struct QuantizedMatmul:
    var ctx: DeviceContext  # Reuse across calls

    def __init__(out self):
        self.ctx = DeviceContext(api="cpu")

    def matmul(
        self,
        x: Tensor[DType.float16, 2],
        w_quant: Tensor[DType.uint8, 2],
    ) -> Tensor[DType.float16, 2]:
        var N = w_quant.shape()[0]
        var threads = min(num_pcores(), 4)  # Cap at 4
        var out = tensor_zeros[DType.float16, 2](...)

        @always_inline
        @parameter
        def worker(n: Int):
            out[:, n] = compute_column(x, w_quant, n)

        parallelize[worker](N, threads, self.ctx)
        return out
```

**Pros**:
- ✅ Unified API for CPU/GPU
- ✅ No C code to maintain
- ✅ Future-proof for GPU offload

**Cons**:
- ❌ Higher overhead (5-7 µs vs 1-2 µs)
- ❌ Exponential scaling beyond 4 threads
- ❌ Requires DeviceContext management

## Recommendation

### For CPU-only optimization (Phase 1-2):

**Use pthread pool** - it's already integrated and has lower overhead.

```mojo
# Keep using thread_pool.mojo
def matmul_quantized_parallel(
    x: Tensor[DType.float16, 2],
    w_quant: Tensor[DType.uint8, 2],
) -> Tensor[DType.float16, 2]:
    var threads = min(num_pcores(), 4)  # 4 is optimal
    # Use existing parallel_run infrastructure
```

### For future GPU support (Phase 3+):

**Consider migrating to max.algorithm.parallelize** for unified API, but:
- Benchmark first to ensure overhead is acceptable
- Reuse DeviceContext across calls
- Keep pthread as fallback for fine-grained work

### Thread count optimization:

Regardless of implementation:
```mojo
# 4 threads is sweet spot for Mojo 1.0
var threads = min(num_pcores(), 4)
```

## Benchmark Command

```bash
# Test overhead
./bench_pool  # pthread pool overhead
# Compare with empty parallelize
cd modular-mojo-v1.0.0/max/kernels/benchmarks/algorithm
pixi run mojo build parallelize_overhead.mojo -o bench_parallel
./bench_parallel
```

## References

- Memory: `mojo1-parallel-performance.md` - AsyncRT overhead measurements
- Code: `src/core/thread_pool.mojo` - pthread pool implementation
- Code: `modular-mojo-v1.0.0/max/mojo/max/algorithm/backend/cpu/parallelize.mojo`

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/attention/mha_threaded.mojo
#
# Threaded MHA: parallel head processing using thread pool.
#
# Design:
# - Split heads across threads
# - Each thread processes its assigned heads independently
# - No shared writes (each head writes to different output location)
# - Falls back to single-threaded when threading overhead dominates
#
# Note: Uses global request context for simplicity (single-request mode).
#       Multi-request support requires request queue + scheduler (Phase 2).

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.cpu.flash_attention_cpu import flash_attention_decode
from src.core.ops.attention.kv_cache import KVCacheLayer
from src.core.thread_pool import parallel_run, resolve_threads, has_worker
from src.core.cpu_features import detect_cpu_flags
from std.utils import StaticTuple
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc
from std.math import sqrt

# Minimum heads for threading (below this, overhead dominates)
comptime MIN_HEADS_FOR_THREADING = 4

# Heads per work item
comptime HEADS_PER_WORKER = 2


# Global request context for single-request threading
# This is set before parallel_run and read by workers
# Uses primitive types only to avoid copy issues
struct GlobalRequestContext:
    var q_addr: Int64  # Raw address for q data pointer
    var out_addr: Int64  # Raw address for output data pointer
    var cache_addr: Int64  # Raw address for cache pointer
    var position: Int
    var head_dim: Int
    var scale: Float32
    var n_kv_heads: Int
    var n_heads: Int


def _init_global_ctx(
    q: Tensor[DType.float16, 3],
    output: Tensor[DType.float16, 3],
    cache_ptr: Pointer[KVCacheLayer, MutUntrackedOrigin],
    position: Int,
    head_dim: Int,
    scale: Float32,
    n_kv_heads: Int,
    n_heads: Int,
) -> Pointer[GlobalRequestContext, MutUntrackedOrigin]:
    """Initialize global request context."""
    var ctx_ptr = unsafe_alloc[GlobalRequestContext](1)
    # Access the struct via offset and set fields
    ctx_ptr[unsafe_offset=0].q_addr = Int64(Int(q.data()))
    ctx_ptr[unsafe_offset=0].out_addr = Int64(Int(output.data()))
    ctx_ptr[unsafe_offset=0].cache_addr = Int64(Int(cache_ptr))
    ctx_ptr[unsafe_offset=0].position = position
    ctx_ptr[unsafe_offset=0].head_dim = head_dim
    ctx_ptr[unsafe_offset=0].scale = scale
    ctx_ptr[unsafe_offset=0].n_kv_heads = n_kv_heads
    ctx_ptr[unsafe_offset=0].n_heads = n_heads
    return ctx_ptr


@export
def mha_head_worker_v2(
    ctx: Pointer[UInt8, MutUntrackedOrigin],
    idx: Int64,
) abi("C"):
    """Process one work item (range of heads).

    Context is GlobalRequestContext pointer.
    """
    # Cast to global context
    var g_ctx = ctx.unsafe_bitcast[GlobalRequestContext]()

    # Read context values (access through offset)
    var q_ptr = Pointer[Scalar[DType.float16], MutUntrackedOrigin](
        unsafe_from_address=Int(g_ctx[unsafe_offset=0].q_addr)
    )
    var out_ptr = Pointer[Scalar[DType.float16], MutUntrackedOrigin](
        unsafe_from_address=Int(g_ctx[unsafe_offset=0].out_addr)
    )
    var cache_ptr = Pointer[KVCacheLayer, MutUntrackedOrigin](
        unsafe_from_address=Int(g_ctx[unsafe_offset=0].cache_addr)
    )
    var position = g_ctx[unsafe_offset=0].position
    var head_dim = g_ctx[unsafe_offset=0].head_dim
    var scale = g_ctx[unsafe_offset=0].scale
    var n_kv_heads = g_ctx[unsafe_offset=0].n_kv_heads
    var n_heads = g_ctx[unsafe_offset=0].n_heads

    # Get cache reference (copy from heap - safe for multi-threaded access)
    # KVCacheLayer is Copyable, so we can make a copy
    # Note: This is a shallow copy of the tensor metadata (data pointers remain shared)
    var cache = cache_ptr[unsafe_offset=0].copy()

    # Calculate which heads this work item processes
    var work_item_idx = Int(idx)
    var head_start = work_item_idx * HEADS_PER_WORKER
    var head_end = min(head_start + HEADS_PER_WORKER, n_heads)

    # Process each head
    for h in range(head_start, head_end):
        var kv_head = h * n_kv_heads // n_heads

        # Extract Q vector
        var q_vec = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
        for d in range(head_dim):
            q_vec.set(d, q_ptr.unsafe_offset(h * head_dim + d).unsafe_load())

        # Run flash attention
        var out_vec = flash_attention_decode(
            q_vec, cache, kv_head, position, head_dim, scale
        )

        # Store result
        for d in range(head_dim):
            out_ptr.unsafe_offset(h * head_dim + d).unsafe_store(val=out_vec.get(d))


def mha_forward_threaded(
    q: Tensor[DType.float16, 3],
    cache: KVCacheLayer,
    position: Int,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    scale: Float32,
    nthreads: Int,
) -> Tensor[DType.float16, 3]:
    """Threaded MHA forward for decode (single query token).

    Splits heads across threads for parallel processing.
    Falls back to single-threaded when:
    - n_heads < MIN_HEADS_FOR_THREADING
    - Thread pool not available
    - nthreads == 1
    """
    # Check if threading is beneficial
    if n_heads < MIN_HEADS_FOR_THREADING or nthreads == 1:
        return _mha_forward_single(q, cache, position, n_heads, n_kv_heads, head_dim, scale)

    # Check if worker is available
    if not has_worker("mha_head_worker_v2"):
        return _mha_forward_single(q, cache, position, n_heads, n_kv_heads, head_dim, scale)

    var threads = resolve_threads(nthreads)
    if threads <= 1:
        return _mha_forward_single(q, cache, position, n_heads, n_kv_heads, head_dim, scale)

    # Prepare output
    var out = tensor_zeros[DType.float16, 3](
        StaticTuple[Int, 3](n_heads, 1, head_dim)
    )

    # Store cache on heap for worker access
    # Use copy to avoid transfer issues (shallow copy of tensor metadata)
    var cache_ptr = unsafe_alloc[KVCacheLayer](1)
    cache_ptr[unsafe_offset=0] = cache.copy()

    # Initialize global context
    var g_ctx = _init_global_ctx(
        q, out, cache_ptr, position, head_dim, scale, n_kv_heads, n_heads
    )

    # Calculate number of work items
    var n_work_items = (n_heads + HEADS_PER_WORKER - 1) // HEADS_PER_WORKER

    # Run parallel
    _ = parallel_run(
        "mha_head_worker_v2",
        g_ctx.unsafe_bitcast[UInt8](),
        n_work_items,
        threads,
    )

    # parallel_run is blocking, so all work is done here

    return out


def _mha_forward_single(
    q: Tensor[DType.float16, 3],
    cache: KVCacheLayer,
    position: Int,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    scale: Float32,
) -> Tensor[DType.float16, 3]:
    """Single-threaded fallback (original implementation)."""
    var out = tensor_zeros[DType.float16, 3](
        StaticTuple[Int, 3](n_heads, 1, head_dim)
    )

    for h in range(n_heads):
        var kv_head = h * n_kv_heads // n_heads

        # Extract Q vector
        var q_vec = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
        for d in range(head_dim):
            q_vec.set(d, q.get(h * head_dim + d))

        # Run flash attention
        var out_vec = flash_attention_decode(
            q_vec, cache, kv_head, position, head_dim, scale
        )

        # Store result
        for d in range(head_dim):
            out.set(h * head_dim + d, out_vec.get(d))

    return out

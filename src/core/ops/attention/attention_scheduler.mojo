# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/attention/attention_scheduler.mojo
#
# Multi-request attention scheduler.
# Supports parallel processing of multiple requests in a single batch.
#
# Design:
# - Each request is a work unit (request-level parallelism)
# - Within each request, heads are processed sequentially
# - Thread pool processes multiple requests in parallel
# - No shared writes: each request writes to its own output tensor

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.cpu.flash_attention_cpu import flash_attention_decode
from src.core.ops.attention.kv_cache import KVCacheLayer
from src.core.thread_pool import parallel_run, resolve_threads, has_worker
from std.utils import StaticTuple
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc
from std.math import sqrt

# Minimum requests for threading (below this, overhead dominates)
comptime MIN_REQUESTS_FOR_THREADING = 2


# Batch request context (primitive types only for easy passing)
struct BatchRequestContext:
    var req_id: Int
    var q_addr: Int64
    var out_addr: Int64
    var cache_addr: Int64
    var position: Int
    var n_heads: Int
    var n_kv_heads: Int
    var head_dim: Int
    var scale: Float32


# Batch context for worker
struct BatchWorkerContext:
    var requests_addr: Int64
    var n_requests: Int
    var head_dim: Int
    var scale: Float32


@export
def batch_attention_worker(
    ctx: Pointer[UInt8, MutUntrackedOrigin],
    idx: Int64,
) abi("C"):
    """Worker function for multi-request batch processing.

    Each invocation processes one request (all heads sequentially).
    """
    # Cast context
    var bctx_ptr = ctx.unsafe_bitcast[BatchWorkerContext]()

    # Get request array
    var requests_ptr = Pointer[BatchRequestContext, MutUntrackedOrigin](
        unsafe_from_address=Int(bctx_ptr[unsafe_offset=0].requests_addr)
    )

    # Process the request at index idx
    var req_idx = Int(idx)
    var n_requests = bctx_ptr[unsafe_offset=0].n_requests
    if req_idx >= n_requests:
        return

    # Access request fields directly through offset
    var req_ptr = requests_ptr.unsafe_offset(req_idx)
    var n_heads = req_ptr[unsafe_offset=0].n_heads
    var n_kv_heads = req_ptr[unsafe_offset=0].n_kv_heads
    var head_dim = req_ptr[unsafe_offset=0].head_dim
    var scale = req_ptr[unsafe_offset=0].scale
    var position = req_ptr[unsafe_offset=0].position

    # Reconstruct pointers
    var q_ptr = Pointer[Scalar[DType.float16], MutUntrackedOrigin](
        unsafe_from_address=Int(req_ptr[unsafe_offset=0].q_addr)
    )
    var out_ptr = Pointer[Scalar[DType.float16], MutUntrackedOrigin](
        unsafe_from_address=Int(req_ptr[unsafe_offset=0].out_addr)
    )
    var cache_ptr = Pointer[KVCacheLayer, MutUntrackedOrigin](
        unsafe_from_address=Int(req_ptr[unsafe_offset=0].cache_addr)
    )

    # Get cache (copy metadata, data pointers remain shared)
    var cache = cache_ptr[unsafe_offset=0].copy()

    # Process all heads for this request
    for h in range(n_heads):
        var kv_head = h * n_kv_heads // n_heads

        # Extract Q vector
        var q_vec = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
        for d in range(head_dim):
            q_vec.set(d, q_ptr.unsafe_offset(h * head_dim + d).unsafe_load())

        # Flash attention
        var out_vec = flash_attention_decode(
            q_vec, cache, kv_head, position, head_dim, scale
        )

        # Store result
        for d in range(head_dim):
            out_ptr.unsafe_offset(h * head_dim + d).unsafe_store(
                val=out_vec.get(d)
            )


def mha_forward_batch(
    queries: List[Tensor[DType.float16, 3]],  # List of [n_heads, 1, head_dim]
    caches: List[KVCacheLayer],
    positions: List[Int],
    n_heads_list: List[Int],
    n_kv_heads: Int,
    head_dim: Int,
    nthreads: Int,
) -> List[Tensor[DType.float16, 3]]:
    """Multi-request batch attention.

    Processes multiple requests in parallel using thread pool.
    Each request is independent (different KV cache, different query).

    Returns list of output tensors [n_heads, 1, head_dim] for each request.
    """
    var n_requests = len(queries)

    # Threading threshold
    if n_requests < MIN_REQUESTS_FOR_THREADING or nthreads == 1:
        # Fall back to sequential processing
        return _mha_forward_batch_sequential(
            queries, caches, positions, n_heads_list, n_kv_heads, head_dim
        )

    # Check worker availability
    if not has_worker("batch_attention_worker"):
        return _mha_forward_batch_sequential(
            queries, caches, positions, n_heads_list, n_kv_heads, head_dim
        )

    var threads = resolve_threads(nthreads)
    if threads <= 1:
        return _mha_forward_batch_sequential(
            queries, caches, positions, n_heads_list, n_kv_heads, head_dim
        )

    # Prepare outputs
    var outputs = List[Tensor[DType.float16, 3]]()
    for i in range(n_requests):
        var n_heads = n_heads_list[i]
        outputs.append(
            tensor_zeros[DType.float16, 3](
                StaticTuple[Int, 3](n_heads, 1, head_dim)
            )
        )

    # Prepare request contexts
    var req_contexts = unsafe_alloc[BatchRequestContext](n_requests)
    for i in range(n_requests):
        var ctx_ptr = req_contexts.unsafe_offset(i)
        ctx_ptr[unsafe_offset=0].req_id = i
        ctx_ptr[unsafe_offset=0].q_addr = Int64(Int(queries[i].data()))
        ctx_ptr[unsafe_offset=0].out_addr = Int64(Int(outputs[i].data()))
        # Store cache on heap
        var cache_heap = unsafe_alloc[KVCacheLayer](1)
        cache_heap[unsafe_offset=0] = caches[i].copy()
        ctx_ptr[unsafe_offset=0].cache_addr = Int64(Int(cache_heap))
        ctx_ptr[unsafe_offset=0].position = positions[i]
        ctx_ptr[unsafe_offset=0].n_heads = n_heads_list[i]
        ctx_ptr[unsafe_offset=0].n_kv_heads = n_kv_heads
        ctx_ptr[unsafe_offset=0].head_dim = head_dim
        ctx_ptr[unsafe_offset=0].scale = Float32(1.0) / sqrt(Float32(head_dim))

    # Prepare batch context
    var bctx_ptr = unsafe_alloc[BatchWorkerContext](1)
    bctx_ptr[unsafe_offset=0].requests_addr = Int64(Int(req_contexts))
    bctx_ptr[unsafe_offset=0].n_requests = n_requests
    bctx_ptr[unsafe_offset=0].head_dim = head_dim
    bctx_ptr[unsafe_offset=0].scale = Float32(1.0) / sqrt(Float32(head_dim))

    # Run parallel
    _ = parallel_run(
        "batch_attention_worker",
        bctx_ptr.unsafe_bitcast[UInt8](),
        n_requests,  # Each request is one work item
        threads,
    )

    return outputs^


def _mha_forward_batch_sequential(
    queries: List[Tensor[DType.float16, 3]],
    caches: List[KVCacheLayer],
    positions: List[Int],
    n_heads_list: List[Int],
    n_kv_heads: Int,
    head_dim: Int,
) -> List[Tensor[DType.float16, 3]]:
    """Sequential fallback for batch attention."""
    var outputs = List[Tensor[DType.float16, 3]]()
    var scale = Float32(1.0) / sqrt(Float32(head_dim))

    for i in range(len(queries)):
        var n_heads = n_heads_list[i]
        var out = tensor_zeros[DType.float16, 3](
            StaticTuple[Int, 3](n_heads, 1, head_dim)
        )

        for h in range(n_heads):
            var kv_head = h * n_kv_heads // n_heads

            var q_vec = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
            for d in range(head_dim):
                q_vec.set(d, queries[i].get(h * head_dim + d))

            var out_vec = flash_attention_decode(
                q_vec, caches[i], kv_head, positions[i], head_dim, scale
            )

            for d in range(head_dim):
                out.set(h * head_dim + d, out_vec.get(d))

        outputs.append(out)

    return outputs^

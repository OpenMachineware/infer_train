# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/gpu/attention_decode_gpu.mojo
#
# GPU attention for decode mode (single token generation).
# Q has shape [n_heads, 1, head_dim], K/V have KV cache length.

from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.sync import barrier
from std.gpu import WARP_SIZE, block_idx, thread_idx, lane_id
from std.memory import Pointer, unsafe_stack_allocation
from std.origin import MutAnyOrigin
from std.math import min as math_min, exp, sqrt
from src.core.tensor import Tensor
from std.utils.static_tuple import StaticTuple
from std.collections.optional import Optional

comptime BLOCK_THREADS = 256


def decode_attention_qk_kernel(
    q: Pointer[Scalar[DType.float16], MutAnyOrigin],  # [n_heads, 1, head_dim]
    k: Pointer[Scalar[DType.float16], MutAnyOrigin],  # [n_heads, kv_len, head_dim]
    scores: Pointer[Scalar[DType.float32], MutAnyOrigin],  # [n_heads, kv_len]
    n_heads: Int32,
    head_dim: Int32,
    kv_len: Int32,
    scale: Float32,
):
    """Compute Q @ K^T for decode (Q has 1 row per head).

    Grid: (n_heads,)
    Each threadgroup computes scores for one head.
    Each thread processes a subset of K positions.
    """
    var head = Int(block_idx.x)
    var n_heads_i = Int(n_heads)
    var head_dim_i = Int(head_dim)
    var kv_len_i = Int(kv_len)

    if head >= n_heads_i:
        return

    var tid = Int(thread_idx.x)

    # Q pointer for this head (single row)
    var q_base = head * head_dim_i
    var k_base = head * kv_len_i * head_dim_i
    var score_base = head * kv_len_i

    # Load Q into threadgroup memory (shared by all threads)
    var q_tile = unsafe_stack_allocation[
        128, DType.float16, address_space=AddressSpace.SHARED  # Max head_dim
    ]()

    for d in range(tid, head_dim_i, BLOCK_THREADS):
        q_tile[unsafe_offset=d] = q[unsafe_offset=q_base + d]

    barrier()

    # Compute Q @ K^T for all K positions
    for k_pos in range(tid, kv_len_i, BLOCK_THREADS):
        var dot = Float32(0.0)
        for d in range(head_dim_i):
            var q_val = Float32(q_tile[unsafe_offset=d])
            var k_val = Float32(k[unsafe_offset=k_base + k_pos * head_dim_i + d])
            dot += q_val * k_val
        scores[unsafe_offset=score_base + k_pos] = dot * scale


def decode_softmax_kernel(
    scores: Pointer[Scalar[DType.float32], MutAnyOrigin],  # [n_heads, kv_len]
    n_heads: Int32,
    kv_len: Int32,
):
    """Apply softmax to decode scores (in-place).

    Grid: (n_heads,)
    Each threadgroup processes one row.
    """
    var head = Int(block_idx.x)
    var n_heads_i = Int(n_heads)
    var kv_len_i = Int(kv_len)

    if head >= n_heads_i:
        return

    var tid = Int(thread_idx.x)
    var score_base = head * kv_len_i

    # Threadgroup memory for reduction
    var max_vals = unsafe_stack_allocation[
        BLOCK_THREADS, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var sum_vals = unsafe_stack_allocation[
        BLOCK_THREADS, DType.float32, address_space=AddressSpace.SHARED
    ]()

    # Find max
    var local_max = Float32(-1e30)
    for k in range(tid, kv_len_i, BLOCK_THREADS):
        var s = Float32(scores[unsafe_offset=score_base + k])
        if s > local_max:
            local_max = s

    max_vals[unsafe_offset=tid] = local_max
    barrier()

    if tid == 0:
        var global_max = Float32(-1e30)
        for i in range(BLOCK_THREADS):
            if max_vals[unsafe_offset=i] > global_max:
                global_max = max_vals[unsafe_offset=i]
        max_vals[unsafe_offset=0] = global_max

    barrier()
    var row_max = max_vals[unsafe_offset=0]

    # Compute exp and sum
    var local_sum = Float32(0.0)
    for k in range(tid, kv_len_i, BLOCK_THREADS):
        var s = Float32(scores[unsafe_offset=score_base + k])
        var e = exp(s - row_max)
        scores[unsafe_offset=score_base + k] = e
        local_sum += e

    sum_vals[unsafe_offset=tid] = local_sum
    barrier()

    if tid == 0:
        var global_sum = Float32(0.0)
        for i in range(BLOCK_THREADS):
            global_sum += sum_vals[unsafe_offset=i]
        sum_vals[unsafe_offset=0] = global_sum

    barrier()
    var row_sum = sum_vals[unsafe_offset=0]

    # Normalize
    for k in range(tid, kv_len_i, BLOCK_THREADS):
        var e = Float32(scores[unsafe_offset=score_base + k])
        scores[unsafe_offset=score_base + k] = e / row_sum


def decode_attention_sv_kernel(
    scores: Pointer[Scalar[DType.float32], MutAnyOrigin],  # [n_heads, kv_len]
    v: Pointer[Scalar[DType.float16], MutAnyOrigin],  # [n_heads, kv_len, head_dim]
    output: Pointer[Scalar[DType.float16], MutAnyOrigin],  # [n_heads, head_dim]
    n_heads: Int32,
    head_dim: Int32,
    kv_len: Int32,
):
    """Compute Scores @ V for decode (single output row per head).

    Grid: (n_heads,)
    Each threadgroup computes one output row.
    Each thread computes a subset of output elements.
    """
    var head = Int(block_idx.x)
    var n_heads_i = Int(n_heads)
    var head_dim_i = Int(head_dim)
    var kv_len_i = Int(kv_len)

    if head >= n_heads_i:
        return

    var tid = Int(thread_idx.x)
    var score_base = head * kv_len_i
    var v_base = head * kv_len_i * head_dim_i
    var out_base = head * head_dim_i

    # Threadgroup memory for scores
    var score_tile = unsafe_stack_allocation[
        2048, DType.float32, address_space=AddressSpace.SHARED  # Max kv_len
    ]()

    # Load scores into threadgroup memory
    for k in range(tid, kv_len_i, BLOCK_THREADS):
        score_tile[unsafe_offset=k] = scores[unsafe_offset=score_base + k]

    barrier()

    # Compute Scores @ V
    for d in range(tid, head_dim_i, BLOCK_THREADS):
        var acc = Float32(0.0)
        for k in range(kv_len_i):
            var score = score_tile[unsafe_offset=k]
            var v_val = Float32(v[unsafe_offset=v_base + k * head_dim_i + d])
            acc += score * v_val
        output[unsafe_offset=out_base + d] = Scalar[DType.float16](acc)


def gpu_attention_decode(
    ctx: DeviceContext,
    q: Tensor[DType.float16, 3],  # [n_heads, 1, head_dim]
    k: Tensor[DType.float16, 3],  # [n_heads, kv_len, head_dim]
    v: Tensor[DType.float16, 3],  # [n_heads, kv_len, head_dim]
) raises -> Tensor[DType.float16, 3]:
    """GPU attention for decode (single token generation).

    Q has 1 row per head, K/V have KV cache length.

    Args:
        ctx: GPU device context
        q: Query tensor [n_heads, 1, head_dim]
        k: Key tensor [n_heads, kv_len, head_dim]
        v: Value tensor [n_heads, kv_len, head_dim]

    Returns:
        Output tensor [n_heads, 1, head_dim]
    """
    var n_heads = q.shape()[0]
    var kv_len = k.shape()[1]
    var head_dim = q.shape()[2]
    var scale = 1.0 / sqrt(Float32(head_dim))

    # Allocate GPU buffers
    var q_buf = ctx.enqueue_create_buffer[DType.float16](n_heads * head_dim)
    var k_buf = ctx.enqueue_create_buffer[DType.float16](n_heads * kv_len * head_dim)
    var v_buf = ctx.enqueue_create_buffer[DType.float16](n_heads * kv_len * head_dim)
    var scores_buf = ctx.enqueue_create_buffer[DType.float32](n_heads * kv_len)
    var output_buf = ctx.enqueue_create_buffer[DType.float16](n_heads * head_dim)

    # Create host buffers and copy data
    var q_host = ctx.enqueue_create_host_buffer[DType.float16](n_heads * head_dim)
    var k_host = ctx.enqueue_create_host_buffer[DType.float16](n_heads * kv_len * head_dim)
    var v_host = ctx.enqueue_create_host_buffer[DType.float16](n_heads * kv_len * head_dim)

    for i in range(n_heads * head_dim):
        q_host[i] = q._data[unsafe_offset=i]

    for i in range(n_heads * kv_len * head_dim):
        k_host[i] = k._data[unsafe_offset=i]
        v_host[i] = v._data[unsafe_offset=i]

    # Upload to GPU
    ctx.enqueue_copy(q_buf, q_host)
    ctx.enqueue_copy(k_buf, k_host)
    ctx.enqueue_copy(v_buf, v_host)

    # Step 1: Q @ K^T
    ctx.enqueue_function[decode_attention_qk_kernel](
        q_buf, k_buf, scores_buf,
        Int32(n_heads), Int32(head_dim), Int32(kv_len), scale,
        grid_dim=(n_heads,),
        block_dim=BLOCK_THREADS
    )

    # Step 2: Softmax (in-place)
    ctx.enqueue_function[decode_softmax_kernel](
        scores_buf,
        Int32(n_heads), Int32(kv_len),
        grid_dim=(n_heads,),
        block_dim=BLOCK_THREADS
    )

    # Step 3: Scores @ V
    ctx.enqueue_function[decode_attention_sv_kernel](
        scores_buf, v_buf, output_buf,
        Int32(n_heads), Int32(head_dim), Int32(kv_len),
        grid_dim=(n_heads,),
        block_dim=BLOCK_THREADS
    )

    # Download result
    var out_host = ctx.enqueue_create_host_buffer[DType.float16](n_heads * head_dim)
    ctx.enqueue_copy(out_host, output_buf)
    ctx.synchronize()

    # Create output tensor
    var result = Tensor[DType.float16, 3](StaticTuple[Int, 3](n_heads, 1, head_dim))
    for i in range(n_heads * head_dim):
        result._data[unsafe_offset=i] = out_host[i]

    return result


def gpu_attention_decode_cached(
    ctx: DeviceContext,
    q: Tensor[DType.float16, 3],
    k: Tensor[DType.float16, 3],
    v: Tensor[DType.float16, 3],
    q_buf: Optional[DeviceBuffer[DType.float16]] = None,
    k_buf: Optional[DeviceBuffer[DType.float16]] = None,
    v_buf: Optional[DeviceBuffer[DType.float16]] = None,
) raises -> Tensor[DType.float16, 3]:
    """GPU decode attention with optional cached buffers.

    Use this when Q, K, V are already on GPU from previous operations.
    """
    var n_heads = q.shape()[0]
    var kv_len = k.shape()[1]
    var head_dim = q.shape()[2]
    var scale = 1.0 / sqrt(Float32(head_dim))

    # Use provided buffers or create new ones
    var q_buf_actual: DeviceBuffer[DType.float16]
    var k_buf_actual: DeviceBuffer[DType.float16]
    var v_buf_actual: DeviceBuffer[DType.float16]

    if q_buf:
        q_buf_actual = q_buf.value()
    else:
        var q_host = ctx.enqueue_create_host_buffer[DType.float16](n_heads * head_dim)
        for i in range(n_heads * head_dim):
            q_host[i] = q._data[unsafe_offset=i]
        q_buf_actual = ctx.enqueue_create_buffer[DType.float16](n_heads * head_dim)
        ctx.enqueue_copy(q_buf_actual, q_host)

    if k_buf:
        k_buf_actual = k_buf.value()
    else:
        var k_host = ctx.enqueue_create_host_buffer[DType.float16](n_heads * kv_len * head_dim)
        for i in range(n_heads * kv_len * head_dim):
            k_host[i] = k._data[unsafe_offset=i]
        k_buf_actual = ctx.enqueue_create_buffer[DType.float16](n_heads * kv_len * head_dim)
        ctx.enqueue_copy(k_buf_actual, k_host)

    if v_buf:
        v_buf_actual = v_buf.value()
    else:
        var v_host = ctx.enqueue_create_host_buffer[DType.float16](n_heads * kv_len * head_dim)
        for i in range(n_heads * kv_len * head_dim):
            v_host[i] = v._data[unsafe_offset=i]
        v_buf_actual = ctx.enqueue_create_buffer[DType.float16](n_heads * kv_len * head_dim)
        ctx.enqueue_copy(v_buf_actual, v_host)

    # Allocate output buffers
    var scores_buf = ctx.enqueue_create_buffer[DType.float32](n_heads * kv_len)
    var output_buf = ctx.enqueue_create_buffer[DType.float16](n_heads * head_dim)

    # Run attention kernels
    ctx.enqueue_function[decode_attention_qk_kernel](
        q_buf_actual, k_buf_actual, scores_buf,
        Int32(n_heads), Int32(head_dim), Int32(kv_len), scale,
        grid_dim=(n_heads,),
        block_dim=BLOCK_THREADS
    )

    ctx.enqueue_function[decode_softmax_kernel](
        scores_buf,
        Int32(n_heads), Int32(kv_len),
        grid_dim=(n_heads,),
        block_dim=BLOCK_THREADS
    )

    ctx.enqueue_function[decode_attention_sv_kernel](
        scores_buf, v_buf_actual, output_buf,
        Int32(n_heads), Int32(head_dim), Int32(kv_len),
        grid_dim=(n_heads,),
        block_dim=BLOCK_THREADS
    )

    # Download result
    var out_host = ctx.enqueue_create_host_buffer[DType.float16](n_heads * head_dim)
    ctx.enqueue_copy(out_host, output_buf)
    ctx.synchronize()

    var result = Tensor[DType.float16, 3](StaticTuple[Int, 3](n_heads, 1, head_dim))
    for i in range(n_heads * head_dim):
        result._data[unsafe_offset=i] = out_host[i]

    return result

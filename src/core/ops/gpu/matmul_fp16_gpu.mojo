# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/gpu/matmul_fp16_gpu.mojo
#
# GPU matrix multiplication for FP16 weights using tensor core.
# This is used for decode mode after pre-dequantization from Q4_K.

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.gpu.gpu_runtime import (
    download2,
    get_gpu_context,
    gpu_available,
    upload,
)
from src.core.ops.cpu.matmul_cpu import matmul_weight_cpu_threaded
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx
from std.memory import Pointer, unsafe_stack_allocation
from std.origin import MutAnyOrigin
from std.utils.static_tuple import StaticTuple
from std.collections.optional import Optional

# Tile dimensions for tensor core
comptime TILE_M = 16
comptime TILE_N = 16
comptime TILE_K = 16
comptime BLOCK_THREADS = 256

# Optimized tile for M=1 (decode mode)
comptime DECODE_TILE_N = 256
comptime DECODE_TILE_K = 64


def kernel_matmul_fp16_decode(
    A: Pointer[Scalar[DType.float16], MutAnyOrigin],  # [1, K]
    B: Pointer[Scalar[DType.float16], MutAnyOrigin],  # [N, K] (transposed)
    C: Pointer[Scalar[DType.float16], MutAnyOrigin],  # [1, N]
    K: Int32,
    N: Int32,
):
    """Optimized FP16 matmul for M=1 (decode mode).

    Uses shared memory to cache input vector A, reducing global memory accesses.
    Each thread block processes a tile of output elements.
    """
    var tx = Int(thread_idx.x)
    var bx = Int(block_idx.x)

    # Shared memory for input vector A (cached once per block)
    var A_shared = unsafe_stack_allocation[1024, DType.float16]()

    # Load A into shared memory (all threads participate)
    var k = tx
    while k < 1024:
        if k < Int(K):
            A_shared[unsafe_offset=k] = A.unsafe_load[width=1](offset=k)
        else:
            A_shared[unsafe_offset=k] = Scalar[DType.float16](0)
        k += BLOCK_THREADS

    barrier()

    # Each thread computes 8 output elements
    var elems_per_thread = 8
    var base_col = (bx * BLOCK_THREADS + tx) * elems_per_thread

    # Compute dot products for multiple output elements
    var sums = StaticTuple[Float32, 8](0, 0, 0, 0, 0, 0, 0, 0)

    # Vectorized loop over K dimension using shared memory
    k = 0
    while k + 8 <= Int(K):
        # Load from shared memory (much faster than global memory)
        var a_vec = A_shared.unsafe_load[width=8](offset=k)

        # Compute for each output element
        for e in range(elems_per_thread):
            var out_col = base_col + e
            if out_col < Int(N):
                var b_vec = B.unsafe_load[width=8](offset=out_col * Int(K) + k)

                # Accumulate
                for i in range(8):
                    sums[e] += Float32(a_vec[i]) * Float32(b_vec[i])

        k += 8

    # Handle remaining elements
    while k < Int(K):
        var a_val = Float32(A_shared.unsafe_load[width=1](offset=k))

        for e in range(elems_per_thread):
            var out_col = base_col + e
            if out_col < Int(N):
                var b_val = Float32(B.unsafe_load[width=1](offset=out_col * Int(K) + k))
                sums[e] += a_val * b_val

        k += 1

    # Write outputs
    for e in range(elems_per_thread):
        var out_col = base_col + e
        if out_col < Int(N):
            C.unsafe_offset(out_col).unsafe_store(val=Scalar[DType.float16](sums[e]))


def kernel_matmul_fp16_tiled(
    A: Pointer[Scalar[DType.float16], MutAnyOrigin],  # [M, K]
    B: Pointer[Scalar[DType.float16], MutAnyOrigin],  # [K, N]
    C: Pointer[Scalar[DType.float16], MutAnyOrigin],  # [M, N]
    M: Int32,
    N: Int32,
    K: Int32,
):
    """FP16 matrix multiplication using tiled approach.

    Each thread block computes a TILE_M x TILE_N output tile.
    Uses shared memory to cache input tiles.
    """
    var bx = Int(block_idx.x)
    var by = Int(block_idx.y)
    var tx = Int(thread_idx.x)

    # Output tile position
    var tile_row = by * TILE_M
    var tile_col = bx * TILE_N

    # Shared memory for tiles
    var A_tile = unsafe_stack_allocation[TILE_M * TILE_K, DType.float16]()
    var B_tile = unsafe_stack_allocation[TILE_K * TILE_N, DType.float16]()

    # Each thread computes multiple output elements
    var num_threads = BLOCK_THREADS
    var elems_per_thread = (TILE_M * TILE_N + num_threads - 1) // num_threads

    # Accumulator for output elements
    var acc = unsafe_stack_allocation[TILE_M * TILE_N, DType.float32]()
    for i in range(TILE_M * TILE_N):
        acc[unsafe_offset=i] = Float32(0)

    # Loop over K dimension in tiles
    for k_tile in range(0, Int(K), TILE_K):
        # Load A tile to shared memory
        var a_elems = TILE_M * TILE_K
        for i in range(tx, a_elems, num_threads):
            var row = i // TILE_K
            var col = i % TILE_K
            var global_row = tile_row + row
            var global_col = k_tile + col
            if global_row < Int(M) and global_col < Int(K):
                A_tile[unsafe_offset=i] = A.unsafe_load[width=1](offset=global_row * Int(K) + global_col)
            else:
                A_tile[unsafe_offset=i] = Scalar[DType.float16](0)

        # Load B tile to shared memory
        var b_elems = TILE_K * TILE_N
        for i in range(tx, b_elems, num_threads):
            var row = i // TILE_N
            var col = i % TILE_N
            var global_row = k_tile + row
            var global_col = tile_col + col
            if global_row < Int(K) and global_col < Int(N):
                B_tile[unsafe_offset=i] = B.unsafe_load[width=1](offset=global_row * Int(N) + global_col)
            else:
                B_tile[unsafe_offset=i] = Scalar[DType.float16](0)

        barrier()

        # Compute partial products
        for e in range(elems_per_thread):
            var elem_idx = tx + e * num_threads
            if elem_idx < TILE_M * TILE_N:
                var out_row = elem_idx // TILE_N
                var out_col = elem_idx % TILE_N
                var sum = Float32(0)
                for k in range(TILE_K):
                    var a_val = Float32(A_tile[unsafe_offset=out_row * TILE_K + k])
                    var b_val = Float32(B_tile[unsafe_offset=k * TILE_N + out_col])
                    sum += a_val * b_val
                acc[unsafe_offset=elem_idx] += sum

        barrier()

    # Write output
    for e in range(elems_per_thread):
        var elem_idx = tx + e * num_threads
        if elem_idx < TILE_M * TILE_N:
            var out_row = elem_idx // TILE_N
            var out_col = elem_idx % TILE_N
            var global_row = tile_row + out_row
            var global_col = tile_col + out_col
            if global_row < Int(M) and global_col < Int(N):
                C.unsafe_offset(global_row * Int(N) + global_col).unsafe_store(
                    val=Scalar[DType.float16](acc[unsafe_offset=elem_idx])
                )


def matmul_fp16_gpu(
    x: Tensor[DType.float16, 2],  # [M, K]
    w: Tensor[DType.float16, 2],  # [N, K] (transposed)
    w_buf_cached: Optional[DeviceBuffer[DType.float16]] = None,
    ctx_cached: Optional[DeviceContext] = None,
    x_buf_cached: Optional[DeviceBuffer[DType.float16]] = None,  # NEW: GPU buffer for input
    keep_output_on_gpu: Bool = False,  # NEW: Keep result on GPU
) -> Tensor[DType.float16, 2]:
    """FP16 matrix multiplication on GPU.

    Args:
        x: Input tensor [M, K]
        w: Weight tensor [N, K] (transposed, so we compute x @ w^T)
        w_buf_cached: Optional pre-uploaded GPU buffer for weights
        ctx_cached: Optional cached DeviceContext
        x_buf_cached: Optional pre-uploaded GPU buffer for input (avoids upload)
        keep_output_on_gpu: Keep result in GPU memory, returns dummy tensor

    Returns:
        Output tensor [M, N]. If keep_output_on_gpu=True, returns empty tensor
        (caller must use the GPU buffer directly).
    """
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]

    if not gpu_available[DType.float16]():
        # Fallback to CPU
        return matmul_weight_cpu_threaded[DType.float16](x, w)

    try:
        var ctx: DeviceContext
        var owns_ctx = False
        if ctx_cached:
            ctx = ctx_cached.value()
        else:
            ctx = get_gpu_context()
            owns_ctx = True

        # Use cached input buffer if available, otherwise upload
        var x_buf: DeviceBuffer[DType.float16]
        if x_buf_cached:
            x_buf = x_buf_cached.value()
        else:
            x_buf = upload[DType.float16, 2](ctx, x)

        # Use cached buffer if available
        var w_buf: DeviceBuffer[DType.float16]
        if w_buf_cached:
            w_buf = w_buf_cached.value()
        else:
            w_buf = upload[DType.float16, 2](ctx, w)

        # Allocate output buffer
        var out_buf = ctx.enqueue_create_buffer[DType.float16](M * N)

        # Choose kernel based on M
        if M == 1:
            # Use optimized decode kernel for M=1
            # Each thread processes 4 elements
            var elems_per_thread = 4
            var threads_needed = (N + elems_per_thread - 1) // elems_per_thread
            var grid_x = (threads_needed + BLOCK_THREADS - 1) // BLOCK_THREADS

            ctx.enqueue_function[kernel_matmul_fp16_decode](
                x_buf,
                w_buf,
                out_buf,
                Int32(K),
                Int32(N),
                grid_dim=(grid_x,),
                block_dim=BLOCK_THREADS,
            )
        else:
            # Use general tiled kernel for M > 1
            var grid_x = (N + TILE_N - 1) // TILE_N
            var grid_y = (M + TILE_M - 1) // TILE_M

            ctx.enqueue_function[kernel_matmul_fp16_tiled](
                x_buf,
                w_buf,
                out_buf,
                Int32(M),
                Int32(N),
                Int32(K),
                grid_dim=(grid_x, grid_y),
                block_dim=BLOCK_THREADS,
            )

        # Download result or keep on GPU
        if keep_output_on_gpu:
            # Return empty tensor, caller uses GPU buffer directly
            if owns_ctx:
                ctx.synchronize()
            return tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](0, 0))

        var out = download2[DType.float16](ctx, out_buf, StaticTuple[Int, 2](M, N))

        if owns_ctx:
            ctx.synchronize()

        return out

    except:
        # Fallback to CPU on error
        return matmul_weight_cpu_threaded[DType.float16](x, w)


# ============================================================================
# GPU Pipeline Version: Returns GPU buffer for chained operations
# ============================================================================


def matmul_fp16_gpu_pipeline(
    x_buf: DeviceBuffer[DType.float16],  # [M, K] on GPU
    w_buf: DeviceBuffer[DType.float16],  # [N, K] on GPU
    ctx: DeviceContext,
    M: Int,
    K: Int,
    N: Int,
) raises -> DeviceBuffer[DType.float16]:
    """FP16 matrix multiplication for GPU pipeline.

    Input and output stay on GPU, enabling zero-copy chaining.

    Args:
        x_buf: Input buffer on GPU [M, K]
        w_buf: Weight buffer on GPU [N, K]
        ctx: Device context
        M, K, N: Matrix dimensions

    Returns:
        Output buffer [M, N] on GPU (caller owns)
    """
    # Allocate output buffer
    var out_buf = ctx.enqueue_create_buffer[DType.float16](M * N)

    # Choose kernel based on M
    if M == 1:
        # Use optimized decode kernel for M=1
        var elems_per_thread = 4
        var threads_needed = (N + elems_per_thread - 1) // elems_per_thread
        var grid_x = (threads_needed + BLOCK_THREADS - 1) // BLOCK_THREADS

        ctx.enqueue_function[kernel_matmul_fp16_decode](
            x_buf,
            w_buf,
            out_buf,
            Int32(K),
            Int32(N),
            grid_dim=(grid_x,),
            block_dim=BLOCK_THREADS,
        )
    else:
        # Use general tiled kernel for M > 1
        var grid_x = (N + TILE_N - 1) // TILE_N
        var grid_y = (M + TILE_M - 1) // TILE_M

        ctx.enqueue_function[kernel_matmul_fp16_tiled](
            x_buf,
            w_buf,
            out_buf,
            Int32(M),
            Int32(N),
            Int32(K),
            grid_dim=(grid_x, grid_y),
            block_dim=BLOCK_THREADS,
        )

    return out_buf

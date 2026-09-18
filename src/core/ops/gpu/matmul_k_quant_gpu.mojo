# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/gpu/matmul_k_quant_gpu.mojo
#
# GPU matrix multiplication for K-series quantized weights with on-device dequantization.
# Supports Q2_K, Q3_K, Q4_K, Q5_K, Q6_K following llama.cpp's Metal implementation.

from src.core.tensor import Tensor
from src.core.ops.quantized.quant_types import QuantType
from src.core.cpu_features import detect_cpu_flags
from src.core.ops.gpu.gpu_runtime import (
    download2,
    get_gpu_context,
    gpu_available,
    upload,
)
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx
from std.memory import Pointer, unsafe_stack_allocation
from std.origin import MutAnyOrigin
from std.utils.static_tuple import StaticTuple
from std.math import min as math_min
from std.collections.optional import Optional

comptime QK_K = 256  # Elements per K-quant super-block

# Block sizes for each format (bytes)
comptime Q2_K_BLOCK = 84
comptime Q3_K_BLOCK = 110
comptime Q4_K_BLOCK = 144
comptime Q5_K_BLOCK = 176
comptime Q6_K_BLOCK = 210

# Tile dimensions
comptime NK = 64  # K tile size (4 chunks of 16 elements)
comptime NR0 = 32  # M tile size
comptime NR1 = 64  # N tile size
comptime BLOCK_THREADS = 256

# Each thread computes multiple outputs
comptime ROWS_PER_THREAD = 4
comptime COLS_PER_THREAD = 2


# -- Q4_K dequantization (144 bytes) -------------------------------------------


@always_inline
def _get_scale_min_k4_just2(
    j: Int, k: Int, scales: Pointer[UInt8, ...]
) -> Tuple[UInt8, UInt8]:
    """Extract scale and min from Q4_K/Q5_K scales array."""
    if j < 4:
        return (
            scales[unsafe_offset=j + k] & 63,
            scales[unsafe_offset=j + 4 + k] & 63,
        )
    else:
        var sc = UInt8(
            (scales[unsafe_offset=j + 4 + k] & 0xF)
            | ((scales[unsafe_offset=j - 4 + k] & 0xC0) >> 2)
        )
        var mn = UInt8(
            (scales[unsafe_offset=j + 4 + k] >> 4)
            | ((scales[unsafe_offset=j + k] & 0xC0) >> 2)
        )
        return (sc, mn)


@always_inline
def dequantize_q4_k_16(
    block_ptr: Pointer[UInt8, ...],
    il: Int,
    dst: Pointer[mut=True, Scalar[DType.float16], _, ...],
):
    """Dequantize 16 elements from a Q4_K block (il: 0-15)."""
    var d_raw = block_ptr.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(d_raw.unsafe_load[width=1](offset=0))
    var dmin = Float32(d_raw.unsafe_load[width=1](offset=1))
    var scales = block_ptr.unsafe_offset(4)
    var qs_base = block_ptr.unsafe_offset(16)

    var is_idx = (il // 4) * 2
    var q_offset = (il // 4) * 32 + 16 * (il & 1)
    var q = qs_base.unsafe_offset(q_offset)
    var il_inner = il & 3

    var (sc, mn) = _get_scale_min_k4_just2(is_idx, il_inner // 2, scales)

    var dl: Float32
    var mask: UInt8
    if il_inner < 2:
        dl = d * Float32(sc)
        mask = 0x0F
    else:
        dl = (d / 16.0) * Float32(sc)
        mask = 0xF0

    var ml = dmin * Float32(mn)

    for i in range(16):
        var byte_val = q.unsafe_load[width=1](offset=i)
        var nibble = Float32(byte_val & mask)
        dst[unsafe_offset=i] = Scalar[DType.float16](dl * nibble - ml)


# -- Q5_K dequantization (176 bytes) ------------------------------------------


@always_inline
def dequantize_q5_k_16(
    block_ptr: Pointer[UInt8, ...],
    il: Int,
    dst: Pointer[mut=True, Scalar[DType.float16], _, ...],
):
    """Dequantize 16 elements from a Q5_K block (il: 0-15).

    Q5_K layout (176 bytes):
    - d: fp16 scale (2 bytes)
    - dmin: fp16 min (2 bytes)
    - scales: 12 bytes
    - qh: 32 bytes (high bits)
    - qs: 128 bytes (4-bit quants)
    """
    var d_raw = block_ptr.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(d_raw.unsafe_load[width=1](offset=0))
    var dmin = Float32(d_raw.unsafe_load[width=1](offset=1))
    var scales = block_ptr.unsafe_offset(4)
    var qh = block_ptr.unsafe_offset(16)
    var qs_base = block_ptr.unsafe_offset(48)

    var is_idx = (il // 4) * 2
    var q_offset = 32 * (il // 4) + 16 * (il & 1)
    var qh_offset = 16 * (il & 1)
    var q = qs_base.unsafe_offset(q_offset)
    var qh_ptr = qh.unsafe_offset(qh_offset)

    var ul = UInt8(1 << (il // 2))  # Pre-compute the bit shift
    var il_inner = il & 3

    var (sc, mn) = _get_scale_min_k4_just2(is_idx, il_inner // 2, scales)

    var dl: Float32
    var mask: UInt8
    var qh_val: Float32
    if il_inner < 2:
        dl = d * Float32(sc)
        mask = 0x0F
        qh_val = 16.0
    else:
        dl = (d / 16.0) * Float32(sc)
        mask = 0xF0
        qh_val = 256.0

    var ml = dmin * Float32(mn)

    for i in range(16):
        var byte_val = q.unsafe_load[width=1](offset=i)
        var qh_byte = qh_ptr.unsafe_load[width=1](offset=i)
        var nibble = Float32(byte_val & mask)
        if (qh_byte & ul) != 0:
            nibble += qh_val
        dst[unsafe_offset=i] = Scalar[DType.float16](dl * nibble - ml)


# -- Q6_K dequantization (210 bytes) ------------------------------------------


@always_inline
def dequantize_q6_k_16(
    block_ptr: Pointer[UInt8, ...],
    il: Int,
    dst: Pointer[mut=True, Scalar[DType.float16], _, ...],
):
    """Dequantize 16 elements from a Q6_K block (il: 0-15).

    Q6_K layout (210 bytes):
    - ql: 128 bytes (low bits)
    - qh: 64 bytes (high bits)
    - scales: 16 bytes (int8)
    - d: fp16 scale (2 bytes) at offset 208

    Values centered at 32: q = (ql | (qh << 4)) - 32
    """
    var d_raw = block_ptr.unsafe_bitcast[Scalar[DType.float16]]()
    var d_all = Float32(d_raw.unsafe_load[width=1](offset=104))  # d at offset 208/2
    var ql_base = block_ptr.unsafe_offset(0)
    var qh_base = block_ptr.unsafe_offset(128)
    var scales = block_ptr.unsafe_offset(192)

    # Compute offsets based on il
    var ql_offset = 32 * (il // 8) + 16 * ((il // 2) & 1) + 8 * (il & 1)
    var qh_offset = 16 * (il // 8) + 8 * (il & 1)
    var sc_idx = (il % 2) + 2 * (il // 2)

    var ql_ptr = ql_base.unsafe_offset(ql_offset)
    var qh_ptr = qh_base.unsafe_offset(qh_offset)

    # Read scale as int8
    var sc_raw = scales[unsafe_offset=sc_idx]
    var sc = Float32(Int8(sc_raw))

    var il_inner = (il // 2) & 3

    # kmask values
    var kmask1: UInt32
    var kmask2: UInt32
    if il_inner > 1:
        kmask2 = 0xF0F0F0F0
        if il_inner > 2:
            kmask1 = 0xC0C0C0C0
        else:
            kmask1 = 0x30303030
    else:
        kmask2 = 0x0F0F0F0F
        if il_inner > 0:
            kmask1 = 0x0C0C0C0C
        else:
            kmask1 = 0x03030303

    var ml = d_all * sc * 32.0
    var dl0 = d_all * sc

    for i in range(4):
        # Read as UInt16 values
        var ql_ptr16 = ql_ptr.unsafe_bitcast[UInt16]()
        var qh_ptr16 = qh_ptr.unsafe_bitcast[UInt16]()

        var lo = UInt32(ql_ptr16.unsafe_load[width=1](offset=i * 2))
        var hi = UInt32(ql_ptr16.unsafe_load[width=1](offset=i * 2 + 1))
        var low = lo | (hi << 16)
        low = low & kmask2

        var qh_lo = UInt32(qh_ptr16.unsafe_load[width=1](offset=i * 2))
        var qh_hi = UInt32(qh_ptr16.unsafe_load[width=1](offset=i * 2 + 1))
        var high = qh_lo | (qh_hi << 16)
        high = high & kmask1

        var shl_h: UInt32
        var shr_h: UInt32
        var shr_l: UInt32
        if il_inner > 1:
            shr_l = 4
            if il_inner > 2:
                shr_h = 2
                shl_h = 0
            else:
                shr_h = 0
                shl_h = 0
        else:
            shr_l = 0
            if il_inner > 0:
                shl_h = 2
            else:
                shl_h = 4
            shr_h = 0

        var q = (high << shl_h) >> shr_h | (low >> shr_l)

        # Extract 4 bytes
        for j in range(4):
            var q_val = Float32((q >> UInt32(j * 8)) & 0xFF)
            dst[unsafe_offset=i * 4 + j] = Scalar[DType.float16](
                dl0 * q_val - ml
            )


# -- Q2_K dequantization (84 bytes) -------------------------------------------


@always_inline
def dequantize_q2_k_16(
    block_ptr: Pointer[UInt8, ...],
    il: Int,
    dst: Pointer[mut=True, Scalar[DType.float16], _, ...],
):
    """Dequantize 16 elements from a Q2_K block (il: 0-15).

    Q2_K layout (84 bytes):
    - d: fp16 scale (2 bytes)
    - dmin: fp16 min (2 bytes)
    - scales: 16 bytes
    - qs: 64 bytes (2-bit quants)
    """
    var d_raw = block_ptr.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(d_raw.unsafe_load[width=1](offset=0))
    var dmin = Float32(d_raw.unsafe_load[width=1](offset=1))
    var scales = block_ptr.unsafe_offset(4)
    var qs = block_ptr.unsafe_offset(20)

    var sc = scales[unsafe_offset=il]

    var q_offset = 32 * (il // 8) + 16 * (il & 1)
    var q = qs.unsafe_offset(q_offset)

    var il_inner = (il // 2) % 4

    var coef: Float32
    var mask: UInt8
    if il_inner > 1:
        if il_inner > 2:
            coef = 1.0 / 64.0
            mask = 192
        else:
            coef = 1.0 / 16.0
            mask = 48
    else:
        if il_inner > 0:
            coef = 1.0 / 4.0
            mask = 12
        else:
            coef = 1.0
            mask = 3

    var dl = d * Float32(sc & 0xF) * coef
    var ml = dmin * Float32(sc >> 4)

    for i in range(16):
        var byte_val = q.unsafe_load[width=1](offset=i)
        var qval = Float32(byte_val & mask)
        dst[unsafe_offset=i] = Scalar[DType.float16](dl * qval - ml)


# -- Q3_K dequantization (110 bytes) ------------------------------------------


@always_inline
def dequantize_q3_k_16(
    block_ptr: Pointer[UInt8, ...],
    il: Int,
    dst: Pointer[mut=True, Scalar[DType.float16], _, ...],
):
    """Dequantize 16 elements from a Q3_K block (il: 0-15).

    Q3_K layout (110 bytes):
    - hmask: 32 bytes
    - qs: 64 bytes (3-bit quants, uses hmask for bit 3)
    - scales: 12 bytes (int8)
    - d: fp16 scale (2 bytes)
    """
    var d_raw = block_ptr.unsafe_bitcast[Scalar[DType.float16]]()
    var d_all = Float32(d_raw.unsafe_load[width=1](offset=55))  # d at offset 110/2
    var hmask = block_ptr.unsafe_offset(0)
    var qs = block_ptr.unsafe_offset(32)
    var scales = block_ptr.unsafe_offset(96)

    var q_offset = 32 * (il // 8) + 16 * (il & 1)
    var h_offset = 16 * (il & 1)
    var q = qs.unsafe_offset(q_offset)
    var h = hmask.unsafe_offset(h_offset)

    var m = UInt8(1 << (il // 2))  # Pre-compute bit shift

    var kmask1: UInt16
    var kmask2: UInt16
    if (il // 4) > 1:
        if (il // 4) > 2:
            kmask1 = 192
        else:
            kmask1 = 48
    else:
        if (il // 4) > 0:
            kmask1 = 12
        else:
            kmask1 = 3

    if (il // 8) != 0:
        kmask2 = 0xF0
    else:
        kmask2 = 0x0F

    var scale_2_raw = scales[unsafe_offset=il % 8]
    var scale_1_raw = scales[unsafe_offset=8 + il % 4]
    var scale_2 = UInt16(scale_2_raw) & kmask2
    var scale_1 = UInt16(scale_1_raw) & kmask1

    var dl_int: Int16
    if (il // 4) & 1:
        dl_int = Int16(scale_2) | (Int16(scale_1) << 2)
    else:
        dl_int = Int16(scale_2) | (Int16(scale_1) << 4)

    var dl: Float32
    if il < 8:
        dl = d_all * (Float32(dl_int) - 32.0)
    else:
        dl = d_all * (Float32(dl_int) / 16.0 - 32.0)

    var ml = 4.0 * dl

    var il_inner = (il // 2) & 3
    var coef: Float32
    var mask: UInt8
    if il_inner > 1:
        if il_inner > 2:
            coef = 1.0 / 64.0
            mask = 192
        else:
            coef = 1.0 / 16.0
            mask = 48
    else:
        if il_inner > 0:
            coef = 1.0 / 4.0
            mask = 12
        else:
            coef = 1.0
            mask = 3

    dl = dl * coef

    for i in range(16):
        var byte_val = q.unsafe_load[width=1](offset=i)
        var h_byte = h.unsafe_load[width=1](offset=i)
        var qval = Float32(byte_val & mask)
        if (h_byte & m) == 0:
            qval += 4.0 * Float32(mask)
        dst[unsafe_offset=i] = Scalar[DType.float16](dl * qval - ml)


# -- Generic K-quant matmul kernel -------------------------------------------


def _matmul_k_quant_kernel[
    quant_type: QuantType
](
    x: Pointer[Scalar[DType.float16], MutAnyOrigin],
    w_quant: Pointer[UInt8, MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    M: Int32,
    K: Int32,
    N: Int32,
    n_blocks_per_row: Int32,
):
    """Generic K-quant GPU matmul kernel with on-device dequantization."""
    var M_i = Int(M)
    var K_i = Int(K)
    var N_i = Int(N)
    var n_blocks = Int(n_blocks_per_row)

    comptime block_bytes = _get_block_bytes(quant_type)

    var block_row = Int(block_idx.y) * NR0
    var block_col = Int(block_idx.x) * NR1
    var tid = Int(thread_idx.x)

    var thread_row_in_tile = tid // (NR1 // COLS_PER_THREAD)
    var thread_col_in_tile = tid % (NR1 // COLS_PER_THREAD)

    # Threadgroup memory
    var tile_w = unsafe_stack_allocation[
        NR1 * NK,
        DType.float16,
        address_space=AddressSpace.SHARED,
    ]()
    var tile_x = unsafe_stack_allocation[
        NR0 * NK,
        DType.float16,
        address_space=AddressSpace.SHARED,
    ]()

    var acc = SIMD[DType.float32, ROWS_PER_THREAD * COLS_PER_THREAD](0.0)

    var k_tiles = (K_i + NK - 1) // NK
    for k_tile in range(k_tiles):
        var k_start = k_tile * NK
        var k_end = math_min(k_start + NK, K_i)
        var k_size = k_end - k_start

        # Phase 1: Dequantize weight blocks
        comptime chunks_per_tile = NK // 16
        var total_chunks = NR1 * chunks_per_tile

        for work in range(tid, total_chunks, BLOCK_THREADS):
            var w_row = work // chunks_per_tile
            var chunk_in_tile = work % chunks_per_tile

            var global_col = block_col + w_row
            if global_col >= N_i:
                continue

            var k_base = k_start + chunk_in_tile * 16
            if k_base >= K_i:
                continue

            var qblock_idx = k_base // QK_K
            var il_base = k_base % QK_K
            var il = il_base // 16

            var block_ptr = w_quant.unsafe_offset(
                (global_col * n_blocks + qblock_idx) * block_bytes
            )

            var dst_offset = w_row * NK + chunk_in_tile * 16
            _dequantize_16[quant_type](
                block_ptr, il, tile_w.unsafe_offset(dst_offset)
            )

        # Phase 2: Load activation tile
        var total_x = NR0 * k_size
        for work in range(tid, total_x, BLOCK_THREADS):
            var x_row = work // k_size
            var x_k = work % k_size

            var global_row = block_row + x_row
            var global_k = k_start + x_k

            if global_row < M_i and global_k < K_i:
                tile_x[unsafe_offset=x_row * NK + x_k] = x[
                    unsafe_offset=global_row * K_i + global_k
                ]
            else:
                tile_x[unsafe_offset=x_row * NK + x_k] = Scalar[DType.float16](0.0)

        barrier()

        # Phase 3: Compute partial dot products
        for k in range(k_size):
            for r in range(ROWS_PER_THREAD):
                var row_in_tile = thread_row_in_tile * ROWS_PER_THREAD + r
                var x_val = Float32(tile_x[unsafe_offset=row_in_tile * NK + k])

                for c in range(COLS_PER_THREAD):
                    var col_in_tile = thread_col_in_tile * COLS_PER_THREAD + c
                    var w_val = Float32(tile_w[unsafe_offset=col_in_tile * NK + k])
                    acc[r * COLS_PER_THREAD + c] += x_val * w_val

        barrier()

    # Store results
    for r in range(ROWS_PER_THREAD):
        var row = block_row + thread_row_in_tile * ROWS_PER_THREAD + r
        for c in range(COLS_PER_THREAD):
            var col = block_col + thread_col_in_tile * COLS_PER_THREAD + c
            if row < M_i and col < N_i:
                dst[unsafe_offset=row * N_i + col] = Scalar[DType.float16](
                    acc[r * COLS_PER_THREAD + c]
                )


@always_inline
def _get_block_bytes(quant_type: QuantType) -> Int:
    """Get block size in bytes for each K-quant format."""
    if quant_type == QuantType.Q4_K_M:
        return Q4_K_BLOCK
    if quant_type == QuantType.Q5_K:
        return Q5_K_BLOCK
    if quant_type == QuantType.Q6_K:
        return Q6_K_BLOCK
    if quant_type == QuantType.Q2_K:
        return Q2_K_BLOCK
    if quant_type == QuantType.Q3_K:
        return Q3_K_BLOCK
    return 0


@always_inline
def _dequantize_16[
    quant_type: QuantType
](
    block_ptr: Pointer[UInt8, ...],
    il: Int,
    dst: Pointer[mut=True, Scalar[DType.float16], _, ...],
):
    """Dispatch to the correct dequantization function."""
    comptime if quant_type == QuantType.Q4_K_M:
        dequantize_q4_k_16(block_ptr, il, dst)
    elif quant_type == QuantType.Q5_K:
        dequantize_q5_k_16(block_ptr, il, dst)
    elif quant_type == QuantType.Q6_K:
        dequantize_q6_k_16(block_ptr, il, dst)
    elif quant_type == QuantType.Q2_K:
        dequantize_q2_k_16(block_ptr, il, dst)
    elif quant_type == QuantType.Q3_K:
        dequantize_q3_k_16(block_ptr, il, dst)


# -- Public entry points ------------------------------------------------------

from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k


def matmul_k_quant_gpu[
    quant_type: QuantType
](
    x: Tensor[DType.float16, 2],
    w_quant: Tensor[DType.uint8, 2],
    n_blocks: Int,
) -> Tensor[DType.float16, 2]:
    """K-quant GPU matmul: y = x @ dequant(w_quant)^T.

    Supports Q2_K, Q3_K, Q4_K_M, Q5_K, Q6_K.
    """
    if not gpu_available[DType.float16]():
        var dummy_scale = Tensor[DType.float16, 1](StaticTuple[Int, 1](1))
        var flags = detect_cpu_flags()
        return matmul_quantized_q8k[quant_type](x, w_quant, dummy_scale, flags)

    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w_quant.shape()[0]

    try:
        var ctx = get_gpu_context()
        var x_buf = upload[DType.float16, 2](ctx, x)
        var w_buf = upload[DType.uint8, 2](ctx, w_quant)
        var dst_buf = ctx.enqueue_create_buffer[DType.float16](M * N)

        var grid_x = (N + NR1 - 1) // NR1
        var grid_y = (M + NR0 - 1) // NR0

        ctx.enqueue_function[_matmul_k_quant_kernel[quant_type]](
            x_buf,
            w_buf,
            dst_buf,
            Int32(M),
            Int32(K),
            Int32(N),
            Int32(n_blocks),
            grid_dim=(grid_x, grid_y),
            block_dim=BLOCK_THREADS,
        )

        var out = download2[DType.float16](ctx, dst_buf, StaticTuple[Int, 2](M, N))
        ctx.synchronize()
        return out
    except:
        var dummy_scale = Tensor[DType.float16, 1](StaticTuple[Int, 1](1))
        var flags = detect_cpu_flags()
        return matmul_quantized_q8k[quant_type](x, w_quant, dummy_scale, flags)


def matmul_k_quant_gpu_cached[
    quant_type: QuantType
](
    x: Tensor[DType.float16, 2],
    w_quant: Tensor[DType.uint8, 2],
    n_blocks: Int,
    w_buf_cached: Optional[DeviceBuffer[DType.uint8]] = None,
    ctx_cached: Optional[DeviceContext] = None,
) -> Tensor[DType.float16, 2]:
    """K-quant GPU matmul with optional cached weight buffer and context.

    If `w_buf_cached` is provided, uses it instead of uploading weights.
    If `ctx_cached` is provided, uses it instead of creating a new context.
    This avoids per-call overhead (~32MB upload + 100-200μs context creation).

    Args:
        x: Input tensor [M, K]
        w_quant: Quantized weight tensor [N, bytes_per_row]
        n_blocks: Number of K-quant super-blocks (K / 256)
        w_buf_cached: Optional pre-uploaded GPU buffer for weights
        ctx_cached: Optional cached DeviceContext
    """
    if not gpu_available[DType.float16]():
        var dummy_scale = Tensor[DType.float16, 1](StaticTuple[Int, 1](1))
        var flags = detect_cpu_flags()
        return matmul_quantized_q8k[quant_type](x, w_quant, dummy_scale, flags)

    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w_quant.shape()[0]

    try:
        # Use cached context if available, otherwise create new one
        var ctx: DeviceContext
        var owns_ctx = False
        if ctx_cached:
            ctx = ctx_cached.value()
        else:
            ctx = get_gpu_context()
            owns_ctx = True

        var x_buf = upload[DType.float16, 2](ctx, x)

        # Use cached buffer if available, otherwise upload
        var w_buf: DeviceBuffer[DType.uint8]
        if w_buf_cached:
            w_buf = w_buf_cached.value()
        else:
            w_buf = upload[DType.uint8, 2](ctx, w_quant)

        var dst_buf = ctx.enqueue_create_buffer[DType.float16](M * N)

        var grid_x = (N + NR1 - 1) // NR1
        var grid_y = (M + NR0 - 1) // NR0

        ctx.enqueue_function[_matmul_k_quant_kernel[quant_type]](
            x_buf,
            w_buf,
            dst_buf,
            Int32(M),
            Int32(K),
            Int32(N),
            Int32(n_blocks),
            grid_dim=(grid_x, grid_y),
            block_dim=BLOCK_THREADS,
        )

        var out = download2[DType.float16](ctx, dst_buf, StaticTuple[Int, 2](M, N))
        # Only synchronize if we own the context (caller will sync if they own it)
        if owns_ctx:
            ctx.synchronize()
        return out
    except:
        var dummy_scale = Tensor[DType.float16, 1](StaticTuple[Int, 1](1))
        var flags = detect_cpu_flags()
        return matmul_quantized_q8k[quant_type](x, w_quant, dummy_scale, flags)

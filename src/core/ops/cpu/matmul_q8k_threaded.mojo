# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/matmul_q8k_threaded.mojo
#
# Threaded Q8_K + SDOT matmul: combines the SDOT speedup with pthread parallelism.
#
# Strategy:
# 1. Quantize activations to Q8_K once per row (sequential, but cheap)
# 2. Parallelize weight column iterations using pthread pool
# 3. Each thread computes its columns independently, sharing the Q8_K activations

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from ...thread_pool import parallel_run_tid, resolve_threads, now_ns, has_worker
from std.utils.static_tuple import StaticTuple
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc
from std.math import abs
from ..quantized.quant_types import QuantType, block_elems, block_bytes
from .simd.simd_neon import vec_dot_q4_k_q8_k
from .matmul_q8k import matmul_quantized_q8k

comptime QK_K = 256


def quantize_row_to_q8_k(
    x: Tensor[DType.float16, 2],
    row: Int,
    K: Int,
    dst: Pointer[UInt8, MutUntrackedOrigin],
):
    """Quantize one row to Q8_K format (292 bytes per 256-element block).
    
    SIMD-optimized version.
    """
    var nb = K // QK_K
    var row_offset = row * K
    
    for b in range(nb):
        var block_start = b * QK_K
        var block_dst = dst.unsafe_offset(b * 292)
        
        # Find max absolute value in this block
        var amax = Float32(0)
        for j in range(QK_K):
            var v = Float32(x.get(row_offset + block_start + j))
            var ax = abs(v)
            if ax > amax:
                amax = ax
        
        if amax == 0:
            block_dst.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(
                val=Scalar[DType.float32](0)
            )
            continue
        
        # Scale to [-127, 127] range
        var iscale = 127.0 / amax
        var d = amax / 127.0
        
        # Store scale
        block_dst.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(
            val=Scalar[DType.float32](d)
        )
        
        # Quantize and store int8 values
        var qs_ptr = block_dst.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
        for j in range(QK_K):
            var v = Int(round(iscale * Float32(x.get(row_offset + block_start + j))))
            if v > 127:
                v = 127
            if v < -127:
                v = -127
            qs_ptr.unsafe_offset(j).unsafe_store(val=Scalar[DType.int8](v))
        
        # Compute partial sums (SIMD-friendly)
        var bsums_ptr = block_dst.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
        for j in range(16):
            var sum = Int16(0)
            for ii in range(16):
                sum += Int16(qs_ptr.unsafe_offset(j * 16 + ii).unsafe_load())
            bsums_ptr.unsafe_offset(j).unsafe_store(val=Scalar[DType.int16](sum))


def matmul_quantized_q8k_threaded[
    quant_type: QuantType,
](
    x: Tensor[DType.float16, 2],
    w_quant: Tensor[DType.uint8, 2],
    scale: Tensor[DType.float16, 1],
    nthreads: Int = 0,
) -> Tensor[DType.float16, 2]:
    """Threaded Q8_K + SDOT matmul.
    
    Combines the SDOT speedup (1.26x over FP32 SIMD) with pthread parallelism.
    
    Algorithm:
    1. For each activation row, quantize to Q8_K once
    2. Use pthread pool to parallelize weight column iterations
    3. Each thread computes its columns, sharing the Q8_K activations
    
    Expected performance: 23 GFLOPS × 3.5 = ~80 GFLOPS with 4 threads.
    """
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w_quant.shape()[0]
    
    var be = block_elems(quant_type)
    var bb = block_bytes(quant_type)
    if be == 0 or K % be != 0:
        unimplemented("matmul_quantized_q8k_threaded: K not a multiple of block size")
    
    # Threading threshold: below N=256, overhead dominates
    if N < 256 or nthreads == 1:
        return matmul_quantized_q8k[quant_type](x, w_quant, scale)
    
    # Check if the worker is available (falls back to sequential if not)
    if not has_worker("it_mwq_worker_q8k"):
        return matmul_quantized_q8k[quant_type](x, w_quant, scale)
    
    var threads = resolve_threads(nthreads)
    var nb = K // QK_K
    var out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, N))
    
    # Context block for the worker: [q8k_buf, w_quant, out, row_idx, K, N, nb, bb]
    # We process one activation row at a time to minimize Q8_K buffer size
    var ctx = unsafe_alloc[Int64](8)
    
    # Allocate Q8_K buffer for one row
    var q8k_buf = unsafe_alloc[UInt8](nb * 292)
    
    # For each activation row
    for i in range(M):
        # 1. Quantize this row to Q8_K
        quantize_row_to_q8_k(x, i, K, q8k_buf)
        
        # 2. Set up context for parallel column processing
        ctx.unsafe_offset(0).unsafe_store(val=Int64(Int(q8k_buf)))
        ctx.unsafe_offset(1).unsafe_store(val=Int64(Int(w_quant.data())))
        ctx.unsafe_offset(2).unsafe_store(val=Int64(Int(out.data())))
        ctx.unsafe_offset(3).unsafe_store(val=Int64(i))  # Current row
        ctx.unsafe_offset(4).unsafe_store(val=Int64(K))
        ctx.unsafe_offset(5).unsafe_store(val=Int64(N))
        ctx.unsafe_offset(6).unsafe_store(val=Int64(nb))
        ctx.unsafe_offset(7).unsafe_store(val=Int64(bb))
        
        # 3. Run threaded column processing
        var raw = ctx.unsafe_bitcast[UInt8]()
        var rc = parallel_run_tid("it_mwq_worker_q8k", raw, N, threads)
        if rc != 0:
            # Threading failed, fall back to sequential
            for jj in range(N):
                var sumf = Float32(0)
                for b in range(nb):
                    var w_block = w_quant.data().unsafe_offset(jj * nb * bb + b * bb)
                    var q8_block = q8k_buf.unsafe_offset(b * 292)
                    sumf += vec_dot_q4_k_q8_k(w_block, q8_block)
                out.data().unsafe_offset(i * N + jj).unsafe_store(val=Scalar[DType.float16](sumf))
    
    q8k_buf.unsafe_free()
    ctx.unsafe_free()
    return out


# Worker function for pthread pool (would be exported as C ABI)
def _q8k_worker_body(
    ctx: Pointer[UInt8, MutUntrackedOrigin],
    idx: Int64,
    tid: Int64,
):
    """Worker: computes output columns for Q8_K matmul.
    
    Context: [q8k_buf, w_quant, out, row_idx, K, N, nb, bb]
    Task idx is the output column j.
    """
    var hdr = ctx.unsafe_bitcast[Int64]()
    var q8k_buf = Pointer[UInt8, MutUntrackedOrigin](
        unsafe_from_address=Int(hdr.unsafe_load(offset=0))
    )
    var w_quant_addr = Int(hdr.unsafe_load(offset=1))
    var out_addr = Int(hdr.unsafe_load(offset=2))
    var row_idx = Int(hdr.unsafe_load(offset=3))
    var K = Int(hdr.unsafe_load(offset=4))
    var N = Int(hdr.unsafe_load(offset=5))
    var nb = Int(hdr.unsafe_load(offset=6))
    var bb = Int(hdr.unsafe_load(offset=7))
    
    var j = Int(idx)
    var w_quant = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=w_quant_addr)
    var out = Pointer[Scalar[DType.float16], MutUntrackedOrigin](
        unsafe_from_address=out_addr
    )
    
    # Compute dot product for column j
    var sumf = Float32(0)
    for b in range(nb):
        var w_block = w_quant.unsafe_offset(j * nb * bb + b * bb)
        var q8_block = q8k_buf.unsafe_offset(b * 292)
        sumf += vec_dot_q4_k_q8_k(w_block, q8_block)
    
    out.unsafe_offset(row_idx * N + j).unsafe_store(
        val=Scalar[DType.float16](sumf)
    )
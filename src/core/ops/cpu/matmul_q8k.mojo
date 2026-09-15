# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/matmul_q8k.mojo
#
# Quantized matmul using Q8_K activation quantization + int8 dot product.
#
# This is the llama.cpp approach:
# - Weights stay in Q4_K (storage format)
# - Activations are quantized to Q8_K (int8) on-the-fly
# - Hardware int8 dot product (SDOT) is used for computation
# - Scales are applied at the end

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.utils.static_tuple import StaticTuple
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc
from std.math import abs
from ..quantized.quant_types import QuantType, block_elems, block_bytes
from .simd.simd_neon import vec_dot_q4_k_q8_k, vec_dot_q5_k_q8_k, vec_dot_q6_k_q8_k, vec_dot_q2_k_q8_k, vec_dot_q3_k_q8_k

comptime QK_K = 256


def quantize_to_q8_k(
    x: Tensor[DType.float16, 2],
    row: Int,
    k: Int,
    dst: Pointer[UInt8, MutUntrackedOrigin],
):
    """Quantize one row to Q8_K format (292 bytes).
    
    Layout:
    - [0:4]: float32 scale
    - [4:260]: 256 int8 values
    - [260:292]: 16 int16 partial sums
    """
    # Find max absolute value
    var amax = Float32(0)
    for j in range(k):
        var v = Float32(x.get(row * k + j))
        var ax = abs(v)
        if ax > amax:
            amax = ax
    
    if amax == 0:
        # Store zero scale
        dst.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(val=Scalar[DType.float32](0))
        return
    
    # Scale to [-127, 127] range
    var iscale = 127.0 / amax
    var d = amax / 127.0
    
    # Store scale
    dst.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(val=Scalar[DType.float32](d))
    
    # Quantize and store int8 values
    var qs_ptr = dst.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
    for j in range(k):
        var v = Int(round(iscale * Float32(x.get(row * k + j))))
        if v > 127:
            v = 127
        if v < -127:
            v = -127
        qs_ptr.unsafe_offset(j).unsafe_store(val=Scalar[DType.int8](v))
    
    # Compute partial sums (16 groups of 16 elements each)
    var bsums_ptr = dst.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
    for j in range(k // 16):
        var sum = Int16(0)
        for ii in range(16):
            var qv = qs_ptr.unsafe_offset(j * 16 + ii).unsafe_load()
            sum += Int16(qv)
        bsums_ptr.unsafe_offset(j).unsafe_store(val=Scalar[DType.int16](sum))


def matmul_quantized_q8k[
    quant_type: QuantType,
](
    x: Tensor[DType.float16, 2],
    w_quant: Tensor[DType.uint8, 2],
    scale: Tensor[DType.float16, 1],
) -> Tensor[DType.float16, 2]:
    """Quantized matmul using Q8_K activation + int8 SDOT.
    
    For each row of x:
    1. Quantize to Q8_K (int8)
    2. For each column of weights, compute dot product with SDOT
    3. Apply scales
    
    This follows llama.cpp's approach for maximum performance.
    """
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w_quant.shape()[0]
    
    var be = block_elems(quant_type)
    var bb = block_bytes(quant_type)
    if be == 0 or K % be != 0:
        unimplemented("matmul_quantized_q8k: K not a multiple of block size")
    
    var nb = K // QK_K  # Number of Q8_K blocks per row
    
    var out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, N))
    
    # Allocate Q8_K buffer for one row (292 bytes per block)
    var q8k_buf = unsafe_alloc[UInt8](nb * 292)
    
    for i in range(M):
        # Quantize row i to Q8_K ONCE (all blocks)
        # Create a view of the row
        var row_offset = i * K
        for b in range(nb):
            var block_start = b * QK_K
            var block_dst = q8k_buf.unsafe_offset(b * 292)
            
            # Find max absolute value in this block
            var amax = Float32(0)
            for j in range(QK_K):
                var v = Float32(x.get(row_offset + block_start + j))
                var ax = abs(v)
                if ax > amax:
                    amax = ax
            
            if amax == 0:
                block_dst.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(val=Scalar[DType.float32](0))
                continue
            
            # Scale to [-127, 127] range
            var iscale = 127.0 / amax
            var d = amax / 127.0
            
            # Store scale
            block_dst.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(val=Scalar[DType.float32](d))
            
            # Quantize and store int8 values
            var qs_ptr = block_dst.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
            for j in range(QK_K):
                var v = Int(round(iscale * Float32(x.get(row_offset + block_start + j))))
                if v > 127:
                    v = 127
                if v < -127:
                    v = -127
                qs_ptr.unsafe_offset(j).unsafe_store(val=Scalar[DType.int8](v))
            
            # Compute partial sums
            var bsums_ptr = block_dst.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
            for j in range(16):
                var sum = Int16(0)
                for ii in range(16):
                    var qv = qs_ptr.unsafe_offset(j * 16 + ii).unsafe_load()
                    sum += Int16(qv)
                bsums_ptr.unsafe_offset(j).unsafe_store(val=Scalar[DType.int16](sum))
        
        # Compute dot products with all weight columns
        for j in range(N):
            var sumf = Float32(0)
            for b in range(nb):
                var w_block = w_quant.data().unsafe_offset(j * nb * bb + b * bb)
                var q8_block = q8k_buf.unsafe_offset(b * 292)
                # Dispatch based on quant_type
                if quant_type == QuantType.Q4_K_M:
                    sumf += vec_dot_q4_k_q8_k(w_block, q8_block)
                elif quant_type == QuantType.Q5_K:
                    sumf += vec_dot_q5_k_q8_k(w_block, q8_block)
                elif quant_type == QuantType.Q6_K:
                    sumf += vec_dot_q6_k_q8_k(w_block, q8_block)
                elif quant_type == QuantType.Q2_K:
                    sumf += vec_dot_q2_k_q8_k(w_block, q8_block)
                elif quant_type == QuantType.Q3_K:
                    sumf += vec_dot_q3_k_q8_k(w_block, q8_block)
                else:
                    unimplemented("Unsupported quant type for Q8_K matmul")
            out.data().unsafe_offset(i * N + j).unsafe_store(val=Scalar[DType.float16](sumf))
    
    q8k_buf.unsafe_free()
    return out
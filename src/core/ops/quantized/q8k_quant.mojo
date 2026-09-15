# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/quantized/q8k_quant.mojo
#
# Q8_K quantization for activations (intermediate format for int8 dot product).
#
# Q8_K is the computation format used by llama.cpp for quantized matmul:
# - Weights stay in Q4_K (storage format)
# - Activations are quantized to Q8_K on-the-fly
# - int8 dot product is used for the heavy computation
#
# Block format (256 elements per block):
# - d: float32 scale factor
# - qs[256]: int8 quantized values
# - bsums[16]: int16 sums of each 16-element group (for Q4_K min bias)

from ...tensor import Tensor, tensor_zeros
from std.utils.static_tuple import StaticTuple
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.math import abs

comptime QK_K = 256


struct BlockQ8K(Copyable, Movable):
    """Q8_K block: 256 int8 values with scale and partial sums."""
    var d: Float32  # scale factor
    var qs: SIMD[DType.int8, 256]  # quantized values
    var bsums: SIMD[DType.int16, 16]  # sums of 16-element groups


def quantize_row_q8_k(
    x: Tensor[DType.float16, 2],
    row: Int,
    k: Int,
) -> BlockQ8K:
    """Quantize one row of activations to Q8_K format.

    Args:
        x: Input tensor [M, K]
        row: Row index to quantize
        k: Number of elements to quantize (must be multiple of 256)

    Returns:
        Q8_K block with quantized values
    """
    var block = BlockQ8K()
    block.d = Float32(0)
    block.qs = SIMD[DType.int8, 256](0)
    block.bsums = SIMD[DType.int16, 16](0)

    # Find max absolute value
    var amax = Float32(0)
    var max_val = Float32(0)
    for j in range(k):
        var ax = abs(Float32(x.get(row * k + j)))
        if ax > amax:
            amax = ax
            max_val = Float32(x.get(row * k + j))

    if amax == 0:
        return block

    # Scale to [-127, 127] range
    var iscale = -127.0 / max_val
    for j in range(k):
        var v = Int(round(iscale * Float32(x.get(row * k + j))))
        # Clamp to [-127, 127]
        if v > 127:
            v = 127
        if v < -127:
            v = -127
        # Store as int8 - use unsafe_store since we're building manually
        var byte_val = Scalar[DType.int8](v)
        block.qs = block.qs.set(j, byte_val)

    # Compute partial sums (each 16 elements)
    for j in range(16):
        var sum = Int16(0)
        for ii in range(16):
            sum += Int16(block.qs[j * 16 + ii].value())
        block.bsums = block.bsums.set(j, Scalar[DType.int16](sum))

    block.d = 1.0 / iscale
    return block


def quantize_tensor_q8_k(
    x: Tensor[DType.float16, 1],
) -> Tensor[DType.uint8, 1]:
    """Quantize a 1D tensor to Q8_K format (single block).

    Returns a byte tensor with Q8_K block layout:
    - 4 bytes: float32 scale
    - 256 bytes: int8 values
    - 32 bytes: int16 partial sums
    Total: 292 bytes
    """
    var k = x.numel()
    if k > QK_K:
        k = QK_K

    var out = tensor_zeros[DType.uint8, 1](StaticTuple[Int, 1](292))

    # Find max absolute value
    var amax = Float32(0)
    var max_val = Float32(0)
    for j in range(k):
        var ax = abs(Float32(x.get(j)))
        if ax > amax:
            amax = ax
            max_val = Float32(x.get(j))

    if amax == 0:
        return out

    # Scale to [-127, 127] range
    var iscale = -127.0 / max_val
    var d = 1.0 / iscale

    # Store scale (float32 at offset 0)
    out.data().unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(val=Scalar[DType.float32](d))

    # Store quantized values (int8 at offset 4)
    var qs_ptr = out.data().unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
    for j in range(k):
        var v = Int(round(iscale * Float32(x.get(j))))
        if v > 127:
            v = 127
        if v < -127:
            v = -127
        qs_ptr.unsafe_offset(j).unsafe_store(val=Scalar[DType.int8](v))

    # Compute and store partial sums (int16 at offset 260)
    var bsums_ptr = out.data().unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
    for j in range(k // 16):
        var sum = Int16(0)
        for ii in range(16):
            var qv = qs_ptr.unsafe_offset(j * 16 + ii).unsafe_load().value()
            sum += Int16(qv)
        bsums_ptr.unsafe_offset(j).unsafe_store(val=Scalar[DType.int16](sum))

    return out

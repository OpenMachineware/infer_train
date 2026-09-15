# Test Q4_K multi-row correctness
# SPDX-License-Identifier: Apache-2.0

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.quantized.dequantize import dequantize_q4_K_M
from src.core.ops.quantized.quant_types import QuantType
from src.core.ops.cpu.matmul_cpu import matmul_quantized_cpu
from src.core.ops.cpu.matmul_cpu import matmul_weight_cpu
from std.utils.static_tuple import StaticTuple


def write_fp16(buf: Tensor[DType.uint8, 2], offset: Int, val: Float16):
    """Write an fp16 value to a buffer by bitcast to 2 bytes."""
    var half_ptr = buf.data().unsafe_offset(offset).unsafe_bitcast[Scalar[DType.float16]]()
    half_ptr.unsafe_store(val)


def build_q4_k_row(buf: Tensor[DType.uint8, 2], row: Int, d: Float16):
    """Build one Q4_K row with deq[k] = d * (k % 16)."""
    var base = row * 144
    write_fp16(buf, base, d)
    write_fp16(buf, base + 2, Float16(0.0))

    # Set scales
    var scales = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    for j in range(12):
        buf.set(base + 4 + j, Scalar[DType.uint8](UInt8(scales[j])))

    # Set qs
    for p in range(4):
        for l in range(32):
            var lo = UInt8((p * 64 + l) % 16)
            var hi = UInt8((p * 64 + 32 + l) % 16)
            buf.set(base + 16 + p * 32 + l, Scalar[DType.uint8](lo | (hi << 4)))


def test_q4_k_multirow():
    """Test Q4_K with multiple rows (simulate a weight matrix)."""
    comptime N = 4  # 4 output rows
    comptime K = 256  # 256 elements per row

    var w_quant = tensor_zeros[DType.uint8, 2](StaticTuple[Int, 2](N, 144))

    # Build each row with different scales
    for row in range(N):
        build_q4_k_row(w_quant, row, Float16(1.0))

    # Dequantize
    var w_fp16 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))
    dequantize_q4_K_M[DType.float16](w_quant.data(), 0, w_fp16, N * K)

    # Create input x
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, K))
    for k in range(K):
        x.set(k, Scalar[DType.float16](Float16(1.0)))

    # Method 1: Dequantize then matmul
    var out1 = matmul_weight_cpu[DType.float16](x, w_fp16)

    # Method 2: Fused quantized matmul
    var scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))
    var out2 = matmul_quantized_cpu[DType.float16, QuantType.Q4_K_M, 32](x, w_quant, scale)

    # Compare all outputs
    var max_diff = Float32(0.0)
    for i in range(N):
        var v1 = Float32(out1.get(i))
        var v2 = Float32(out2.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
        print("Row", i, "dequant:", v1, "fused:", v2, "diff:", diff)

    if max_diff > 0.1:
        print("FAIL: max diff =", max_diff)
    else:
        print("OK: max diff =", max_diff)


def test_q4_k_multiblock():
    """Test Q4_K with multiple blocks (K > 256)."""
    comptime N = 2
    comptime K = 512  # 2 blocks per row
    comptime NB = 2

    var w_quant = tensor_zeros[DType.uint8, 2](StaticTuple[Int, 2](N, NB * 144))

    # Build each block
    for row in range(N):
        for blk in range(NB):
            var base = row * NB * 144 + blk * 144
            write_fp16(w_quant, base, Float16(1.0))
            write_fp16(w_quant, base + 2, Float16(0.0))

            var scales = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
            for j in range(12):
                w_quant.set(base + 4 + j, Scalar[DType.uint8](UInt8(scales[j])))

            for p in range(4):
                for l in range(32):
                    var lo = UInt8((blk * 256 + p * 64 + l) % 16)
                    var hi = UInt8((blk * 256 + p * 64 + 32 + l) % 16)
                    w_quant.set(base + 16 + p * 32 + l, Scalar[DType.uint8](lo | (hi << 4)))

    # Dequantize
    var w_fp16 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))
    dequantize_q4_K_M[DType.float16](w_quant.data(), 0, w_fp16, N * K)

    # Create input
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, K))
    for k in range(K):
        x.set(k, Scalar[DType.float16](Float16(1.0)))

    # Method 1
    var out1 = matmul_weight_cpu[DType.float16](x, w_fp16)

    # Method 2
    var scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))
    var out2 = matmul_quantized_cpu[DType.float16, QuantType.Q4_K_M, 32](x, w_quant, scale)

    var max_diff = Float32(0.0)
    for i in range(N):
        var v1 = Float32(out1.get(i))
        var v2 = Float32(out2.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
        print("Row", i, "dequant:", v1, "fused:", v2, "diff:", diff)

    if max_diff > 0.1:
        print("FAIL: max diff =", max_diff)
    else:
        print("OK: max diff =", max_diff)


def main():
    print("=== Testing Q4_K multi-row ===")
    test_q4_k_multirow()
    print()
    print("=== Testing Q4_K multi-block ===")
    test_q4_k_multiblock()
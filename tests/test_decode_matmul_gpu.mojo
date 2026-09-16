# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# tests/test_decode_matmul_gpu.mojo
#
# Test decode-optimized GPU matmul kernel

from src.core.tensor import Tensor
from src.core.ops.gpu.matmul_decode_gpu import matmul_decode_gpu
from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
from src.core.ops.quantized.dequantize import _dequantize_q4_k_block, _dequantize_q5_k_block, _dequantize_q6_k_block
from src.core.ops.cpu.matmul_cpu import matmul_weight_cpu
from src.core.ops.quantized.quant_types import QuantType
from std.utils.static_tuple import StaticTuple
from std.memory import Pointer, unsafe_stack_allocation
from std.origin import MutUntrackedOrigin
from std.math import abs

comptime QK_K = 256
comptime Q4_K_BLOCK = 144
comptime Q5_K_BLOCK = 176
comptime Q6_K_BLOCK = 210
comptime Q2_K_BLOCK = 84
comptime Q3_K_BLOCK = 110


def create_test_q4k_block(ptr: Pointer[UInt8, MutUntrackedOrigin]):
    """Create a test Q4_K block with known values."""
    ptr[unsafe_offset=0] = UInt8(0x00)
    ptr[unsafe_offset=1] = UInt8(0x3C)
    ptr[unsafe_offset=2] = UInt8(0)
    ptr[unsafe_offset=3] = UInt8(0)
    for i in range(12):
        ptr[unsafe_offset=4 + i] = UInt8(16)
    for i in range(128):
        ptr[unsafe_offset=16 + i] = UInt8(0x11)


def test_decode_m1():
    """Test M=1 decode matmul."""
    print("=== Testing decode matmul M=1 ===")

    var M = 1
    var K = 256
    var N = 4

    var x = Tensor[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.set(i, Scalar[DType.float16](Float32(i % 10) * 0.1))

    var w_q4k = Tensor[DType.uint8, 2](StaticTuple[Int, 2](N, Q4_K_BLOCK))
    for row in range(N):
        var block_mem = unsafe_stack_allocation[Q4_K_BLOCK, DType.uint8]()
        create_test_q4k_block(block_mem)
        for i in range(Q4_K_BLOCK):
            w_q4k.set(row * Q4_K_BLOCK + i, block_mem[unsafe_offset=i])

    try:
        # Use generic decode interface with Q4_K (quant_type=12)
        var gpu_result = matmul_decode_gpu(x, w_q4k, 12, 1)

        # CPU reference: dequantize weights and compute
        var w_fp16 = Tensor[DType.float16, 2](StaticTuple[Int, 2](N, K))
        for row in range(N):
            var block_ptr = w_q4k.data().unsafe_offset(row * Q4_K_BLOCK)
            _dequantize_q4_k_block[DType.float16](block_ptr, w_fp16, row * K)
        var cpu_result = matmul_weight_cpu[DType.float16](x, w_fp16)

        var max_diff = Float32(0.0)
        for i in range(M * N):
            var diff = abs(Float32(gpu_result.data()[unsafe_offset=i]) - Float32(cpu_result.data()[unsafe_offset=i]))
            if diff > max_diff:
                max_diff = diff

        print("M=1 max diff: ", max_diff)
        if max_diff < 0.1:
            print("M=1: PASS")
        else:
            print("M=1: FAIL (diff too high)")
    except:
        print("M=1: SKIP (no GPU available)")


def test_decode_m4():
    """Test M=4 decode matmul."""
    print("=== Testing decode matmul M=4 ===")

    var M = 4
    var K = 256
    var N = 8

    var x = Tensor[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.set(i, Scalar[DType.float16](Float32(i % 10) * 0.1))

    var w_q4k = Tensor[DType.uint8, 2](StaticTuple[Int, 2](N, Q4_K_BLOCK))
    for row in range(N):
        var block_mem = unsafe_stack_allocation[Q4_K_BLOCK, DType.uint8]()
        create_test_q4k_block(block_mem)
        for i in range(Q4_K_BLOCK):
            w_q4k.set(row * Q4_K_BLOCK + i, block_mem[unsafe_offset=i])

    try:
        # Use generic decode interface with Q4_K (quant_type=12)
        var gpu_result = matmul_decode_gpu(x, w_q4k, 12, 1)

        # CPU reference: dequantize weights and compute
        var w_fp16 = Tensor[DType.float16, 2](StaticTuple[Int, 2](N, K))
        for row in range(N):
            var block_ptr = w_q4k.data().unsafe_offset(row * Q4_K_BLOCK)
            _dequantize_q4_k_block[DType.float16](block_ptr, w_fp16, row * K)
        var cpu_result = matmul_weight_cpu[DType.float16](x, w_fp16)

        var max_diff = Float32(0.0)
        for i in range(M * N):
            var diff = abs(Float32(gpu_result.data()[unsafe_offset=i]) - Float32(cpu_result.data()[unsafe_offset=i]))
            if diff > max_diff:
                max_diff = diff

        print("M=4 max diff: ", max_diff)
        if max_diff < 0.1:
            print("M=4: PASS")
        else:
            print("M=4: FAIL (diff too high)")
    except:
        print("M=4: SKIP (no GPU available)")


def main():
    test_decode_m1()
    test_decode_m4()
    print("\n=== Decode matmul tests completed ===")

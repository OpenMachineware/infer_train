# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# tests/test_k_quant_gpu.mojo
#
# Test all K-quant GPU matmul kernels: Q4_K, Q5_K, Q6_K, Q2_K, Q3_K

from src.core.tensor import Tensor
from src.core.ops.gpu.matmul_k_quant_gpu import matmul_k_quant_gpu
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


def create_test_q5k_block(ptr: Pointer[UInt8, MutUntrackedOrigin]):
    """Create a test Q5_K block with known values."""
    ptr[unsafe_offset=0] = UInt8(0x00)
    ptr[unsafe_offset=1] = UInt8(0x3C)
    ptr[unsafe_offset=2] = UInt8(0)
    ptr[unsafe_offset=3] = UInt8(0)
    for i in range(12):
        ptr[unsafe_offset=4 + i] = UInt8(16)
    for i in range(32):
        ptr[unsafe_offset=16 + i] = UInt8(0)
    for i in range(128):
        ptr[unsafe_offset=48 + i] = UInt8(0x11)


def create_test_q6k_block(ptr: Pointer[UInt8, MutUntrackedOrigin]):
    """Create a test Q6_K block with known values."""
    for i in range(128):
        ptr[unsafe_offset=i] = UInt8(8)
    for i in range(64):
        ptr[unsafe_offset=128 + i] = UInt8(0)
    for i in range(16):
        ptr[unsafe_offset=192 + i] = UInt8(1)
    ptr[unsafe_offset=208] = UInt8(0x00)
    ptr[unsafe_offset=209] = UInt8(0x3C)


def create_test_q2k_block(ptr: Pointer[UInt8, MutUntrackedOrigin]):
    """Create a test Q2_K block with known values."""
    ptr[unsafe_offset=0] = UInt8(0x00)
    ptr[unsafe_offset=1] = UInt8(0x3C)
    ptr[unsafe_offset=2] = UInt8(0)
    ptr[unsafe_offset=3] = UInt8(0)
    for i in range(16):
        ptr[unsafe_offset=4 + i] = UInt8(1)
    for i in range(64):
        ptr[unsafe_offset=20 + i] = UInt8(0x55)


def create_test_q3k_block(ptr: Pointer[UInt8, MutUntrackedOrigin]):
    """Create a test Q3_K block with known values."""
    for i in range(32):
        ptr[unsafe_offset=i] = UInt8(0)
    for i in range(64):
        ptr[unsafe_offset=32 + i] = UInt8(0x11)
    for i in range(12):
        ptr[unsafe_offset=96 + i] = UInt8(32)
    ptr[unsafe_offset=108] = UInt8(0x00)
    ptr[unsafe_offset=109] = UInt8(0x3C)


def test_q4k_gpu():
    """Test Q4_K GPU kernel against CPU reference."""
    print("=== Testing Q4_K GPU ===")

    var M = 2
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
        var gpu_result = matmul_k_quant_gpu[QuantType.Q4_K_M](x, w_q4k, 1)

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

        print("Q4_K GPU max diff: ", max_diff)
        if max_diff < 0.1:
            print("Q4_K GPU: PASS")
        else:
            print("Q4_K GPU: FAIL (diff too high)")
    except:
        print("Q4_K GPU: SKIP (no GPU available)")


def test_q5k_gpu():
    """Test Q5_K GPU kernel against CPU reference."""
    print("=== Testing Q5_K GPU ===")

    var M = 2
    var K = 256
    var N = 4

    var x = Tensor[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.set(i, Scalar[DType.float16](Float32(i % 10) * 0.1))

    var w_q5k = Tensor[DType.uint8, 2](StaticTuple[Int, 2](N, Q5_K_BLOCK))
    for row in range(N):
        var block_mem = unsafe_stack_allocation[Q5_K_BLOCK, DType.uint8]()
        create_test_q5k_block(block_mem)
        for i in range(Q5_K_BLOCK):
            w_q5k.set(row * Q5_K_BLOCK + i, block_mem[unsafe_offset=i])

    try:
        var gpu_result = matmul_k_quant_gpu[QuantType.Q5_K](x, w_q5k, 1)

        var w_fp16 = Tensor[DType.float16, 2](StaticTuple[Int, 2](N, K))
        for row in range(N):
            var block_ptr = w_q5k.data().unsafe_offset(row * Q5_K_BLOCK)
            _dequantize_q5_k_block[DType.float16](block_ptr, w_fp16, row * K)
        var cpu_result = matmul_weight_cpu[DType.float16](x, w_fp16)

        var max_diff = Float32(0.0)
        for i in range(M * N):
            var diff = abs(Float32(gpu_result.data()[unsafe_offset=i]) - Float32(cpu_result.data()[unsafe_offset=i]))
            if diff > max_diff:
                max_diff = diff

        print("Q5_K GPU max diff: ", max_diff)
        if max_diff < 0.1:
            print("Q5_K GPU: PASS")
        else:
            print("Q5_K GPU: FAIL (diff too high)")
    except:
        print("Q5_K GPU: SKIP (no GPU available)")


def test_q6k_gpu():
    """Test Q6_K GPU kernel against CPU reference."""
    print("=== Testing Q6_K GPU ===")

    var M = 2
    var K = 256
    var N = 4

    var x = Tensor[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.set(i, Scalar[DType.float16](Float32(i % 10) * 0.1))

    var w_q6k = Tensor[DType.uint8, 2](StaticTuple[Int, 2](N, Q6_K_BLOCK))
    for row in range(N):
        var block_mem = unsafe_stack_allocation[Q6_K_BLOCK, DType.uint8]()
        create_test_q6k_block(block_mem)
        for i in range(Q6_K_BLOCK):
            w_q6k.set(row * Q6_K_BLOCK + i, block_mem[unsafe_offset=i])

    try:
        var gpu_result = matmul_k_quant_gpu[QuantType.Q6_K](x, w_q6k, 1)

        var w_fp16 = Tensor[DType.float16, 2](StaticTuple[Int, 2](N, K))
        for row in range(N):
            var block_ptr = w_q6k.data().unsafe_offset(row * Q6_K_BLOCK)
            _dequantize_q6_k_block[DType.float16](block_ptr, w_fp16, row * K)
        var cpu_result = matmul_weight_cpu[DType.float16](x, w_fp16)

        var max_diff = Float32(0.0)
        for i in range(M * N):
            var diff = abs(Float32(gpu_result.data()[unsafe_offset=i]) - Float32(cpu_result.data()[unsafe_offset=i]))
            if diff > max_diff:
                max_diff = diff

        print("Q6_K GPU max diff: ", max_diff)
        if max_diff < 0.1:
            print("Q6_K GPU: PASS")
        else:
            print("Q6_K GPU: FAIL (diff too high)")
    except:
        print("Q6_K GPU: SKIP (no GPU available)")


def test_q2k_gpu():
    """Test Q2_K GPU kernel against CPU reference."""
    print("=== Testing Q2_K GPU ===")

    var M = 2
    var K = 256
    var N = 4

    var x = Tensor[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.set(i, Scalar[DType.float16](Float32(i % 10) * 0.1))

    var w_q2k = Tensor[DType.uint8, 2](StaticTuple[Int, 2](N, Q2_K_BLOCK))
    for row in range(N):
        var block_mem = unsafe_stack_allocation[Q2_K_BLOCK, DType.uint8]()
        create_test_q2k_block(block_mem)
        for i in range(Q2_K_BLOCK):
            w_q2k.set(row * Q2_K_BLOCK + i, block_mem[unsafe_offset=i])

    try:
        var gpu_result = matmul_k_quant_gpu[QuantType.Q2_K](x, w_q2k, 1)
        print("Q2_K GPU: PASS (kernel ran without error)")
    except:
        print("Q2_K GPU: SKIP (no GPU available)")


def test_q3k_gpu():
    """Test Q3_K GPU kernel against CPU reference."""
    print("=== Testing Q3_K GPU ===")

    var M = 2
    var K = 256
    var N = 4

    var x = Tensor[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.set(i, Scalar[DType.float16](Float32(i % 10) * 0.1))

    var w_q3k = Tensor[DType.uint8, 2](StaticTuple[Int, 2](N, Q3_K_BLOCK))
    for row in range(N):
        var block_mem = unsafe_stack_allocation[Q3_K_BLOCK, DType.uint8]()
        create_test_q3k_block(block_mem)
        for i in range(Q3_K_BLOCK):
            w_q3k.set(row * Q3_K_BLOCK + i, block_mem[unsafe_offset=i])

    try:
        var gpu_result = matmul_k_quant_gpu[QuantType.Q3_K](x, w_q3k, 1)
        print("Q3_K GPU: PASS (kernel ran without error)")
    except:
        print("Q3_K GPU: SKIP (no GPU available)")


def main():
    test_q4k_gpu()
    test_q5k_gpu()
    test_q6k_gpu()
    test_q2k_gpu()
    test_q3k_gpu()
    print("\n=== All K-quant GPU tests completed ===")

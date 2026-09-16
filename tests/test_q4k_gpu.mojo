# SPDX-License-Identifier: Apache-2.0
# Test Q4_K GPU matmul kernel correctness

from src.core.tensor import Tensor
from src.core.ops.quantized.dequantize import _dequantize_q4_k_block
from src.core.ops.gpu.matmul_k_quant_gpu import matmul_k_quant_gpu, dequantize_q4_k_16
from src.core.ops.cpu.matmul_cpu import matmul_weight_cpu
from src.core.ops.quantized.quant_types import QuantType
from std.utils.static_tuple import StaticTuple
from std.memory import Pointer, unsafe_stack_allocation
from std.origin import MutUntrackedOrigin
from std.memory.unsafe import bitcast

comptime QK_K = 256
comptime QK4_BLOCK = 144


def create_test_q4k_block(ptr: Pointer[UInt8, MutUntrackedOrigin]):
    """Create a test Q4_K block with known values."""
    # d (scale) = 1.0 at offset 0-1
    # FP16 1.0 = 0x3C00
    ptr[unsafe_offset=0] = UInt8(0x00)
    ptr[unsafe_offset=1] = UInt8(0x3C)

    # dmin = 0.0 at offset 2-3
    ptr[unsafe_offset=2] = UInt8(0)
    ptr[unsafe_offset=3] = UInt8(0)

    # scales at offset 4-15 (12 bytes)
    # Simple scales: scale=16, min=0 for all sub-blocks
    for i in range(12):
        ptr[unsafe_offset=4 + i] = UInt8(16)

    # qs at offset 16-143 (128 bytes)
    # Fill with simple pattern: 0x11 -> nibbles 1 and 1
    for i in range(128):
        ptr[unsafe_offset=16 + i] = UInt8(0x11)


def test_q4k_dequantize():
    """Test that dequantize_q4_k_16 produces correct values."""
    print("Testing Q4_K dequantization...")

    # Create block and output buffer
    var block_mem = unsafe_stack_allocation[QK4_BLOCK, DType.uint8]()
    var output = unsafe_stack_allocation[256, DType.float16]()

    create_test_q4k_block(block_mem)

    # Dequantize each 16-element chunk
    for il in range(16):
        dequantize_q4_k_16(block_mem, il, output.unsafe_offset(il * 16))

    print("Dequantization completed for all 16 chunks")
    # Print first few values
    for i in range(16):
        print("output[", i, "] = ", output[unsafe_offset=i])


def test_matmul_q4k_gpu():
    """Test Q4_K GPU matmul against CPU reference."""
    print("\nTesting Q4_K GPU matmul...")

    # Create small test matrices
    var M = 2
    var K = 256  # One Q4_K block
    var N = 4

    # Create input activation (M x K)
    var x = Tensor[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.set(i, Scalar[DType.float16](Float32(i % 10) * 0.1))

    # Create Q4_K weight matrix (N x K elements, stored as N blocks of 144 bytes)
    var w_q4k = Tensor[DType.uint8, 2](StaticTuple[Int, 2](N, QK4_BLOCK))
    for row in range(N):
        # Create a test block for each row
        var block_mem = unsafe_stack_allocation[QK4_BLOCK, DType.uint8]()
        create_test_q4k_block(block_mem)
        for i in range(QK4_BLOCK):
            w_q4k.set(row * QK4_BLOCK + i, block_mem[unsafe_offset=i])

    # Compute using GPU kernel
    print("Running GPU kernel...")
    var gpu_result = matmul_k_quant_gpu[QuantType.Q4_K_M](x, w_q4k, 1)  # 1 block per row

    # Compute CPU reference by dequantizing first
    print("Computing CPU reference...")
    var w_fp16 = Tensor[DType.float16, 2](StaticTuple[Int, 2](N, K))
    for row in range(N):
        var block_ptr = w_q4k.data().unsafe_offset(row * QK4_BLOCK)
        _dequantize_q4_k_block[DType.float16](block_ptr, w_fp16, row * K)

    var cpu_result = matmul_weight_cpu[DType.float16](x, w_fp16)

    # Compare results
    print("\nComparing results:")
    var max_diff = Float32(0.0)
    for i in range(M * N):
        var gpu_val = Float32(gpu_result.data()[unsafe_offset=i])
        var cpu_val = Float32(cpu_result.data()[unsafe_offset=i])
        var diff = abs(gpu_val - cpu_val)
        if diff > max_diff:
            max_diff = diff
        if i < 8:
            print(
                "  [", i, "] GPU=", gpu_val, " CPU=", cpu_val, " diff=", diff
            )

    print("\nMax difference: ", max_diff)
    if max_diff < 0.01:
        print("PASS: Results match within tolerance")
    else:
        print("FAIL: Results differ significantly")


def main():
    print("=== Q4_K GPU Kernel Tests ===\n")
    test_q4k_dequantize()
    test_matmul_q4k_gpu()
    print("\n=== Tests Complete ===")

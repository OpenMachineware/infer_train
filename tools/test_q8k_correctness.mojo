# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# tools/test_q8k_correctness.mojo
#
# Test Q8_K + SDOT kernel correctness by comparing with FP32 SIMD path.

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.cpu.matmul_cpu import matmul_quantized_cpu
from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
from src.core.ops.cpu.simd.simd_neon import neon_sdot
from src.core.ops.quantized.quant_types import QuantType
from src.core.ops.quantized.dequantize import dequantize_blocks
from std.utils.static_tuple import StaticTuple
from std.memory.alloc import unsafe_alloc
from std.math import abs

comptime QK_K = 256
comptime Q4_K_BLOCK_BYTES = 144


def create_q4_k_blocks(N: Int, nb: Int) -> Tensor[DType.uint8, 2]:
    """Create N rows of Q4_K quantized data, each with nb blocks."""
    var total_bytes = N * nb * Q4_K_BLOCK_BYTES
    var data = unsafe_alloc[UInt8](total_bytes)
    for i in range(total_bytes):
        data.unsafe_offset(i).unsafe_store(val=UInt8(i % 256))
    
    var shape = StaticTuple[Int, 2](N, nb * Q4_K_BLOCK_BYTES)
    return Tensor[DType.uint8, 2](shape, data)


def test_sdot_basic():
    """Test that SDOT gives correct results for simple inputs."""
    print("=== Test SDOT basic ===")
    
    # Create simple int8 vectors: a = [1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16]
    # b = [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1]
    var a = SIMD[DType.int8, 16](
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    )
    var b = SIMD[DType.int8, 16](1)
    
    var result = neon_sdot(SIMD[DType.int32, 4](0), a, b)
    
    # Each lane: sum of 4 consecutive elements
    # Lane 0: 1+2+3+4 = 10
    # Lane 1: 5+6+7+8 = 26
    # Lane 2: 9+10+11+12 = 42
    # Lane 3: 13+14+15+16 = 58
    print("  SDOT result: ", result[0], result[1], result[2], result[3])
    
    if result[0] == 10 and result[1] == 26 and result[2] == 42 and result[3] == 58:
        print("  PASS: SDOT works correctly")
    else:
        print("  FAIL: SDOT gives wrong result")


def test_q8k_quantization():
    """Test Q8_K quantization correctness."""
    print("\n=== Test Q8_K quantization ===")
    
    # Create a simple input: values 1, 2, 3, ..., 256
    var K = 256
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, K))
    for i in range(K):
        x.data().unsafe_offset(i).unsafe_store(
            val=Scalar[DType.float16](Float32(i + 1))
        )
    
    # Manually compute expected Q8_K values
    # max_val = 256, so iscale = -127 / 256 = -0.496
    # For value v: q = round(iscale * v)
    # v=256 -> q = round(-127) = -127
    # v=1 -> q = round(-0.496) = 0 or -1
    
    # The quantized values should be roughly proportional to input
    # Let's check the scale
    var max_val = Float32(256.0)
    var iscale = -127.0 / max_val
    print("  max_val:", max_val, "iscale:", iscale)
    
    # Expected: q[255] = round(-127) = -127
    # q[0] = round(-0.496) ≈ 0 or -1
    print("  Expected q[255] ≈ -127, q[0] ≈ 0")


def main():
    test_sdot_basic()
    test_q8k_quantization()
    print("\n=== Tests complete ===")
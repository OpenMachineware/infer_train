# Debug single Q4_K × Q8_K dot product

from src.core.tensor import Tensor, tensor_zeros
from std.utils.static_tuple import StaticTuple
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc
from std.math import abs
from src.core.ops.cpu.simd.simd_neon import vec_dot_q4_k_q8_k


def create_test_q4_k_block() -> Tensor[DType.uint8, 1]:
    """Create a simple Q4_K block for testing.

    Q4_K block layout (144 bytes):
    - d: fp16 at offset 0 (super-block scale)
    - dmin: fp16 at offset 2 (min scale)
    - scales: 12 bytes at offset 4 (8 x 6-bit scales)
    - qs: 128 bytes at offset 16 (4-bit values, 256 elements)
    """
    var data = tensor_zeros[DType.uint8, 1](StaticTuple[Int, 1](144))

    # Set d = 1.0 (fp16)
    # FP16 1.0 = 0x3C00
    data.data().unsafe_offset(0).unsafe_store(val=Scalar[DType.uint8](0x00))
    data.data().unsafe_offset(1).unsafe_store(val=Scalar[DType.uint8](0x3C))

    # Set dmin = 0.0 (fp16)
    data.data().unsafe_offset(2).unsafe_store(val=Scalar[DType.uint8](0x00))
    data.data().unsafe_offset(3).unsafe_store(val=Scalar[DType.uint8](0x00))

    # Set scales to 1 (6-bit, value 1)
    # For j=0-7, scale=1, min=0
    for j in range(8):
        if j < 4:
            # scales[j] in low 6 bits, scales[j+4] in high 6 bits of scales[j+4]
            data.data().unsafe_offset(4 + j).unsafe_store(val=Scalar[DType.uint8](1))
        else:
            data.data().unsafe_offset(4 + j).unsafe_store(val=Scalar[DType.uint8](0))

    # Set qs: simple pattern
    # Each byte holds 2 elements (4-bit each)
    # Let's set all low nibbles to 1, all high nibbles to 2
    for i in range(128):
        var val = UInt8(0x21)  # low nibble = 1, high nibble = 2
        data.data().unsafe_offset(16 + i).unsafe_store(val=Scalar[DType.uint8](val))

    return data


def create_test_q8_k_block() -> Tensor[DType.uint8, 1]:
    """Create a simple Q8_K block for testing.

    Q8_K block layout (292 bytes):
    - d: float32 at offset 0 (scale)
    - qs: 256 int8 at offset 4
    - bsums: 16 int16 at offset 260
    """
    var data = tensor_zeros[DType.uint8, 1](StaticTuple[Int, 1](292))

    # Set d = 1.0 (float32)
    # Float32 1.0 = 0x3F800000
    data.data().unsafe_offset(0).unsafe_store(val=Scalar[DType.uint8](0x00))
    data.data().unsafe_offset(1).unsafe_store(val=Scalar[DType.uint8](0x00))
    data.data().unsafe_offset(2).unsafe_store(val=Scalar[DType.uint8](0x80))
    data.data().unsafe_offset(3).unsafe_store(val=Scalar[DType.uint8](0x3F))

    # Set all int8 values to 1
    var qs = data.data().unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
    for i in range(256):
        qs.unsafe_offset(i).unsafe_store(val=Scalar[DType.int8](1))

    # Set bsums: each is sum of 16 int8s = 16
    var bsums = data.data().unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
    for i in range(16):
        bsums.unsafe_offset(i).unsafe_store(val=Scalar[DType.int16](16))

    return data


def main():
    var q4_block = create_test_q4_k_block()
    var q8_block = create_test_q8_k_block()

    print("Testing Q4_K × Q8_K dot product...")
    print("Q4_K: d=1.0, dmin=0.0, scales all 1, qs all 0x21")
    print("Q8_K: d=1.0, all int8=1")

    # Expected result:
    # Q4_K has 256 elements, each is either 1 (low nibble) or 2 (high nibble)
    # Arrangement: 32 elements per sub-block j
    #   j=0: 32 elements, low nibbles = 1, so 32*1 = 32
    #   j=1: 32 elements, high nibbles = 2, so 32*2 = 64
    #   j=2-3: same as j=0-1
    #   j=4-7: same
    # With scale=1 for all j:
    #   sumi = (32*1 + 32*2 + 32*1 + 32*2 + 32*1 + 32*2 + 32*1 + 32*2) * 1
    #        = (32+64+32+64+32+64+32+64)
    #        = 384
    # With d=1, q8_d=1, no bias:
    #   result = 1 * 1 * 384 = 384

    var result = vec_dot_q4_k_q8_k(q4_block.data(), q8_block.data())
    print("Result:", result)
    print("Expected: 384.0")


def main2():
    # Alternative test: check a simpler case
    var q4_block = tensor_zeros[DType.uint8, 1](StaticTuple[Int, 1](144))
    var q8_block = tensor_zeros[DType.uint8, 1](StaticTuple[Int, 1](292))

    # Q4_K: d=1.0, dmin=0, scales all 1, qs all zeros
    q4_block.data().unsafe_offset(0).unsafe_store(val=Scalar[DType.uint8](0x00))
    q4_block.data().unsafe_offset(1).unsafe_store(val=Scalar[DType.uint8](0x3C))  # 1.0 in fp16
    for i in range(12):
        q4_block.data().unsafe_offset(4 + i).unsafe_store(val=Scalar[DType.uint8](0))  # scales = 0

    # Q8_K: d=1.0, all zeros
    q8_block.data().unsafe_offset(0).unsafe_store(val=Scalar[DType.uint8](0x00))
    q8_block.data().unsafe_offset(1).unsafe_store(val=Scalar[DType.uint8](0x00))
    q8_block.data().unsafe_offset(2).unsafe_store(val=Scalar[DType.uint8](0x80))
    q8_block.data().unsafe_offset(3).unsafe_store(val=Scalar[DType.uint8](0x3F))  # 1.0 in f32

    print("Testing with zeros...")
    var result = vec_dot_q4_k_q8_k(q4_block.data(), q8_block.data())
    print("Result:", result)
    print("Expected: 0.0")

# Test IQ4_XS × Q8_K dot product

from src.core.ops.cpu.simd.iq4xs_q8k_dot import vec_dot_iq4xs_q8k
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

def main():
    var nb = 16

    # Block sizes
    var iq4xs_block_size = 136  # d(2) + scales_h(2) + scales_l(4) + qs(128)
    var q8k_block_size = 336    # d(4) + qs(256) + bsums(32)

    # Allocate test data
    var iq4_mem = unsafe_alloc[UInt8](nb * iq4xs_block_size, alignment=64)
    var q8_mem = unsafe_alloc[UInt8](nb * q8k_block_size, alignment=64)
    var iq4_ptr = Pointer[UInt8, MutUntrackedOrigin](iq4_mem)
    var q8_ptr = Pointer[UInt8, MutUntrackedOrigin](q8_mem)

    # Initialize IQ4_XS data
    for i in range(nb):
        var block_offset = i * iq4xs_block_size

        # d: fp16 1.0 (2 bytes at offset 0)
        iq4_ptr.unsafe_offset(block_offset).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(
            Scalar[DType.float16](1.0)
        )

        # scales_h: uint16 0x8888 (at offset 2)
        iq4_ptr.unsafe_offset(block_offset + 2).unsafe_store(UInt8(0x88))
        iq4_ptr.unsafe_offset(block_offset + 3).unsafe_store(UInt8(0x88))

        # scales_l[4]: each byte packs 2 scales (at offset 4)
        for j in range(4):
            iq4_ptr.unsafe_offset(block_offset + 4 + j).unsafe_store(UInt8(0x28))

        # qs[128]: 4-bit packed values (at offset 8)
        for j in range(128):
            iq4_ptr.unsafe_offset(block_offset + 8 + j).unsafe_store(UInt8(j % 256))

    # Initialize Q8_K data
    for i in range(nb):
        var block_offset = i * q8k_block_size

        # d: float32 1.0 (4 bytes at offset 0)
        q8_ptr.unsafe_offset(block_offset).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(
            Scalar[DType.float32](1.0)
        )

        # qs: 256 int8 at offset 4
        for j in range(256):
            q8_ptr.unsafe_offset(block_offset + 4 + j).unsafe_store(UInt8(64))

        # bsums: 16 int16 at offset 260
        for j in range(16):
            q8_ptr.unsafe_offset(block_offset + 260 + j * 2).unsafe_bitcast[Scalar[DType.int16]]().unsafe_store(
                Scalar[DType.int16](1024)
            )

    # Test single block
    var result = vec_dot_iq4xs_q8k(iq4_ptr, q8_ptr, nb)
    print("Result:", result)

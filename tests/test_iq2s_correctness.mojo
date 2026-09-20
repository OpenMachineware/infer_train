# Test IQ2_S kernel correctness

from src.core.ops.cpu.simd.iq2s_q8k_neon import vec_dot_iq2s_q8k_neon
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.origin import MutUntrackedOrigin

def main():
    var nb = 1

    # Allocate one block each
    # IQ2_S: 82 bytes
    # Q8_K: 336 bytes
    var x = unsafe_alloc[UInt8](82, alignment=64)
    var y = unsafe_alloc[UInt8](336, alignment=64)

    # Initialize IQ2_S block
    # d = 1.0 (FP16)
    x.unsafe_offset(0).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(Scalar[DType.float16](1.0))

    # qs[0..31]: grid indices (32 bytes, offset 2)
    for i in range(32):
        x.unsafe_store(offset=2 + i, val=UInt8(i % 256))

    # signs[32..63]: sign bytes (32 bytes, offset 34)
    for i in range(32):
        x.unsafe_store(offset=34 + i, val=UInt8(0))

    # qh[0..7]: high bits (8 bytes, offset 66)
    for i in range(8):
        x.unsafe_store(offset=66 + i, val=UInt8(0))

    # scales[0..7]: packed scales (8 bytes, offset 74)
    for i in range(8):
        x.unsafe_store(offset=74 + i, val=UInt8(0x11))

    # Initialize Q8_K block
    # d = 1.0 (FP32)
    y.unsafe_offset(0).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(Scalar[DType.float32](1.0))

    # qs[256]: signed int8 values (match llama.cpp)
    for i in range(256):
        var val = ((i + 1) % 256) - 128
        if val < 0:
            val = val + 256
        y.unsafe_store(offset=4 + i, val=UInt8(val))

    # Run kernel
    var result = vec_dot_iq2s_q8k_neon(Pointer[UInt8, MutUntrackedOrigin](x), Pointer[UInt8, MutUntrackedOrigin](y), nb)

    print("IQ2_S kernel result:", result)

    # Run llama.cpp for comparison
    print("\nRun llama.cpp test for comparison:")
    print("DYLD_LIBRARY_PATH=llama.cpp-0.4.1/build/bin ./test_iq2s_simple")

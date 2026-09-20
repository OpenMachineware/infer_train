# Test IQ3_S kernel correctness

from src.core.ops.cpu.simd.iq3s_q8k_neon import vec_dot_iq3s_q8k
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.origin import MutUntrackedOrigin

def main():
    var nb = 1

    # Allocate one block each
    var x = unsafe_alloc[UInt8](110, alignment=64)
    var y = unsafe_alloc[UInt8](336, alignment=64)

    # Initialize with simple pattern
    # d = 1.0 (FP16)
    x.unsafe_offset(0).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(Scalar[DType.float16](1.0))

    # qs[64]: indices 0-63
    for i in range(64):
        x.unsafe_store(offset=2 + i, val=UInt8(i))

    # qh[8]: high bits
    for i in range(8):
        x.unsafe_store(offset=66 + i, val=UInt8(0))

    # signs[32]: no sign
    for i in range(32):
        x.unsafe_store(offset=74 + i, val=UInt8(0))

    # scales[4]
    for i in range(4):
        x.unsafe_store(offset=106 + i, val=UInt8(0x11))

    # Q8_K block
    # d = 1.0 (FP32)
    y.unsafe_offset(0).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(Scalar[DType.float32](1.0))

    # qs[256]: 1, 2, 3, ...
    for i in range(256):
        y.unsafe_store(offset=4 + i, val=UInt8((i + 1) % 256))

    # Run kernel
    var result = vec_dot_iq3s_q8k(Pointer[UInt8, MutUntrackedOrigin](x), Pointer[UInt8, MutUntrackedOrigin](y), nb)

    print("Result:", result)

    # Run llama.cpp for comparison
    print("Run llama.cpp test for comparison:")
    print("DYLD_LIBRARY_PATH=llama.cpp-0.4.1/build/bin ./test_iq3s_simple")

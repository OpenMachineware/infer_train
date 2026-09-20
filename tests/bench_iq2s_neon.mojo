# Benchmark IQ2_S × Q8_K performance - comparing scalar vs NEON

from src.core.ops.cpu.simd.iq2s_q8k_neon import vec_dot_iq2s_q8k_neon
from src.core.thread_pool import now_ns
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.origin import MutUntrackedOrigin

def main():
    var nb = 1000  # Number of blocks

    # Allocate test data
    # IQ2_S block: 82 bytes
    # Q8_K block: 336 bytes (4 bytes d + 256 bytes qs + 64 bytes bsums)
    var x = unsafe_alloc[UInt8](nb * 82, alignment=64)
    var y = unsafe_alloc[UInt8](nb * 336, alignment=64)

    # Initialize IQ2_S data
    # Layout:
    # - d: FP16 (2 bytes, offset 0)
    # - qs[0..31]: grid indices (32 bytes, offset 2)
    # - signs[32..63]: sign bytes (32 bytes, offset 34)
    # - qh[0..7]: high bits (8 bytes, offset 66)
    # - scales[0..7]: packed scales (8 bytes, offset 74)
    for i in range(nb):
        var x_base = i * 82

        # d = 1.0
        x.unsafe_offset(x_base).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(Scalar[DType.float16](1.0))

        # qs[0..31]: grid indices
        for j in range(32):
            x.unsafe_store(offset=x_base + 2 + j, val=UInt8((i + j) % 256))

        # signs[0..31]: sign bytes
        for j in range(32):
            x.unsafe_store(offset=x_base + 34 + j, val=UInt8(0))

        # qh[0..7]: high bits for grid index
        for j in range(8):
            x.unsafe_store(offset=x_base + 66 + j, val=UInt8(0))

        # scales[0..7]: 4-bit scales (nibbles)
        for j in range(8):
            x.unsafe_store(offset=x_base + 74 + j, val=UInt8(0x11))

    # Initialize Q8_K data
    for i in range(nb):
        var y_base = i * 336

        # d = 1.0
        y.unsafe_offset(y_base).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(Scalar[DType.float32](1.0))

        # qs[256]
        for j in range(256):
            y.unsafe_store(offset=y_base + 4 + j, val=UInt8((i + j + 1) % 256))

    var x_ptr = Pointer[UInt8, MutUntrackedOrigin](x)
    var y_ptr = Pointer[UInt8, MutUntrackedOrigin](y)

    # Test NEON version
    print("=== IQ2_S NEON Version ===")
    var result_neon = vec_dot_iq2s_q8k_neon(x_ptr, y_ptr, nb)

    # Warmup
    for _ in range(3):
        result_neon = vec_dot_iq2s_q8k_neon(x_ptr, y_ptr, nb)

    # Benchmark NEON
    var t0 = now_ns()
    for _ in range(10):
        result_neon = vec_dot_iq2s_q8k_neon(x_ptr, y_ptr, nb)
    var t1 = now_ns()

    var elapsed_neon = Float64(t1 - t0) / 1e9 / 10.0
    var flops = Float64(nb) * 512.0
    var gflops_neon = flops / elapsed_neon / 1e9

    print("Result:", result_neon)
    print("Time:", elapsed_neon, "seconds")
    print("GFLOPS:", gflops_neon)

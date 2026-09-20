# Benchmark IQ3_S × Q8_K performance - comparing scalar vs NEON

from src.core.ops.cpu.simd.iq3s_q8k_neon import vec_dot_iq3s_q8k, vec_dot_iq3s_q8k_neon
from src.core.thread_pool import now_ns
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.origin import MutUntrackedOrigin

def main():
    var nb = 1000  # Number of blocks

    # Allocate test data
    var x = unsafe_alloc[UInt8](nb * 110, alignment=64)
    var y = unsafe_alloc[UInt8](nb * 292, alignment=64)

    # Initialize IQ3_S data
    for i in range(nb):
        var x_base = i * 110

        # d = 1.0
        x.unsafe_offset(x_base).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(Scalar[DType.float16](1.0))

        # qs[64]
        for j in range(64):
            x.unsafe_store(offset=x_base + 2 + j, val=UInt8((i + j) % 256))

        # qh[8]
        for j in range(8):
            x.unsafe_store(offset=x_base + 66 + j, val=UInt8(0))

        # signs[32]
        for j in range(32):
            x.unsafe_store(offset=x_base + 74 + j, val=UInt8(0))

        # scales[4]
        for j in range(4):
            x.unsafe_store(offset=x_base + 106 + j, val=UInt8(0x11))

    # Initialize Q8_K data
    for i in range(nb):
        var y_base = i * 292

        # d = 1.0
        y.unsafe_offset(y_base).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(Scalar[DType.float32](1.0))

        # qs[256]
        for j in range(256):
            y.unsafe_store(offset=y_base + 4 + j, val=UInt8((i + j + 1) % 256))

    var x_ptr = Pointer[UInt8, MutUntrackedOrigin](x)
    var y_ptr = Pointer[UInt8, MutUntrackedOrigin](y)

    # Test scalar version
    print("=== Scalar Version ===")
    var result_scalar = vec_dot_iq3s_q8k(x_ptr, y_ptr, nb)

    # Warmup
    for _ in range(3):
        result_scalar = vec_dot_iq3s_q8k(x_ptr, y_ptr, nb)

    # Benchmark scalar
    var t0 = now_ns()
    for _ in range(10):
        result_scalar = vec_dot_iq3s_q8k(x_ptr, y_ptr, nb)
    var t1 = now_ns()

    var elapsed_scalar = Float64(t1 - t0) / 1e9 / 10.0
    var flops = Float64(nb) * 512.0
    var gflops_scalar = flops / elapsed_scalar / 1e9

    print("Result:", result_scalar)
    print("Time:", elapsed_scalar, "seconds")
    print("GFLOPS:", gflops_scalar)

    # Test NEON version
    print("\n=== NEON Version ===")
    var result_neon = vec_dot_iq3s_q8k_neon(x_ptr, y_ptr, nb)

    # Warmup
    for _ in range(3):
        result_neon = vec_dot_iq3s_q8k_neon(x_ptr, y_ptr, nb)

    # Benchmark NEON
    t0 = now_ns()
    for _ in range(10):
        result_neon = vec_dot_iq3s_q8k_neon(x_ptr, y_ptr, nb)
    t1 = now_ns()

    var elapsed_neon = Float64(t1 - t0) / 1e9 / 10.0
    var gflops_neon = flops / elapsed_neon / 1e9

    print("Result:", result_neon)
    print("Time:", elapsed_neon, "seconds")
    print("GFLOPS:", gflops_neon)

    # Comparison
    print("\n=== Comparison ===")
    print("Speedup:", gflops_neon / gflops_scalar, "x")
    print("Results match:", result_scalar == result_neon)

# Benchmark IQ3_S × Q8_K performance

from src.core.ops.cpu.simd.iq3s_q8k_neon import vec_dot_iq3s_q8k
from src.core.thread_pool import now_ns
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.origin import MutUntrackedOrigin

def main():
    var nb = 1000  # Number of blocks

    # Allocate test data
    var x = unsafe_alloc[UInt8](nb * 110, alignment=64)
    var y = unsafe_alloc[UInt8](nb * 336, alignment=64)

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
        var y_base = i * 336

        # d = 1.0
        y.unsafe_offset(y_base).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(Scalar[DType.float32](1.0))

        # qs[256]
        for j in range(256):
            y.unsafe_store(offset=y_base + 4 + j, val=UInt8((i + j + 1) % 256))

    # Warmup
    var result = vec_dot_iq3s_q8k(Pointer[UInt8, MutUntrackedOrigin](x), Pointer[UInt8, MutUntrackedOrigin](y), nb)

    # Benchmark
    var t0 = now_ns()
    for _ in range(10):
        result = vec_dot_iq3s_q8k(Pointer[UInt8, MutUntrackedOrigin](x), Pointer[UInt8, MutUntrackedOrigin](y), nb)
    var t1 = now_ns()

    var elapsed_ns = t1 - t0
    var elapsed_s = Float64(elapsed_ns) / 1e9 / 10.0  # Average over 10 iterations

    # Calculate GFLOPS
    # Each block: 256 multiply-accumulate operations
    # Total FLOPs per block: 512 (256 mul + 256 add)
    # Total FLOPs: nb * 512
    var flops = Float64(nb) * 512.0
    var gflops = flops / elapsed_s / 1e9

    print("Result:", result)
    print("Time:", elapsed_s, "seconds")
    print("GFLOPS:", gflops)
    print("Blocks/second:", Float64(nb) / elapsed_s)

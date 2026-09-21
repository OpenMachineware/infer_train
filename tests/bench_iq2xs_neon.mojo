# Benchmark IQ2_XS kernel performance

from src.core.ops.cpu.simd.iq2xs_q8k_neon import vec_dot_iq2xs_q8k_neon
from src.core.thread_pool import now_ns
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.origin import MutUntrackedOrigin

def main():
    var nb = 1000
    var iterations = 10000
    var warmup = 1000

    # Allocate IQ2_XS blocks (74 bytes each) and Q8_K blocks (292 bytes each)
    var x = unsafe_alloc[UInt8](nb * 74, alignment=64)
    var y = unsafe_alloc[UInt8](nb * 292, alignment=64)

    # Initialize IQ2_XS blocks (matching llama.cpp)
    for i in range(nb):
        var x_base = i * 74
        # d = 1.0 (FP16) - manual store due to Mojo FP16 bug
        # FP16 1.0 = 0x3C00 (little-endian: 0x00, 0x3C)
        x.unsafe_store(offset=x_base, val=UInt8(0x00))
        x.unsafe_store(offset=x_base + 1, val=UInt8(0x3C))
        # qs[32] = uint16 array (64 bytes)
        for j in range(32):
            var val = UInt16((i + j) % 256)
            # Store as little-endian uint16
            x.unsafe_store(offset=x_base + 2 + j * 2, val=UInt8(val & 0xFF))
            x.unsafe_store(offset=x_base + 2 + j * 2 + 1, val=UInt8(val >> 8))
        # scales[8]
        for j in range(8):
            x.unsafe_store(offset=x_base + 66 + j, val=UInt8(0x11))

    # Initialize Q8_K blocks (matching llama.cpp)
    for i in range(nb):
        var y_base = i * 292
        # d = 1.0 (FP32)
        y.unsafe_offset(y_base).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(Scalar[DType.float32](1.0))
        # qs[256] = signed int8 values
        for j in range(256):
            var val = (j + 1) % 256
            if val >= 128:
                val = val - 256
            y.unsafe_store(offset=y_base + 4 + j, val=UInt8(val & 0xFF))

    var x_ptr = Pointer[UInt8, MutUntrackedOrigin](x)
    var y_ptr = Pointer[UInt8, MutUntrackedOrigin](y)

    # Warmup
    for _ in range(warmup):
        var _ = vec_dot_iq2xs_q8k_neon(x_ptr, y_ptr, nb)

    # Benchmark
    var start = now_ns()
    var result = Float32(0)
    for _ in range(iterations):
        result = Float32(0)  # Reset each iteration (matching llama.cpp)
        result += vec_dot_iq2xs_q8k_neon(x_ptr, y_ptr, nb)
    var elapsed = now_ns() - start

    var elapsed_ns = Float64(elapsed)
    var elapsed_ms = elapsed_ns / 1e6
    var flops = 2.0 * Float64(nb) * 256.0 * Float64(iterations) * 1e9 / elapsed_ns
    var gflops = flops / 1e9

    print("IQ2_XS Mojo benchmark (nb=", nb, ")")
    print("  Result: ", result)
    print("  Time: ", elapsed_ms, " ms")
    print("  GFLOPS: ", gflops)

# Mojo test calling REAL IQ1_S kernel with instrumentation
# Must compare with llama.cpp per-step timing

from src.core.ops.cpu.simd.iq1s_q8k_neon import vec_dot_iq1s_q8k_neon
from src.core.thread_pool import now_ns
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.origin import MutUntrackedOrigin

def main():
    var nb = 1000  # 1000 blocks
    var iterations = 10000
    var warmup = 1000

    # Allocate test data
    # IQ1_S: 50 bytes/block
    var x = unsafe_alloc[UInt8](nb * 50, alignment=64)
    # Q8_K: 336 bytes/block
    var y = unsafe_alloc[UInt8](nb * 336, alignment=64)

    # Initialize with simple pattern
    for i in range(nb):
        var x_base = i * 50
        var y_base = i * 336

        # IQ1_S block
        # d = 1.0 (FP16)
        x.unsafe_offset(x_base).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(Scalar[DType.float16](1.0))

        # qs[32]: grid indices
        for j in range(32):
            x.unsafe_store(offset=x_base + 2 + j, val=UInt8(j % 256))

        # qh[8]: high bits + scale + sign
        for j in range(8):
            x.unsafe_offset(x_base + 34 + j * 2).unsafe_bitcast[Scalar[DType.uint16]]().unsafe_store(Scalar[DType.uint16](0))

        # Q8_K block
        # d = 1.0 (FP32)
        y.unsafe_offset(y_base).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(Scalar[DType.float32](1.0))

        # qs[256]: int8 values (same as C: (int8_t)((j + 1) % 256))
        for j in range(256):
            var val = (j + 1) % 256
            # Convert to int8_t: values 128-255 become negative
            var q8_val = Int8(val if val < 128 else val - 256)
            y.unsafe_offset(y_base + 4 + j).unsafe_bitcast[Scalar[DType.int8]]().unsafe_store(Scalar[DType.int8](q8_val))

        # bsums[16]: sum of quants (same as C)
        for j in range(16):
            var sum: Int = 0
            for k in range(16):
                sum += Int(y.unsafe_offset(y_base + 4 + j * 16 + k).unsafe_bitcast[Scalar[DType.int8]]().unsafe_load[width=1](offset=0))
            y.unsafe_offset(y_base + 260 + j * 2).unsafe_bitcast[Scalar[DType.int16]]().unsafe_store(Scalar[DType.int16](Int16(sum)))

    var result = Float32(0)

    # Warmup
    for iter in range(warmup):
        result = 0
        for i in range(nb):
            result += vec_dot_iq1s_q8k_neon(
                Pointer[UInt8, MutUntrackedOrigin](x.unsafe_offset(i * 50)),
                Pointer[UInt8, MutUntrackedOrigin](y.unsafe_offset(i * 336)),
                1
            )

    # Instrumented run
    print("Instrumentation results (calling REAL Mojo kernel):")
    print("nb=", nb, ", iterations=", iterations)

    var t0 = now_ns()
    var total_kernel_ns: Int = 0
    var total_loop_ns: Int = 0

    for iter in range(iterations):
        var t1 = now_ns()

        result = 0
        for i in range(nb):
            var t2 = now_ns()

            result += vec_dot_iq1s_q8k_neon(
                Pointer[UInt8, MutUntrackedOrigin](x.unsafe_offset(i * 50)),
                Pointer[UInt8, MutUntrackedOrigin](y.unsafe_offset(i * 336)),
                1
            )

            var t3 = now_ns()
            total_kernel_ns += (t3 - t2)

        var t4 = now_ns()
        total_loop_ns += (t4 - t1)

    var t5 = now_ns()
    var total_ns = t5 - t0

    print("Total time:           ", total_ns / 1_000_000, " ms")
    print("Total loop time:      ", total_loop_ns / 1_000_000, " ms (per iteration: ", total_loop_ns / iterations / 1_000, " us)")
    print("Total kernel time:    ", total_kernel_ns / 1_000_000, " ms (per block: ", total_kernel_ns / (iterations * nb), " ns)")
    print("Overhead (loop - kernel): ", (total_loop_ns - total_kernel_ns) / 1_000_000, " ms")
    var total_flops = Float64(2) * Float64(nb) * 256.0 * Float64(iterations)
    var kernel_seconds = Float64(total_kernel_ns) / 1e9
    print("Average kernel GFLOPS: ", total_flops / kernel_seconds / 1e9)

    print()
    print("Final result: ", result)

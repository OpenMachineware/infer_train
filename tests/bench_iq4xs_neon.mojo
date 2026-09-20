# Benchmark IQ4_XS × Q8_K SIMD version
from src.core.ops.cpu.simd.iq4xs_q8k_neon import vec_dot_iq4xs_q8k
from src.core.thread_pool import now_ns
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

def main():
    var nb = 16
    var iterations = 100000
    var warmup = 10000

    var iq4xs_block_size = 136
    var q8k_block_size = 336

    # Allocate test data
    var iq4_mem = unsafe_alloc[UInt8](nb * iq4xs_block_size, alignment=64)
    var q8_mem = unsafe_alloc[UInt8](nb * q8k_block_size, alignment=64)
    var iq4_ptr = Pointer[UInt8, MutUntrackedOrigin](iq4_mem)
    var q8_ptr = Pointer[UInt8, MutUntrackedOrigin](q8_mem)

    # Initialize IQ4_XS data
    for i in range(nb):
        var block_offset = i * iq4xs_block_size

        # d: fp16 1.0
        iq4_ptr.unsafe_offset(block_offset).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(
            Scalar[DType.float16](1.0)
        )

        # scales_h: uint16 0x8888
        iq4_ptr.unsafe_offset(block_offset + 2).unsafe_store(UInt8(0x88))
        iq4_ptr.unsafe_offset(block_offset + 3).unsafe_store(UInt8(0x88))

        # scales_l[4]
        for j in range(4):
            iq4_ptr.unsafe_offset(block_offset + 4 + j).unsafe_store(UInt8(0x28))

        # qs[128]
        for j in range(128):
            iq4_ptr.unsafe_offset(block_offset + 8 + j).unsafe_store(UInt8(j % 256))

    # Initialize Q8_K data
    for i in range(nb):
        var block_offset = i * q8k_block_size

        # d: float32 1.0
        q8_ptr.unsafe_offset(block_offset).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(
            Scalar[DType.float32](1.0)
        )

        # qs: 256 int8
        for j in range(256):
            q8_ptr.unsafe_offset(block_offset + 4 + j).unsafe_store(UInt8(64))

        # bsums: 16 int16
        for j in range(16):
            q8_ptr.unsafe_offset(block_offset + 260 + j * 2).unsafe_bitcast[Scalar[DType.int16]]().unsafe_store(
                Scalar[DType.int16](1024)
            )

    # Test correctness first
    var result_scalar = Float32(0)
    for i in range(nb):
        result_scalar += vec_dot_iq4xs_q8k(iq4_ptr.unsafe_offset(i * iq4xs_block_size),
                                           q8_ptr.unsafe_offset(i * q8k_block_size), 1)
    print("Correctness check (per-block):", result_scalar)

    # Warmup
    var result = Float32(0)
    for _ in range(warmup):
        var sum = Float32(0)
        for i in range(nb):
            sum += vec_dot_iq4xs_q8k(iq4_ptr.unsafe_offset(i * iq4xs_block_size),
                                     q8_ptr.unsafe_offset(i * q8k_block_size), 1)
        result = sum

    # Benchmark
    var start = now_ns()
    for _ in range(iterations):
        var sum = Float32(0)
        for i in range(nb):
            sum += vec_dot_iq4xs_q8k(iq4_ptr.unsafe_offset(i * iq4xs_block_size),
                                     q8_ptr.unsafe_offset(i * q8k_block_size), 1)
        result = sum
    var end = now_ns()

    var elapsed_ns = Float64(end - start)
    var elapsed_s = elapsed_ns / 1_000_000_000.0
    var flops = 2.0 * Float64(nb * 256) * Float64(iterations)
    var gflops = flops / elapsed_s / 1_000_000_000.0

    print("Mojo IQ4_XS SIMD GFLOPS:", gflops)
    print("Result:", result)

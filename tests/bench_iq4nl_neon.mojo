# Benchmark IQ4_NL × Q8_0 SIMD version
from src.core.ops.cpu.simd.iq4nl_q80_neon import vec_dot_iq4nl_q80_neon
from src.core.thread_pool import now_ns
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

def main():
    var nb = 8000  # Match llama.cpp
    var iterations = 10000
    var warmup = 1000

    var iq4nl_block_size = 18
    var q80_block_size = 34

    # Allocate test data
    var iq4_mem = unsafe_alloc[UInt8](nb * iq4nl_block_size, alignment=64)
    var q8_mem = unsafe_alloc[UInt8](nb * q80_block_size, alignment=64)
    var iq4_ptr = Pointer[UInt8, MutUntrackedOrigin](iq4_mem)
    var q8_ptr = Pointer[UInt8, MutUntrackedOrigin](q8_mem)

    # Initialize IQ4_NL data
    for i in range(nb):
        var block_offset = i * iq4nl_block_size

        # d = 1.0 (FP16) - manual store
        # FP16 1.0 = 0x3C00 (little-endian: 0x00, 0x3C)
        iq4_ptr.unsafe_store(offset=block_offset, val=UInt8(0x00))
        iq4_ptr.unsafe_store(offset=block_offset + 1, val=UInt8(0x3C))

        # qs[16]: packed 4-bit values
        for j in range(16):
            iq4_ptr.unsafe_offset(block_offset + 2 + j).unsafe_store(UInt8((i + j) % 256))

    # Initialize Q8_0 data
    for i in range(nb):
        var block_offset = i * q80_block_size

        # d = 1.0 (FP16) - manual store
        # FP16 1.0 = 0x3C00 (little-endian: 0x00, 0x3C)
        # Note: Q8_0 uses FP16, not FP32 like Q8_K
        q8_ptr.unsafe_store(offset=block_offset, val=UInt8(0x00))
        q8_ptr.unsafe_store(offset=block_offset + 1, val=UInt8(0x3C))

        # qs: 32 int8
        for j in range(32):
            var val = (j + 1) % 256
            if val >= 128:
                val = val - 256
            q8_ptr.unsafe_offset(block_offset + 2 + j).unsafe_store(UInt8(val & 0xFF))

    # Warmup
    var result = Float32(0)
    for _ in range(warmup):
        result = Float32(0)
        for i in range(nb):
            result = result + vec_dot_iq4nl_q80_neon(iq4_ptr.unsafe_offset(i * iq4nl_block_size),
                                                     q8_ptr.unsafe_offset(i * q80_block_size), 1)

    # Benchmark
    var start = now_ns()
    for _ in range(iterations):
        result = Float32(0)
        for i in range(nb):
            result = result + vec_dot_iq4nl_q80_neon(iq4_ptr.unsafe_offset(i * iq4nl_block_size),
                                                     q8_ptr.unsafe_offset(i * q80_block_size), 1)
    var end = now_ns()

    var elapsed_ns = Float64(end - start)
    var elapsed_ms = elapsed_ns / 1e6
    var flops = 2.0 * Float64(nb) * 32.0 * Float64(iterations)
    var gflops = flops / elapsed_ns

    print("IQ4_NL Mojo benchmark (nb=", nb, ", iterations=", iterations, ")")
    print("  Result: ", result)
    print("  Time: ", elapsed_ms, " ms")
    print("  GFLOPS: ", gflops)

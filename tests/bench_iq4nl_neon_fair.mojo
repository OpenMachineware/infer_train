# Fair Benchmark IQ4_NL × Q8_0 - Process all blocks in one call
from src.core.ops.cpu.simd.iq4nl_q80_neon import vec_dot_iq4nl_q80_neon
from std.time import now
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

def main():
    var nb = 250  # Match llama.cpp fair benchmark (250 blocks per call)
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
        iq4_ptr.unsafe_store(offset=block_offset, val=UInt8(0x00))
        iq4_ptr.unsafe_store(offset=block_offset + 1, val=UInt8(0x3C))
        for j in range(16):
            iq4_ptr.unsafe_offset(block_offset + 2 + j).unsafe_store(UInt8((i + j) % 256))

    # Initialize Q8_0 data
    for i in range(nb):
        var block_offset = i * q80_block_size
        q8_ptr.unsafe_store(offset=block_offset, val=UInt8(0x00))
        q8_ptr.unsafe_store(offset=block_offset + 1, val=UInt8(0x3C))
        for j in range(32):
            var val = (j + 1) % 256
            if val >= 128:
                val = val - 256
            q8_ptr.unsafe_offset(block_offset + 2 + j).unsafe_store(UInt8(val & 0xFF))

    # Warmup
    var result = Float32(0)
    for _ in range(warmup):
        result = vec_dot_iq4nl_q80_neon(iq4_ptr, q8_ptr, nb)

    # Benchmark
    var start = now()
    for _ in range(iterations):
        result = vec_dot_iq4nl_q80_neon(iq4_ptr, q8_ptr, nb)
    var end = now()

    var elapsed_ns = Float64((end - start).total_nanoseconds())
    var elapsed_ms = elapsed_ns / 1e6
    var flops = 2.0 * Float64(nb) * 32.0 * Float64(iterations)
    var gflops = flops / elapsed_ns

    print("IQ4_NL Mojo fair benchmark (nb=", nb, " blocks per call, iterations=", iterations, ")")
    print("  Elements: ", nb * 32)
    print("  Result: ", result)
    print("  Time: ", elapsed_ms, " ms")
    print("  GFLOPS: ", gflops)

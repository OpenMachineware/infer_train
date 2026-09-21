# Benchmark IQ1_M × Q8_K SIMD version
from src.core.ops.cpu.simd.iq1m_q8k_neon import vec_dot_iq1m_q8k_neon
from src.core.thread_pool import now_ns
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

def main():
    var nb = 1000  # Match llama.cpp
    var iterations = 10000
    var warmup = 1000

    var iq1m_block_size = 56
    var q8k_block_size = 292  # Fixed: Q8_K is 292 bytes, not 336

    # Allocate test data
    var iq1_mem = unsafe_alloc[UInt8](nb * iq1m_block_size, alignment=64)
    var q8_mem = unsafe_alloc[UInt8](nb * q8k_block_size, alignment=64)
    var iq1_ptr = Pointer[UInt8, MutUntrackedOrigin](iq1_mem)
    var q8_ptr = Pointer[UInt8, MutUntrackedOrigin](q8_mem)

    # Initialize IQ1_M data (matching llama.cpp)
    for i in range(nb):
        var block_offset = i * iq1m_block_size

        # qs[32]: grid indices - match llama.cpp
        for j in range(32):
            iq1_ptr.unsafe_offset(block_offset + j).unsafe_store(UInt8(j % 256))

        # qh[16]: high bits + shift - match llama.cpp
        for j in range(16):
            iq1_ptr.unsafe_offset(block_offset + 32 + j).unsafe_store(UInt8(0))

        # scales[8]: packed 4-bit scales - match llama.cpp
        for j in range(8):
            iq1_ptr.unsafe_offset(block_offset + 48 + j).unsafe_store(UInt8(0x11))  # Scale = 1

    # Initialize Q8_K data (matching llama.cpp exactly)
    for i in range(nb):
        var block_offset = i * q8k_block_size

        # d: float32 1.0
        q8_ptr.unsafe_offset(block_offset).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(
            Scalar[DType.float32](1.0)
        )

        # qs: 256 int8 - match llama.cpp: y[i].qs[j] = (int8_t)((j + 1) % 256)
        for j in range(256):
            var val = (j + 1) % 256
            q8_ptr.unsafe_offset(block_offset + 4 + j).unsafe_store(UInt8(val))

        # bsums: 16 int16 - match llama.cpp: calculate sum for each group
        for j in range(16):
            var sum = 0
            for k in range(16):
                var idx = j * 16 + k
                var val = (idx + 1) % 256
                var signed_val = val if val < 128 else val - 256
                sum += signed_val
            q8_ptr.unsafe_offset(block_offset + 260 + j * 2).unsafe_bitcast[Scalar[DType.int16]]().unsafe_store(
                Scalar[DType.int16](Int16(sum))
            )

    # Warmup
    var result = Float32(0)
    for _ in range(warmup):
        var sum = Float32(0)
        for i in range(nb):
            sum += vec_dot_iq1m_q8k_neon(iq1_ptr.unsafe_offset(i * iq1m_block_size),
                                         q8_ptr.unsafe_offset(i * q8k_block_size), 1)
        result = sum

    # Benchmark
    var start = now_ns()
    for _ in range(iterations):
        var sum = Float32(0)
        for i in range(nb):
            sum += vec_dot_iq1m_q8k_neon(iq1_ptr.unsafe_offset(i * iq1m_block_size),
                                         q8_ptr.unsafe_offset(i * q8k_block_size), 1)
        result = sum
    var end = now_ns()

    var elapsed_ns = Float64(end - start)
    var elapsed_s = elapsed_ns / 1_000_000_000.0
    var flops = 2.0 * Float64(nb * 256) * Float64(iterations)
    var gflops = flops / elapsed_s / 1_000_000_000.0

    print("Mojo IQ1_M SIMD GFLOPS:", gflops)
    print("Result:", result)

# Mojo Q3_K test - matching llama.cpp test exactly
# Following llama_cpp测试模板.md methodology
from src.core.ops.cpu.simd.simd_neon import vec_dot_q3_k_q8_k
from std.time import perf_counter_ns
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

def main():
    var nb = 16
    var warmup = 10000
    var iterations = 100000

    # Q3_K block: hmask(32) + qs(64) + scales(12) + d(2) = 110 bytes
    var q3_block_size = 110
    # Q8_K block: d(4) + qs(256) + bsums(32) = 292 bytes
    var q8_block_size = 292

    # Allocate test data
    var q3_mem = unsafe_alloc[UInt8](nb * q3_block_size, alignment=64)
    var q8_mem = unsafe_alloc[UInt8](nb * q8_block_size, alignment=64)
    var q3_ptr = Pointer[UInt8, MutUntrackedOrigin](q3_mem)
    var q8_ptr = Pointer[UInt8, MutUntrackedOrigin](q8_mem)

    # Initialize Q3_K data - EXACTLY matching the C test
    for i in range(nb):
        var block_offset = i * q3_block_size

        # hmask: mix of 0 and 1 bits (32 bytes at offset 0)
        for j in range(32):
            var val = UInt8(0xFF if j % 2 == 0 else 0x00)
            q3_ptr.unsafe_offset(block_offset + j).unsafe_store(val)

        # qs: low 2-bit values (64 bytes at offset 32)
        for j in range(64):
            var val = UInt8((j * 4) % 256)
            q3_ptr.unsafe_offset(block_offset + 32 + j).unsafe_store(val)

        # scales: packed 6-bit values (12 bytes at offset 96)
        for j in range(12):
            var val = UInt8(40 + (j % 8))
            q3_ptr.unsafe_offset(block_offset + 96 + j).unsafe_store(val)

        # d: 1.0f in fp16 format (2 bytes at offset 108)
        q3_ptr.unsafe_offset(block_offset + 108).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(Scalar[DType.float16](1.0))

    # Initialize Q8_K data - EXACTLY matching the C test
    for i in range(nb):
        var block_offset = i * q8_block_size

        # d: 1.0f (4 bytes at offset 0)
        q8_ptr.unsafe_offset(block_offset).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(Scalar[DType.float32](1.0))

        # qs: int8 values (256 bytes at offset 4)
        for j in range(256):
            q8_ptr.unsafe_offset(block_offset + 4 + j).unsafe_store(UInt8(64))

        # bsums: block sums (32 bytes at offset 260)
        for j in range(16):
            q8_ptr.unsafe_offset(block_offset + 260 + j * 2).unsafe_bitcast[Scalar[DType.int16]]().unsafe_store(Scalar[DType.int16](1024))

    # Warmup runs
    var result = Float32(0)
    for _ in range(warmup):
        var sum = Float32(0)
        for i in range(nb):
            sum += vec_dot_q3_k_q8_k(
                q3_ptr.unsafe_offset(i * q3_block_size),
                q8_ptr.unsafe_offset(i * q8_block_size),
            )
        result = sum

    # Benchmark
    var start = perf_counter_ns()
    result = Float32(0)
    for _ in range(iterations):
        var sum = Float32(0)
        for i in range(nb):
            sum += vec_dot_q3_k_q8_k(
                q3_ptr.unsafe_offset(i * q3_block_size),
                q8_ptr.unsafe_offset(i * q8_block_size),
            )
        result = sum
    var end = perf_counter_ns()

    var elapsed_ns = Float64(end - start)
    var elapsed_s = elapsed_ns / 1_000_000_000.0
    var n = nb * 256
    var flops = 2.0 * Float64(n) * Float64(iterations)
    var gflops = flops / elapsed_s / 1_000_000_000.0

    print("Mojo Q3_K GFLOPS:", gflops)
    print("Result:", result)

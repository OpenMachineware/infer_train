# Mojo Q4_K test - same data and method as llama.cpp test
from src.core.ops.cpu.simd.q4k_q8k_dot import vec_dot_q4_k_q8_k
from src.core.thread_pool import now_ns
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

def main():
    var nb = 16
    var iterations = 100000
    var warmup = 10000

    var q4_block_size = 144
    var q8_block_size = 292

    # Allocate test data
    var q4_mem = unsafe_alloc[UInt8](nb * q4_block_size, alignment=64)
    var q8_mem = unsafe_alloc[UInt8](nb * q8_block_size, alignment=64)
    var q4_ptr = Pointer[UInt8, MutUntrackedOrigin](q4_mem)
    var q8_ptr = Pointer[UInt8, MutUntrackedOrigin](q8_mem)

    # Initialize Q4_K data (matching C test exactly)
    for i in range(nb):
        var block_offset = i * q4_block_size

        # d: fp16 1.0 (2 bytes at offset 0)
        q4_ptr.unsafe_offset(block_offset).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(
            Scalar[DType.float16](1.0)
        )

        # dmin: fp16 0.0 (2 bytes at offset 2)
        q4_ptr.unsafe_offset(block_offset + 2).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(
            Scalar[DType.float16](0.0)
        )

        # scales: 12 bytes at offset 4
        for j in range(12):
            var val = UInt8(40 + (j % 8))
            q4_ptr.unsafe_offset(block_offset + 4 + j).unsafe_store(val)

        # qs: 128 bytes at offset 16
        for j in range(128):
            var val = UInt8((j * 2) % 256)
            q4_ptr.unsafe_offset(block_offset + 16 + j).unsafe_store(val)

    # Initialize Q8_K data (matching C test exactly)
    for i in range(nb):
        var block_offset = i * q8_block_size

        # d: float32 1.0 (4 bytes at offset 0)
        q8_ptr.unsafe_offset(block_offset).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(
            Scalar[DType.float32](1.0)
        )

        # qs: 256 int8 at offset 4
        for j in range(256):
            q8_ptr.unsafe_offset(block_offset + 4 + j).unsafe_store(UInt8(64))

        # bsums: 16 int16 at offset 260
        for j in range(16):
            q8_ptr.unsafe_offset(block_offset + 260 + j * 2).unsafe_bitcast[Scalar[DType.int16]]().unsafe_store(
                Scalar[DType.int16](1024)
            )

    # Warmup runs
    var result = Float32(0)
    for _ in range(warmup):
        var sum = Float32(0)
        for i in range(nb):
            sum += vec_dot_q4_k_q8_k(
                q4_ptr.unsafe_offset(i * q4_block_size),
                q8_ptr.unsafe_offset(i * q8_block_size)
            )
        result = sum

    # Benchmark
    var start = now_ns()
    for _ in range(iterations):
        var sum = Float32(0)
        for i in range(nb):
            sum += vec_dot_q4_k_q8_k(
                q4_ptr.unsafe_offset(i * q4_block_size),
                q8_ptr.unsafe_offset(i * q8_block_size)
            )
        result = sum
    var end = now_ns()

    var elapsed_ns = Float64(end - start)
    var elapsed_s = elapsed_ns / 1_000_000_000.0
    var flops = 2.0 * Float64(nb * 256) * Float64(iterations)
    var gflops = flops / elapsed_s / 1_000_000_000.0

    print("Mojo Q4_K GFLOPS:", gflops)
    print("Result:", result)

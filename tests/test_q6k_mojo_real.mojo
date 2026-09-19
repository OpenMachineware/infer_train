# Mojo Q6_K test - same data and method as llama.cpp test
from src.core.ops.cpu.simd.q6k_q8k_dot import vec_dot_q6_k_q8_k
from src.core.thread_pool import now_ns
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

def main():
    var nb = 16
    var iterations = 100000
    var warmup = 10000

    var q6_block_size = 210
    var q8_block_size = 292

    # Allocate test data
    var q6_mem = unsafe_alloc[UInt8](nb * q6_block_size, alignment=64)
    var q8_mem = unsafe_alloc[UInt8](nb * q8_block_size, alignment=64)
    var q6_ptr = Pointer[UInt8, MutUntrackedOrigin](q6_mem)
    var q8_ptr = Pointer[UInt8, MutUntrackedOrigin](q8_mem)

    # Initialize data (same as C test)
    for i in range(nb):
        var block_offset = i * q6_block_size
        # Initialize ql (128 bytes)
        for j in range(128):
            q6_ptr.unsafe_offset(block_offset + j).unsafe_store(UInt8((j * 7) % 256))
        # Initialize qh (64 bytes)
        for j in range(64):
            q6_ptr.unsafe_offset(block_offset + 128 + j).unsafe_store(UInt8((j * 11) % 256))
        # Initialize scales (16 bytes)
        for j in range(16):
            q6_ptr.unsafe_offset(block_offset + 192 + j).unsafe_store(UInt8((j * 13 + 64) % 256))
        # Initialize d (fp16)
        q6_ptr.unsafe_offset(block_offset + 208).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(Scalar[DType.float16](1.0))

    for i in range(nb):
        var block_offset = i * q8_block_size
        # Initialize d
        q8_ptr.unsafe_offset(block_offset).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(Scalar[DType.float32](1.0))
        # Initialize qs
        for j in range(256):
            q8_ptr.unsafe_offset(block_offset + 4 + j).unsafe_store(UInt8(64))
        # Initialize bsums
        for j in range(16):
            q8_ptr.unsafe_offset(block_offset + 260 + j * 2).unsafe_bitcast[Scalar[DType.int16]]().unsafe_store(Scalar[DType.int16](1024))

    print("=== Mojo Q6_K Performance ===")
    print("Blocks:", nb, ", Warmup:", warmup, ", Iterations:", iterations)
    print("")

    # Warmup
    for _ in range(warmup):
        var sum = Float32(0)
        for i in range(nb):
            sum += vec_dot_q6_k_q8_k(
                q6_ptr.unsafe_offset(i * q6_block_size),
                q8_ptr.unsafe_offset(i * q8_block_size)
            )

    # Run benchmark
    var start = now_ns()
    var result = Float32(0)
    for _ in range(iterations):
        var sum = Float32(0)
        for i in range(nb):
            sum += vec_dot_q6_k_q8_k(
                q6_ptr.unsafe_offset(i * q6_block_size),
                q8_ptr.unsafe_offset(i * q8_block_size)
            )
        result = sum
    var end = now_ns()

    var elapsed_ns = Float64(end - start)
    var elapsed_ms = elapsed_ns / 1_000_000.0
    var flops = 2.0 * Float64(nb * 256) * Float64(iterations)
    var gflops = flops / (elapsed_ms / 1000.0) / 1_000_000_000.0

    print("Time:   ", elapsed_ms, " ms")
    print("Result: ", result)
    print("GFLOPS: ", gflops)

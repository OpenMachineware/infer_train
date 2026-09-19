# Mojo Q6_K test - parameterized by nb
from src.core.ops.cpu.simd.q6k_q8k_dot import vec_dot_q6_k_q8_k
from src.core.thread_pool import now_ns
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin

def main():
    # Parse nb from command line (default 16)
    var nb = 16
    # Note: Mojo doesn't have easy CLI parsing, so we test specific sizes

    var q6_block_size = 210
    var q8_block_size = 292

    var q6_mem = unsafe_alloc[UInt8](nb * q6_block_size, alignment=64)
    var q8_mem = unsafe_alloc[UInt8](nb * q8_block_size, alignment=64)
    var q6_ptr = Pointer[UInt8, MutUntrackedOrigin](q6_mem)
    var q8_ptr = Pointer[UInt8, MutUntrackedOrigin](q8_mem)

    # Initialize data (same as C test with seed 12345)
    var rng = 12345
    for i in range(nb):
        var block_offset = i * q6_block_size
        for j in range(128):
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            q6_ptr.unsafe_offset(block_offset + j).unsafe_store(UInt8(rng % 256))
        for j in range(64):
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            q6_ptr.unsafe_offset(block_offset + 128 + j).unsafe_store(UInt8(rng % 256))
        for j in range(16):
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            q6_ptr.unsafe_offset(block_offset + 192 + j).unsafe_store(UInt8((rng % 256) - 128))
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        q6_ptr.unsafe_offset(block_offset + 208).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(Scalar[DType.float16](Float16(rng % 1000) / 100.0))

    for i in range(nb):
        var block_offset = i * q8_block_size
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        q8_ptr.unsafe_offset(block_offset).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(Scalar[DType.float32](Float32(rng % 1000) / 100.0))
        for j in range(256):
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            q8_ptr.unsafe_offset(block_offset + 4 + j).unsafe_store(UInt8((rng % 256) - 128))
        for j in range(16):
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            q8_ptr.unsafe_offset(block_offset + 260 + j * 2).unsafe_bitcast[Scalar[DType.int16]]().unsafe_store(Scalar[DType.int16](Int16((rng % 65536) - 32768)))

    var iterations = 100000
    var warmup = 10000

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

    print("Blocks:", nb, ", Warmup:", warmup, ", Iterations:", iterations)
    print("Time:   ", elapsed_ms, " ms")
    print("Result: ", result)
    print("GFLOPS: ", gflops)

# Benchmark KV cache dequantization speed
# Test all 5 formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0

from src.core.ops.attention.kv_cache_simd import (
    dequantize_row_q4_0_neon, dequantize_row_q4_1_neon,
    dequantize_row_q5_0_neon, dequantize_row_q5_1_neon,
    dequantize_row_q8_0_neon, dequantize_row_q8_0_f32_neon,
)
from std.memory.alloc import unsafe_alloc
from std.time import perf_counter_ns

comptime KV_QK = 32
comptime WARMUP = 1000
comptime ITERATIONS = 10000


def bench_q4_0():
    """Benchmark Q4_0 dequantization."""
    var n = 4096
    var nb = n // KV_QK

    var src = unsafe_alloc[UInt8](nb * 18 + 16, alignment=64)
    var dst = unsafe_alloc[Scalar[DType.float16]](n, alignment=64)

    for i in range(nb):
        src.unsafe_store(i * 18, UInt8(0x00))
        src.unsafe_store(i * 18 + 1, UInt8(0x38))
        for j in range(16):
            src.unsafe_store(i * 18 + 2 + j, UInt8((j * 17) % 256))

    for _ in range(WARMUP):
        dequantize_row_q4_0_neon(src, dst, n)

    # Force memory access (like volatile in C)
    var sink = unsafe_alloc[Float64](1, alignment=64)
    sink.unsafe_store(0, 0.0)

    var start = perf_counter_ns()
    for _ in range(ITERATIONS):
        dequantize_row_q4_0_neon(src, dst, n)
        var cur = sink.unsafe_load[width=1](offset=0)
        sink.unsafe_store(0, cur + Float64(dst.unsafe_load[width=1](offset=0)))
    var end = perf_counter_ns()

    # Use sink to prevent optimization
    print("Q4_0 checksum: ", sink.unsafe_load[width=1](offset=0))

    var elapsed_ns = Float64(end - start)
    var throughput = Float64(n * ITERATIONS) / (elapsed_ns / 1e6) / 1000
    var per_block_ns = elapsed_ns / Float64(nb * ITERATIONS)

    print("Q4_0: ", throughput, " M/s, ", per_block_ns, " ns/block")


def bench_q4_1():
    """Benchmark Q4_1 dequantization."""
    var n = 4096
    var nb = n // KV_QK

    var src = unsafe_alloc[UInt8](nb * 20 + 16, alignment=64)
    var dst = unsafe_alloc[Scalar[DType.float16]](n, alignment=64)

    for i in range(nb):
        src.unsafe_store(i * 20, UInt8(0x00))      # d
        src.unsafe_store(i * 20 + 1, UInt8(0x38))
        src.unsafe_store(i * 20 + 2, UInt8(0x00))  # m
        src.unsafe_store(i * 20 + 3, UInt8(0x38))
        for j in range(16):
            src.unsafe_store(i * 20 + 4 + j, UInt8((j * 17) % 256))

    for _ in range(WARMUP):
        dequantize_row_q4_1_neon(src, dst, n)

    var sink = unsafe_alloc[Float64](1, alignment=64)
    sink.unsafe_store(0, 0.0)

    var start = perf_counter_ns()
    for _ in range(ITERATIONS):
        dequantize_row_q4_1_neon(src, dst, n)
        var cur = sink.unsafe_load[width=1](offset=0)
        sink.unsafe_store(0, cur + Float64(dst.unsafe_load[width=1](offset=0)))
    var end = perf_counter_ns()

    print("Q4_1 checksum: ", sink.unsafe_load[width=1](offset=0))

    var elapsed_ns = Float64(end - start)
    var throughput = Float64(n * ITERATIONS) / (elapsed_ns / 1e6) / 1000
    var per_block_ns = elapsed_ns / Float64(nb * ITERATIONS)

    print("Q4_1: ", throughput, " M/s, ", per_block_ns, " ns/block")


def bench_q5_0():
    """Benchmark Q5_0 dequantization."""
    var n = 4096
    var nb = n // KV_QK

    var src = unsafe_alloc[UInt8](nb * 22 + 16, alignment=64)
    var dst = unsafe_alloc[Scalar[DType.float16]](n, alignment=64)

    for i in range(nb):
        src.unsafe_store(i * 22, UInt8(0x00))      # d
        src.unsafe_store(i * 22 + 1, UInt8(0x38))
        for j in range(4):
            src.unsafe_store(i * 22 + 2 + j, UInt8(j * 17))  # qh
        for j in range(16):
            src.unsafe_store(i * 22 + 6 + j, UInt8((j * 13) % 256))  # qs

    for _ in range(WARMUP):
        dequantize_row_q5_0_neon(src, dst, n)

    var sink = unsafe_alloc[Float64](1, alignment=64)
    sink.unsafe_store(0, 0.0)

    var start = perf_counter_ns()
    for _ in range(ITERATIONS):
        dequantize_row_q5_0_neon(src, dst, n)
        var cur = sink.unsafe_load[width=1](offset=0)
        sink.unsafe_store(0, cur + Float64(dst.unsafe_load[width=1](offset=0)))
    var end = perf_counter_ns()

    print("Q5_0 checksum: ", sink.unsafe_load[width=1](offset=0))

    var elapsed_ns = Float64(end - start)
    var throughput = Float64(n * ITERATIONS) / (elapsed_ns / 1e6) / 1000
    var per_block_ns = elapsed_ns / Float64(nb * ITERATIONS)

    print("Q5_0: ", throughput, " M/s, ", per_block_ns, " ns/block")


def bench_q5_1():
    """Benchmark Q5_1 dequantization."""
    var n = 4096
    var nb = n // KV_QK

    var src = unsafe_alloc[UInt8](nb * 24 + 16, alignment=64)
    var dst = unsafe_alloc[Scalar[DType.float16]](n, alignment=64)

    for i in range(nb):
        src.unsafe_store(i * 24, UInt8(0x00))      # d
        src.unsafe_store(i * 24 + 1, UInt8(0x38))
        src.unsafe_store(i * 24 + 2, UInt8(0x00))  # m
        src.unsafe_store(i * 24 + 3, UInt8(0x38))
        for j in range(4):
            src.unsafe_store(i * 24 + 4 + j, UInt8(j * 17))  # qh
        for j in range(16):
            src.unsafe_store(i * 24 + 8 + j, UInt8((j * 13) % 256))  # qs

    for _ in range(WARMUP):
        dequantize_row_q5_1_neon(src, dst, n)

    var sink = unsafe_alloc[Float64](1, alignment=64)
    sink.unsafe_store(0, 0.0)

    var start = perf_counter_ns()
    for _ in range(ITERATIONS):
        dequantize_row_q5_1_neon(src, dst, n)
        var cur = sink.unsafe_load[width=1](offset=0)
        sink.unsafe_store(0, cur + Float64(dst.unsafe_load[width=1](offset=0)))
    var end = perf_counter_ns()

    print("Q5_1 checksum: ", sink.unsafe_load[width=1](offset=0))

    var elapsed_ns = Float64(end - start)
    var throughput = Float64(n * ITERATIONS) / (elapsed_ns / 1e6) / 1000
    var per_block_ns = elapsed_ns / Float64(nb * ITERATIONS)

    print("Q5_1: ", throughput, " M/s, ", per_block_ns, " ns/block")


def bench_q8_0():
    """Benchmark Q8_0 dequantization."""
    var n = 4096
    var nb = n // KV_QK

    var src = unsafe_alloc[UInt8](nb * 34, alignment=64)
    var dst = unsafe_alloc[Scalar[DType.float16]](n, alignment=64)

    for i in range(nb):
        src.unsafe_store(i * 34, UInt8(0x00))
        src.unsafe_store(i * 34 + 1, UInt8(0x38))
        for j in range(32):
            src.unsafe_store(i * 34 + 2 + j, UInt8((j * 13) % 256))

    for _ in range(WARMUP):
        dequantize_row_q8_0_neon(src, dst, n)

    var sink = unsafe_alloc[Float64](1, alignment=64)
    sink.unsafe_store(0, 0.0)

    var start = perf_counter_ns()
    for _ in range(ITERATIONS):
        dequantize_row_q8_0_neon(src, dst, n)
        var cur = sink.unsafe_load[width=1](offset=0)
        sink.unsafe_store(0, cur + Float64(dst.unsafe_load[width=1](offset=0)))
    var end = perf_counter_ns()

    print("Q8_0 checksum: ", sink.unsafe_load[width=1](offset=0))

    var elapsed_ns = Float64(end - start)
    var throughput = Float64(n * ITERATIONS) / (elapsed_ns / 1e6) / 1000
    var per_block_ns = elapsed_ns / Float64(nb * ITERATIONS)

    print("Q8_0: ", throughput, " M/s, ", per_block_ns, " ns/block")


def bench_q8_0_f32():
    """Benchmark Q8_0 dequantization to FP32."""
    var n = 4096
    var nb = n // KV_QK

    var src = unsafe_alloc[UInt8](nb * 34, alignment=64)
    var dst = unsafe_alloc[Scalar[DType.float32]](n, alignment=64)

    for i in range(nb):
        src.unsafe_store(i * 34, UInt8(0x00))
        src.unsafe_store(i * 34 + 1, UInt8(0x38))
        for j in range(32):
            src.unsafe_store(i * 34 + 2 + j, UInt8((j * 13) % 256))

    for _ in range(WARMUP):
        dequantize_row_q8_0_f32_neon(src, dst, n)

    var sink = unsafe_alloc[Float64](1, alignment=64)
    sink.unsafe_store(0, 0.0)

    var start = perf_counter_ns()
    for _ in range(ITERATIONS):
        dequantize_row_q8_0_f32_neon(src, dst, n)
        var cur = sink.unsafe_load[width=1](offset=0)
        sink.unsafe_store(0, cur + Float64(dst.unsafe_load[width=1](offset=0)))
    var end = perf_counter_ns()

    print("Q8_0 F32 checksum: ", sink.unsafe_load[width=1](offset=0))

    var elapsed_ns = Float64(end - start)
    var throughput = Float64(n * ITERATIONS) / (elapsed_ns / 1e6) / 1000
    var per_block_ns = elapsed_ns / Float64(nb * ITERATIONS)

    print("Q8_0 F32: ", throughput, " M/s, ", per_block_ns, " ns/block")


def main():
    print("=== Mojo SIMD KV Cache Dequantization ===")
    bench_q4_0()
    bench_q4_1()
    bench_q5_0()
    bench_q5_1()
    bench_q8_0()
    bench_q8_0_f32()

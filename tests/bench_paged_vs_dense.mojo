# Benchmark: Paged KV vs Dense KV
# Compare memory allocation and access performance

from src.core.ops.attention.kv_cache import KVCacheLayer, KVCacheType
from src.core.tensor import Tensor, tensor_zeros
from std.time import perf_counter_ns
from std.utils import StaticTuple
from std.math import sqrt

comptime HEAD_DIM = 64
comptime N_KV_HEADS = 8
comptime PAGE_SIZE = 256  # Optimized for near-zero overhead

def bench_allocation():
    """Compare allocation time: paged vs dense."""
    print("=== Allocation Benchmark ===\n")

    var ctx_lengths = [100, 500, 1000, 2000, 4000]
    var iterations = 10

    print("ctx_len | Dense(ms) | Paged(ms) | Speedup")
    print("------------------------------------------")

    for ctx_len in ctx_lengths:
        # Dense allocation
        var start = perf_counter_ns()
        for i in range(iterations):
            var dense = KVCacheLayer(N_KV_HEADS, ctx_len, HEAD_DIM, KVCacheType.FP16)
        var end = perf_counter_ns()
        var dense_ms = Float64(end - start) / Float64(iterations) / 1e6

        # Paged allocation (start empty, grow on demand)
        start = perf_counter_ns()
        for i in range(iterations):
            var paged = KVCacheLayer(N_KV_HEADS, 0, HEAD_DIM, KVCacheType.FP16)
            paged.enable_paged(PAGE_SIZE, N_KV_HEADS, HEAD_DIM)
            paged.ensure_capacity(ctx_len)
        end = perf_counter_ns()
        var paged_ms = Float64(end - start) / Float64(iterations) / 1e6

        var speedup = dense_ms / paged_ms if paged_ms > 0 else 0.0
        print(ctx_len, "   |", dense_ms, "  |", paged_ms, "  |", speedup, "x")


def bench_access():
    """Compare access time: paged vs dense."""
    print("\n=== Access Benchmark ===\n")

    var ctx_lengths = [100, 500, 1000, 2000]
    var iterations = 100

    print("ctx_len | Dense(us/write) | Paged(us/write) | Overhead")
    print("---------------------------------------------------------")

    for ctx_len in ctx_lengths:
        # Dense
        var dense = KVCacheLayer(N_KV_HEADS, ctx_len, HEAD_DIM, KVCacheType.FP16)
        var k_row = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](HEAD_DIM))
        var v_row = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](HEAD_DIM))
        for d in range(HEAD_DIM):
            k_row.set(d, Scalar[DType.float16](0.5))
            v_row.set(d, Scalar[DType.float16](0.5))

        var start = perf_counter_ns()
        for iter in range(iterations):
            for pos in range(ctx_len):
                for h in range(N_KV_HEADS):
                    dense.set_kv_row(h, pos, k_row, v_row)
        var end = perf_counter_ns()
        var dense_us = Float64(end - start) / Float64(iterations * ctx_len) / 1000.0

        # Paged
        var paged = KVCacheLayer(N_KV_HEADS, 0, HEAD_DIM, KVCacheType.FP16)
        paged.enable_paged(PAGE_SIZE, N_KV_HEADS, HEAD_DIM)
        paged.ensure_capacity(ctx_len)

        start = perf_counter_ns()
        for iter in range(iterations):
            for pos in range(ctx_len):
                for h in range(N_KV_HEADS):
                    paged.set_kv_row(h, pos, k_row, v_row)
        end = perf_counter_ns()
        var paged_us = Float64(end - start) / Float64(iterations * ctx_len) / 1000.0

        var overhead = (paged_us / dense_us - 1.0) * 100.0 if dense_us > 0 else 0.0
        print(ctx_len, "   |", dense_us, "        |", paged_us, "        |", overhead, "%")


def bench_long_sequence():
    """Test dynamic growth beyond initial allocation."""
    print("\n=== Long Sequence Benchmark (Dynamic Growth) ===\n")

    # Start with 100 tokens, grow to 10000
    var paged = KVCacheLayer(N_KV_HEADS, 0, HEAD_DIM, KVCacheType.FP16)
    paged.enable_paged(PAGE_SIZE, N_KV_HEADS, HEAD_DIM)

    print("Starting with 100 tokens...")
    paged.ensure_capacity(100)

    var k_row = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](HEAD_DIM))
    var v_row = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](HEAD_DIM))
    for d in range(HEAD_DIM):
        k_row.set(d, Scalar[DType.float16](0.5))
        v_row.set(d, Scalar[DType.float16](0.5))

    # Write initial 100
    var start = perf_counter_ns()
    for pos in range(100):
        for h in range(N_KV_HEADS):
            paged.set_kv_row(h, pos, k_row, v_row)
    var end = perf_counter_ns()
    print("100 tokens written in", Float64(end - start) / 1e6, "ms")
    print("  n_blocks:", paged.n_blocks)

    # Grow to 1000
    print("\nGrowing to 1000 tokens...")
    start = perf_counter_ns()
    paged.ensure_capacity(1000)
    for pos in range(100, 1000):
        for h in range(N_KV_HEADS):
            paged.set_kv_row(h, pos, k_row, v_row)
    end = perf_counter_ns()
    print("900 more tokens in", Float64(end - start) / 1e6, "ms")
    print("  n_blocks:", paged.n_blocks)

    # Grow to 10000
    print("\nGrowing to 10000 tokens...")
    start = perf_counter_ns()
    paged.ensure_capacity(10000)
    for pos in range(1000, 10000):
        for h in range(N_KV_HEADS):
            paged.set_kv_row(h, pos, k_row, v_row)
    end = perf_counter_ns()
    print("9000 more tokens in", Float64(end - start) / 1e6, "ms")
    print("  n_blocks:", paged.n_blocks)

    print("\nTotal memory usage:")
    var bytes_per_block = N_KV_HEADS * PAGE_SIZE * HEAD_DIM * 2  # K + V, fp16
    var total_bytes = paged.n_blocks * bytes_per_block
    print("  ", Float64(total_bytes) / 1024.0 / 1024.0, "MB")


def main():
    bench_allocation()
    bench_access()
    bench_long_sequence()

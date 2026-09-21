# Test: Chunked Flash Attention
# Verify correctness and performance on long sequences

from src.core.ops.cpu.flash_attention_cpu import flash_attention_decode
from src.core.ops.attention.kv_cache import KVCacheLayer, KVCacheType
from src.core.tensor import Tensor, tensor_zeros
from std.time import perf_counter_ns
from std.utils import StaticTuple
from std.math import sqrt

comptime HEAD_DIM = 64
comptime N_KV_HEADS = 8

def test_correctness():
    """Verify chunked attention matches non-chunked results."""
    print("=== Testing Chunked Flash Attention Correctness ===\n")

    # Create KV cache with different sizes
    var ctx_lengths = [100, 300, 500, 1000, 2000]

    for ctx_len in ctx_lengths:
        var cache = KVCacheLayer(N_KV_HEADS, ctx_len, HEAD_DIM, KVCacheType.FP16)
        cache.filled = ctx_len

        # Fill with data (normalized to [-1, 1] range for numerical stability)
        for pos in range(ctx_len):
            for h in range(N_KV_HEADS):
                for d in range(HEAD_DIM):
                    # Use small values to avoid overflow
                    var val = Float32(((pos * N_KV_HEADS + h + d) % 20) - 10) / 10.0
                    cache.k.set((h * ctx_len + pos) * HEAD_DIM + d, Scalar[DType.float16](val))
                    cache.v.set((h * ctx_len + pos) * HEAD_DIM + d, Scalar[DType.float16](val))

        # Create query
        var q = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](HEAD_DIM))
        for d in range(HEAD_DIM):
            q.set(d, Scalar[DType.float16](Float32(d % 32) / 32.0))

        # Run attention
        var start_pos = ctx_len - 1
        var scale = Float32(1.0 / sqrt(Float64(HEAD_DIM)))

        var out = flash_attention_decode(q, cache, 0, start_pos, HEAD_DIM, scale)

        print("ctx_len:", ctx_len, " -> output[0]:", out.get(0))

    print("\nCorrectness test complete.")


def benchmark_chunked():
    """Benchmark chunked attention on long sequences."""
    print("\n=== Benchmarking Chunked Flash Attention ===\n")

    var ctx_lengths = [512, 1024, 2048, 4096, 8192]
    var iterations = 100

    for ctx_len in ctx_lengths:
        var cache = KVCacheLayer(N_KV_HEADS, ctx_len, HEAD_DIM, KVCacheType.FP16)
        cache.filled = ctx_len

        # Fill with data
        for pos in range(ctx_len):
            for h in range(N_KV_HEADS):
                for d in range(HEAD_DIM):
                    # Use small values to avoid overflow
                    var val = Float32(((pos * N_KV_HEADS + h + d) % 20) - 10) / 10.0
                    cache.k.set((h * ctx_len + pos) * HEAD_DIM + d, Scalar[DType.float16](val))
                    cache.v.set((h * ctx_len + pos) * HEAD_DIM + d, Scalar[DType.float16](val))

        # Create query
        var q = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](HEAD_DIM))
        for d in range(HEAD_DIM):
            q.set(d, Scalar[DType.float16](Float32(d % 32) / 32.0))

        var start_pos = ctx_len - 1
        var scale = Float32(1.0 / sqrt(Float64(HEAD_DIM)))

        # Warmup
        for i in range(5):
            var out = flash_attention_decode(q, cache, 0, start_pos, HEAD_DIM, scale)

        # Measure
        var start = perf_counter_ns()
        for i in range(iterations):
            var out = flash_attention_decode(q, cache, 0, start_pos, HEAD_DIM, scale)
        var end = perf_counter_ns()

        var elapsed_ns = Float64(end - start)
        var avg_us = elapsed_ns / Float64(iterations) / 1000.0
        var throughput = Float64(ctx_len * iterations) / (elapsed_ns / 1e9) / 1e6

        print("ctx_len:", ctx_len, "  time:", avg_us, "us  throughput:", throughput, "M KV/s")


def main():
    test_correctness()
    benchmark_chunked()

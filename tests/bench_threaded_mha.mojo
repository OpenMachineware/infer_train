# Benchmark: Threaded MHA vs Single-threaded MHA
# Tests multi-head attention parallel processing

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.attention.kv_cache import KVCacheLayer, KVCacheType
from src.core.ops.attention.mha_threaded import mha_forward_threaded, _mha_forward_single
from std.math import sqrt
from std.time import perf_counter_ns
from std.utils import StaticTuple
from src.core.thread_pool import resolve_threads


comptime HEAD_DIM = 128
comptime N_HEADS = 32  # Qwen2-7B style
comptime N_KV_HEADS = 32
comptime WARMUP = 3
comptime ITERATIONS = 10


def fill_kv_cache(mut cache: KVCacheLayer, n_positions: Int, head_dim: Int):
    var k_row = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    var v_row = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    for pos in range(n_positions):
        for h in range(N_KV_HEADS):
            for d in range(head_dim):
                k_row.set(d, Scalar[DType.float16](
                    Float16(Float32((pos * N_KV_HEADS + h * head_dim + d) % 256) / 256.0)
                ))
                v_row.set(d, Scalar[DType.float16](
                    Float16(Float32((pos * N_KV_HEADS + h * head_dim + d + 128) % 256) / 256.0)
                ))
            cache.set_kv_row(h, pos, k_row, v_row)
    cache.filled = n_positions


def test_correctness() -> Bool:
    print("=== Testing correctness (32 heads) ===")

    var cache = KVCacheLayer(
        N_KV_HEADS,
        1024,
        HEAD_DIM,
        KVCacheType.FP16,
    )
    fill_kv_cache(cache, 512, HEAD_DIM)

    # Create query for all heads
    var q = tensor_zeros[DType.float16, 3](
        StaticTuple[Int, 3](N_HEADS, 1, HEAD_DIM)
    )
    for h in range(N_HEADS):
        for d in range(HEAD_DIM):
            q.set(h * HEAD_DIM + d, Scalar[DType.float16](
                Float16(Float32((h * HEAD_DIM + d) % 64) / 64.0)
            ))

    var scale = Float32(1.0) / sqrt(Float32(HEAD_DIM))
    var position = 511

    # Run single-threaded
    var out_single = _mha_forward_single(
        q, cache, position, N_HEADS, N_KV_HEADS, HEAD_DIM, scale
    )

    # Run multi-threaded (force 4 threads)
    var out_threaded = mha_forward_threaded(
        q, cache, position, N_HEADS, N_KV_HEADS, HEAD_DIM, scale, 4
    )

    # Compare results
    var max_diff = Float32(0.0)
    for h in range(N_HEADS):
        for d in range(HEAD_DIM):
            var diff = abs(
                Float32(out_single.get(h * HEAD_DIM + d)) -
                Float32(out_threaded.get(h * HEAD_DIM + d))
            )
            if diff > max_diff:
                max_diff = diff

    print("Max difference: ", max_diff)

    if max_diff < Float32(1e-3):
        print("✅ Correctness verified (threaded == single)")
        return True
    else:
        print("❌ Mismatch detected!")
        return False


def benchmark():
    print("\n=== Benchmarking Threaded MHA (32 heads) ===")

    var context_lengths = [64, 256, 1024, 4096]
    var scale = Float32(1.0) / sqrt(Float32(HEAD_DIM))
    var threads = resolve_threads(0)

    print("Available threads: ", threads)
    print("")

    for ctx_len in context_lengths:
        var cache = KVCacheLayer(
            N_KV_HEADS,
            ctx_len,
            HEAD_DIM,
            KVCacheType.FP16,
        )
        fill_kv_cache(cache, ctx_len, HEAD_DIM)

        var q = tensor_zeros[DType.float16, 3](
            StaticTuple[Int, 3](N_HEADS, 1, HEAD_DIM)
        )
        for h in range(N_HEADS):
            for d in range(HEAD_DIM):
                q.set(h * HEAD_DIM + d, Scalar[DType.float16](
                    Float16(Float32((h * HEAD_DIM + d) % 64) / 64.0)
                ))

        var position = ctx_len - 1

        # Warmup
        for _ in range(WARMUP):
            _ = _mha_forward_single(
                q, cache, position, N_HEADS, N_KV_HEADS, HEAD_DIM, scale
            )
            _ = mha_forward_threaded(
                q, cache, position, N_HEADS, N_KV_HEADS, HEAD_DIM, scale, threads
            )

        # Measure single-threaded
        var start_single = perf_counter_ns()
        for _ in range(ITERATIONS):
            _ = _mha_forward_single(
                q, cache, position, N_HEADS, N_KV_HEADS, HEAD_DIM, scale
            )
        var single_ns = perf_counter_ns() - start_single

        # Measure multi-threaded
        var start_threaded = perf_counter_ns()
        for _ in range(ITERATIONS):
            _ = mha_forward_threaded(
                q, cache, position, N_HEADS, N_KV_HEADS, HEAD_DIM, scale, threads
            )
        var threaded_ns = perf_counter_ns() - start_threaded

        var single_ms = Float64(single_ns) / Float64(ITERATIONS) / 1e6
        var threaded_ms = Float64(threaded_ns) / Float64(ITERATIONS) / 1e6
        var speedup = Float64(single_ns) / Float64(threaded_ns)

        print("ctx=", ctx_len, ":")
        print("  single-threaded: ", single_ms, " ms")
        print("  multi-threaded:  ", threaded_ms, " ms")
        print("  speedup:         ", speedup, "x")
        print("")


def main():
    print("Threaded MHA Benchmark\n")

    var correct = test_correctness()

    if correct:
        benchmark()
    else:
        print("\n⚠️  Skipping benchmark due to correctness issues")

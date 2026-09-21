# Benchmark: Flash Attention vs Current Implementation
# Tests decode mode performance

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.cpu.flash_attention_cpu import flash_attention_decode
from src.core.ops.attention.kv_cache import KVCacheLayer, KVCacheType
from std.math import exp, sqrt
from std.time import perf_counter_ns
from std.utils import StaticTuple


comptime HEAD_DIM = 128
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


def current_attention_decode(
    q: Tensor[DType.float16, 1],
    cache: KVCacheLayer,
    kv_head: Int,
    start_pos: Int,
    head_dim: Int,
    scale: Float32,
) -> Tensor[DType.float16, 1]:
    """Current implementation (3-pass softmax with score storage)."""
    var max_len = cache.max_len
    var first = cache.first_position()
    if first < 0:
        first = 0
    var seq = start_pos + 1
    var n_scores = seq - first

    var scores = List[Float32]()
    var k_row = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))

    for t in range(first, seq):
        var acc = Float32(0)
        cache.get_k_row(kv_head, t, k_row)
        for d in range(head_dim):
            acc += Float32(q.get(d)) * Float32(k_row.get(d))
        scores.append(acc * scale)

    var mx = Float32(-3.0e38)
    for i in range(n_scores):
        if scores[i] > mx:
            mx = scores[i]

    var total = Float32(0)
    for i in range(n_scores):
        var e = exp(scores[i] - mx)
        scores[i] = e
        total += e

    var inv = Float32(1.0) / total
    for i in range(n_scores):
        scores[i] = scores[i] * inv

    var out = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    var v_row = Tensor[DType.float16, 1](StaticTuple[Int, 1](head_dim))
    for d in range(head_dim):
        var acc_f = Float32(0)
        for i in range(n_scores):
            cache.get_v_row(kv_head, first + i, v_row)
            acc_f += scores[i] * Float32(v_row.get(d))
        out.set(d, Scalar[DType.float16](acc_f))

    return out


def test_correctness_decode() -> Bool:
    print("=== Testing correctness (decode mode) ===")

    var cache = KVCacheLayer(
        N_KV_HEADS,
        1024,
        HEAD_DIM,
        KVCacheType.FP16,
    )

    fill_kv_cache(cache, 512, HEAD_DIM)

    var q = Tensor[DType.float16, 1](StaticTuple[Int, 1](HEAD_DIM))
    for d in range(HEAD_DIM):
        q.set(d, Scalar[DType.float16](Float16(Float32(d % 64) / 64.0)))

    var scale = Float32(1.0) / sqrt(Float32(HEAD_DIM))

    var out_flash = flash_attention_decode(q, cache, 0, 511, HEAD_DIM, scale)
    var out_current = current_attention_decode(q, cache, 0, 511, HEAD_DIM, scale)

    var max_diff = Float32(0.0)
    for d in range(HEAD_DIM):
        var diff = abs(Float32(out_flash.get(d)) - Float32(out_current.get(d)))
        if diff > max_diff:
            max_diff = diff

    print("Max difference: ", max_diff)

    if max_diff < Float32(1e-3):
        print("✅ Correctness verified (decode)")
        return True
    else:
        print("❌ Mismatch detected!")
        for d in range(min(5, HEAD_DIM)):
            print("  d=", d, " flash=", out_flash.get(d), " current=", out_current.get(d))
        return False


def benchmark_decode():
    print("\n=== Benchmarking decode mode ===")

    var context_lengths = [64, 256, 1024, 4096]
    var scale = Float32(1.0) / sqrt(Float32(HEAD_DIM))

    for ctx_len in context_lengths:
        var cache = KVCacheLayer(
            N_KV_HEADS,
            ctx_len,
            HEAD_DIM,
            KVCacheType.FP16,
        )
        fill_kv_cache(cache, ctx_len, HEAD_DIM)

        var q = Tensor[DType.float16, 1](StaticTuple[Int, 1](HEAD_DIM))
        for d in range(HEAD_DIM):
            q.set(d, Scalar[DType.float16](Float16(Float32(d % 64) / 64.0)))

        # Warmup
        for _ in range(WARMUP):
            _ = flash_attention_decode(q, cache, 0, ctx_len - 1, HEAD_DIM, scale)
            _ = current_attention_decode(q, cache, 0, ctx_len - 1, HEAD_DIM, scale)

        # Measure Flash Attention
        var start = perf_counter_ns()
        for _ in range(ITERATIONS):
            _ = flash_attention_decode(q, cache, 0, ctx_len - 1, HEAD_DIM, scale)
        var flash_ns = perf_counter_ns() - start

        # Measure Current
        start = perf_counter_ns()
        for _ in range(ITERATIONS):
            _ = current_attention_decode(q, cache, 0, ctx_len - 1, HEAD_DIM, scale)
        var current_ns = perf_counter_ns() - start

        var flash_ms = Float64(flash_ns) / Float64(ITERATIONS) / 1e6
        var current_ms = Float64(current_ns) / Float64(ITERATIONS) / 1e6
        var speedup = current_ms / flash_ms

        print("\nContext length: ", ctx_len)
        print("  Flash Attention: ", flash_ms, " ms")
        print("  Current impl:    ", current_ms, " ms")
        print("  Speedup:         ", speedup, "x")


def main():
    print("Flash Attention CPU Benchmark\n")

    var correct = test_correctness_decode()

    if correct:
        benchmark_decode()
    else:
        print("\n⚠️  Skipping benchmark due to correctness issues")

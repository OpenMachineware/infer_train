# Test: Multi-request batch attention
# Tests parallel processing of multiple requests

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.attention.kv_cache import KVCacheLayer, KVCacheType
from src.core.ops.attention.attention_scheduler import mha_forward_batch
from src.core.thread_pool import resolve_threads
from std.math import sqrt
from std.time import perf_counter_ns
from std.utils import StaticTuple


comptime HEAD_DIM = 128
comptime N_HEADS = 32
comptime N_KV_HEADS = 32
comptime WARMUP = 2
comptime ITERATIONS = 5


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
    print("=== Testing correctness (4 requests) ===")

    var n_requests = 4
    var ctx_len = 256

    # Prepare requests
    var queries = List[Tensor[DType.float16, 3]]()
    var caches = List[KVCacheLayer]()
    var positions = List[Int]()
    var n_heads_list = List[Int]()

    for i in range(n_requests):
        var q = tensor_zeros[DType.float16, 3](
            StaticTuple[Int, 3](N_HEADS, 1, HEAD_DIM)
        )
        for h in range(N_HEADS):
            for d in range(HEAD_DIM):
                q.set(h * HEAD_DIM + d, Scalar[DType.float16](
                    Float16(Float32((i * N_HEADS * HEAD_DIM + h * HEAD_DIM + d) % 64) / 64.0)
                ))
        queries.append(q)

        var cache = KVCacheLayer(N_KV_HEADS, ctx_len, HEAD_DIM, KVCacheType.FP16)
        fill_kv_cache(cache, ctx_len, HEAD_DIM)
        caches.append(cache.copy())

        positions.append(ctx_len - 1)
        n_heads_list.append(N_HEADS)

    var threads = resolve_threads(0)
    print("Using threads: ", threads)

    # Run batch
    var outputs = mha_forward_batch(
        queries, caches, positions, n_heads_list, N_KV_HEADS, HEAD_DIM, threads
    )

    print("Batch outputs: ", len(outputs), " tensors")
    print("First output shape: [", outputs[0].shape()[0], ", ", outputs[0].shape()[1], ", ", outputs[0].shape()[2], "]")

    # Verify each output has valid values
    var all_valid = True
    for i in range(n_requests):
        var has_nan = False
        for h in range(N_HEADS):
            for d in range(HEAD_DIM):
                var val = Float32(outputs[i].get(h * HEAD_DIM + d))
                if val != val:  # NaN check
                    has_nan = True
                    break
            if has_nan:
                break
        if has_nan:
            print("Request ", i, " has NaN values!")
            all_valid = False

    if all_valid:
        print("✅ All outputs valid (no NaN)")
        return True
    else:
        print("❌ Invalid outputs detected")
        return False


def benchmark():
    print("\n=== Benchmarking Batch Attention ===")

    var request_counts = [2, 4, 8]
    var ctx_len = 256
    var threads = resolve_threads(0)

    print("Available threads: ", threads)
    print("")

    for n_requests in request_counts:
        # Prepare requests
        var queries = List[Tensor[DType.float16, 3]]()
        var caches = List[KVCacheLayer]()
        var positions = List[Int]()
        var n_heads_list = List[Int]()

        for i in range(n_requests):
            var q = tensor_zeros[DType.float16, 3](
                StaticTuple[Int, 3](N_HEADS, 1, HEAD_DIM)
            )
            for h in range(N_HEADS):
                for d in range(HEAD_DIM):
                    q.set(h * HEAD_DIM + d, Scalar[DType.float16](
                        Float16(Float32((h * HEAD_DIM + d) % 64) / 64.0)
                    ))
            queries.append(q)

            var cache = KVCacheLayer(N_KV_HEADS, ctx_len, HEAD_DIM, KVCacheType.FP16)
            fill_kv_cache(cache, ctx_len, HEAD_DIM)
            caches.append(cache.copy())

            positions.append(ctx_len - 1)
            n_heads_list.append(N_HEADS)

        # Warmup
        for _ in range(WARMUP):
            _ = mha_forward_batch(
                queries, caches, positions, n_heads_list, N_KV_HEADS, HEAD_DIM, threads
            )

        # Measure
        var start = perf_counter_ns()
        for _ in range(ITERATIONS):
            _ = mha_forward_batch(
                queries, caches, positions, n_heads_list, N_KV_HEADS, HEAD_DIM, threads
            )
        var elapsed_ns = perf_counter_ns() - start

        var total_heads = n_requests * N_HEADS
        var ms_per_iter = Float64(elapsed_ns) / Float64(ITERATIONS) / 1e6
        var us_per_head = (ms_per_iter * 1000.0) / Float64(total_heads)

        print(n_requests, " requests (", total_heads, " heads total):")
        print("  total time:  ", ms_per_iter, " ms")
        print("  per request: ", ms_per_iter / Float64(n_requests), " ms")
        print("  per head:    ", us_per_head, " us")
        print("")


def main():
    print("Batch Attention Test\n")

    var correct = test_correctness()

    if correct:
        benchmark()
    else:
        print("\n⚠️  Skipping benchmark due to correctness issues")

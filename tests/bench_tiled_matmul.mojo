# Benchmark tiled GPU matmul

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.gpu.matmul_tiled_gpu import matmul_weight_tiled_gpu
from src.core.ops.gpu.matmul_gpu import matmul_weight_gpu
from src.core.thread_pool import now_ns
from std.utils.static_tuple import StaticTuple


def benchmark_decode() -> Bool:
    """Benchmark for decode (M=1)."""
    var M = 1  # Decode
    var K = 1536
    var N = 1536

    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.set(i, Scalar[DType.float16](Float32(i) * 0.001))

    var w = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))
    for i in range(N * K):
        w.set(i, Scalar[DType.float16](Float32(i) * 0.001))

    # Warm up
    var _ = matmul_weight_gpu[DType.float16](x, w)
    var _ = matmul_weight_tiled_gpu[DType.float16](x, w)

    # Benchmark
    var iterations = 100
    var start = now_ns()
    for _ in range(iterations):
        var _ = matmul_weight_gpu[DType.float16](x, w)
    var naive_ns = now_ns() - start

    start = now_ns()
    for _ in range(iterations):
        var _ = matmul_weight_tiled_gpu[DType.float16](x, w)
    var tiled_ns = now_ns() - start

    var naive_ms = naive_ns // 1_000_000
    var tiled_ms = tiled_ns // 1_000_000

    print("=== Decode (M=1) ===")
    print("naive: ", naive_ms, "ms for ", iterations, " iterations (", Float32(naive_ms) / Float32(iterations), "ms/iter)")
    print("tiled: ", tiled_ms, "ms for ", iterations, " iterations (", Float32(tiled_ms) / Float32(iterations), "ms/iter)")
    print("speedup: ", Float32(naive_ms) / Float32(tiled_ms), "x")
    return True


def benchmark_prefill() -> Bool:
    """Benchmark for prefill (M>1)."""
    var M = 32
    var K = 1536
    var N = 1536

    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.set(i, Scalar[DType.float16](Float32(i) * 0.001))

    var w = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))
    for i in range(N * K):
        w.set(i, Scalar[DType.float16](Float32(i) * 0.001))

    # Warm up
    var _ = matmul_weight_gpu[DType.float16](x, w)
    var _ = matmul_weight_tiled_gpu[DType.float16](x, w)

    var iterations = 20
    var start = now_ns()
    for _ in range(iterations):
        var _ = matmul_weight_gpu[DType.float16](x, w)
    var naive_ns = now_ns() - start

    start = now_ns()
    for _ in range(iterations):
        var _ = matmul_weight_tiled_gpu[DType.float16](x, w)
    var tiled_ns = now_ns() - start

    var naive_ms = naive_ns // 1_000_000
    var tiled_ms = tiled_ns // 1_000_000

    print("=== Prefill (M=32) ===")
    print("naive: ", naive_ms, "ms for ", iterations, " iterations (", Float32(naive_ms) / Float32(iterations), "ms/iter)")
    print("tiled: ", tiled_ms, "ms for ", iterations, " iterations (", Float32(tiled_ms) / Float32(iterations), "ms/iter)")
    print("speedup: ", Float32(naive_ms) / Float32(tiled_ms), "x")

    # Verify correctness
    var y_naive = matmul_weight_gpu[DType.float16](x, w)
    var y_tiled = matmul_weight_tiled_gpu[DType.float16](x, w)

    var max_diff = Float32(0.0)
    for i in range(M * N):
        var diff = abs(Float32(y_tiled.get(i)) - Float32(y_naive.get(i)))
        if diff > max_diff:
            max_diff = diff

    print("max_diff: ", max_diff)
    if max_diff > 0.5:
        print("FAIL: max_diff too large")
        return False
    return True


def main():
    var all_passed = True
    if not benchmark_decode():
        all_passed = False
    if not benchmark_prefill():
        all_passed = False

    if all_passed:
        print("benchmark_tiled OK")
    else:
        print("benchmark_tiled FAILED")

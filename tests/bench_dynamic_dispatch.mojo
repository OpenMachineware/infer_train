# Benchmark dynamic dispatch between naive and tiled kernels
# Tests the threshold logic (M > 8 and K >= 64)

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.gpu.matmul_gpu import matmul_weight_gpu
from src.core.thread_pool import now_ns
from std.math import sin
from std.utils.static_tuple import StaticTuple

def fill_tensor(t: Tensor[DType.float16, 2], seed: Int):
    """Fill tensor with deterministic pattern."""
    var n = t.numel()
    for i in range(n):
        var v = sin(Float32(i + seed) * Float32(0.13)) * Float32(0.8)
        t.set(i, Scalar[DType.float16](v))


def benchmark_matmul(M: Int, K: Int, N: Int, iterations: Int = 10) -> Float64:
    """Benchmark matmul_weight_gpu with given dimensions."""
    # Create tensors
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    var w = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))
    fill_tensor(x, 1)
    fill_tensor(w, 2)

    # Warmup
    var _ = matmul_weight_gpu[DType.float16](x, w)

    # Benchmark
    var start = now_ns()
    for i in range(iterations):
        var _ = matmul_weight_gpu[DType.float16](x, w)
    var end = now_ns()

    return Float64(end - start) / 1e6 / Float64(iterations)  # ms per iteration


def main():
    print("Dynamic dispatch benchmark (threshold: M > 8 and K >= 64)")
    print("=" * 60)
    print()

    var K = 4096  # Typical hidden dimension
    var N = 14336  # Typical FFN output dimension

    # Test different batch sizes around the threshold
    var batch_sizes = [1, 2, 4, 8, 9, 16, 32, 64]

    print("Batch  Kernel      Time(ms)  Notes")
    print("-" * 60)

    for M in batch_sizes:
        var use_tiled = M > 8 and K >= 64
        var kernel_name = "naive " if not use_tiled else "tiled "
        var time_ms = benchmark_matmul(M, K, N, iterations=5)
        var note = "decode" if M == 1 else "prefill" if M >= 32 else ""
        print(M, "   ", kernel_name, "   ", time_ms, " ", note)

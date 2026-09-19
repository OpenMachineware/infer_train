# Test optimized FP16/FP32 matmul performance with pre-transposed weights

from src.core.tensor import tensor_zeros, Tensor
from src.core.ops.cpu.matmul_cpu import _matmul_kernel_f16, _matmul_kernel_f32
from src.core.ops.cpu.matmul_fp_optimized import (
    transpose_f16,
    transpose_f32,
    matmul_f16_pretransposed,
    matmul_f32_pretransposed,
)
from src.core.thread_pool import now_ns
from std.utils import StaticTuple

def test_single_size(dtype: DType, M: Int, K: Int, N: Int, label: String):
    print("\n=== Testing", label, "(M=", M, ", K=", K, ", N=", N, ") ===")

    if dtype == DType.float16:
        # Create tensors
        var a = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
        var b = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](K, N))

        # Initialize
        for i in range(M * K):
            a.set(i, Scalar[DType.float16](0.5))
        for i in range(K * N):
            b.set(i, Scalar[DType.float16](0.3))

        # Test old implementation
        for _ in range(10):
            var _ = _matmul_kernel_f16(a, b)

        var iterations = 100
        var start = now_ns()
        for _ in range(iterations):
            var _ = _matmul_kernel_f16(a, b)
        var elapsed_old = Float64(now_ns() - start) / 1e9
        var flops = Float64(2 * M * K * N * iterations)
        var gflops_old = flops / elapsed_old / 1e9

        print("  Old impl (row×col):", gflops_old, "GFLOPS")

        # Test optimized with pre-transposed B
        var b_transposed = transpose_f16(b)  # Pre-transpose (one-time cost for inference)

        for _ in range(10):
            var _ = matmul_f16_pretransposed(a, b_transposed)

        start = now_ns()
        for _ in range(iterations):
            var _ = matmul_f16_pretransposed(a, b_transposed)
        var elapsed_new = Float64(now_ns() - start) / 1e9
        var gflops_new = flops / elapsed_new / 1e9

        print("  Optimized (row×row, pre-transposed B):", gflops_new, "GFLOPS")
        print("  Speedup:", gflops_new / gflops_old, "x")

    else:  # FP32
        var a = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](M, K))
        var b = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](K, N))

        for i in range(M * K):
            a.set(i, Scalar[DType.float32](0.5))
        for i in range(K * N):
            b.set(i, Scalar[DType.float32](0.3))

        # Test old
        for _ in range(10):
            var _ = _matmul_kernel_f32(a, b)

        var iterations = 100
        var start = now_ns()
        for _ in range(iterations):
            var _ = _matmul_kernel_f32(a, b)
        var elapsed_old = Float64(now_ns() - start) / 1e9
        var flops = Float64(2 * M * K * N * iterations)
        var gflops_old = flops / elapsed_old / 1e9

        print("  Old impl (row×col):", gflops_old, "GFLOPS")

        # Test optimized with pre-transposed B
        var b_transposed = transpose_f32(b)

        for _ in range(10):
            var _ = matmul_f32_pretransposed(a, b_transposed)

        start = now_ns()
        for _ in range(iterations):
            var _ = matmul_f32_pretransposed(a, b_transposed)
        var elapsed_new = Float64(now_ns() - start) / 1e9
        var gflops_new = flops / elapsed_new / 1e9

        print("  Optimized (row×row, pre-transposed B):", gflops_new, "GFLOPS")
        print("  Speedup:", gflops_new / gflops_old, "x")


def main():
    print("FP16/FP32 Matmul Optimization Test")
    print("==================================")

    # Test different sizes
    # Small: M=1 (decode), Medium: M=16 (small batch), Large: M=64 (batch prefill)
    var sizes = [(1, 4096, 512), (16, 4096, 512), (64, 4096, 512)]

    print("\n--- FP16 Tests ---")
    for size in sizes:
        test_single_size(DType.float16, size[0], size[1], size[2], "FP16")

    print("\n--- FP32 Tests ---")
    for size in sizes:
        test_single_size(DType.float32, size[0], size[1], size[2], "FP32")

    print("\n=== Comparison with llama.cpp ===")
    print("llama.cpp FP16: ~50 GFLOPS")
    print("llama.cpp FP32: ~23 GFLOPS")

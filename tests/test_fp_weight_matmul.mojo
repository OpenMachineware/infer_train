# Test FP16/FP32 weight-major matmul performance (actual inference kernel)

from src.core.tensor import tensor_zeros, Tensor
from src.core.ops.cpu.matmul_cpu import (
    matmul_weight_cpu,
    matmul_weight_cpu_threaded,
)
from src.core.ops.cpu.matmul_fp_weight_optimized import matmul_weight_f16_optimized
from src.core.ops.cpu.matmul_fp_weight_block import matmul_weight_f16_block
from src.core.thread_pool import now_ns
from src.core.cpu_features import detect_cpu_flags
from std.utils import StaticTuple

def test_weight_matmul(dtype: DType, M: Int, K: Int, N: Int, label: String):
    """Test the actual weight-major matmul used in inference.

    Layout: weights [N, K], activations [M, K], output [M, N]
    Formula: y[i,j] = sum_k x[i,k] * w[j,k] (row×row dot product)
    """
    print("\n=== Testing", label, "(M=", M, ", K=", K, ", N=", N, ") ===")

    if dtype == DType.float16:
        # Create tensors: x [M, K], w [N, K]
        var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
        var w = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))

        # Initialize with fixed values
        for i in range(M * K):
            x.set(i, Scalar[DType.float16](0.5))
        for i in range(N * K):
            w.set(i, Scalar[DType.float16](0.3))

        # Warmup
        for _ in range(10):
            var _ = matmul_weight_cpu[DType.float16](x, w)

        # Benchmark
        var iterations = 100
        var start = now_ns()
        for _ in range(iterations):
            var _ = matmul_weight_cpu[DType.float16](x, w)
        var elapsed = Float64(now_ns() - start) / 1e9

        var flops = Float64(2 * M * K * N * iterations)
        var gflops = flops / elapsed / 1e9

        print("  matmul_weight_cpu (row×row):", gflops, "GFLOPS")

        # Test optimized version with NEON FMA
        for _ in range(10):
            var _ = matmul_weight_f16_optimized(x, w)

        start = now_ns()
        for _ in range(iterations):
            var _ = matmul_weight_f16_optimized(x, w)
        var elapsed_opt = Float64(now_ns() - start) / 1e9
        var gflops_opt = flops / elapsed_opt / 1e9
        print("  matmul_weight_f16_optimized (NEON FMA):", gflops_opt, "GFLOPS")
        print("  Speedup:", gflops_opt / gflops, "x")

        # Test block GEMM version with CPU flags
        var flags = detect_cpu_flags()
        for _ in range(10):
            var _ = matmul_weight_f16_block(x, w, flags)

        start = now_ns()
        for _ in range(iterations):
            var _ = matmul_weight_f16_block(x, w, flags)
        var elapsed_block = Float64(now_ns() - start) / 1e9
        var gflops_block = flops / elapsed_block / 1e9
        print("  matmul_weight_f16_block (RN=8):", gflops_block, "GFLOPS")
        print("  Speedup vs baseline:", gflops_block / gflops, "x")

        # Test threaded version for batch (M > 1)
        if M > 1:
            start = now_ns()
            for _ in range(iterations):
                var _ = matmul_weight_cpu_threaded[DType.float16](x, w, nthreads=0)
            var elapsed_t = Float64(now_ns() - start) / 1e9
            var gflops_t = flops / elapsed_t / 1e9
            print("  matmul_weight_cpu_threaded:", gflops_t, "GFLOPS")

    else:  # FP32
        var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](M, K))
        var w = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](N, K))

        for i in range(M * K):
            x.set(i, Scalar[DType.float32](0.5))
        for i in range(N * K):
            w.set(i, Scalar[DType.float32](0.3))

        # Warmup
        for _ in range(10):
            var _ = matmul_weight_cpu[DType.float32](x, w)

        # Benchmark
        var iterations = 100
        var start = now_ns()
        for _ in range(iterations):
            var _ = matmul_weight_cpu[DType.float32](x, w)
        var elapsed = Float64(now_ns() - start) / 1e9

        var flops = Float64(2 * M * K * N * iterations)
        var gflops = flops / elapsed / 1e9

        print("  matmul_weight_cpu (row×row):", gflops, "GFLOPS")

        # Test threaded version for batch
        if M > 1:
            start = now_ns()
            for _ in range(iterations):
                var _ = matmul_weight_cpu_threaded[DType.float32](x, w, nthreads=0)
            var elapsed_t = Float64(now_ns() - start) / 1e9
            var gflops_t = flops / elapsed_t / 1e9
            print("  matmul_weight_cpu_threaded:", gflops_t, "GFLOPS")


def main():
    print("FP16/FP32 Weight-Major Matmul Performance Test")
    print("=============================================")
    print("Testing actual inference kernel (row×row dot product)")
    print("Layout: weights [N, K], activations [M, K]")

    # Test different sizes
    # Small: M=1 (decode), Medium: M=16 (small batch), Large: M=64 (batch prefill)
    var sizes = [(1, 4096, 512), (16, 4096, 512), (64, 4096, 512)]

    print("\n--- FP16 Tests ---")
    for size in sizes:
        test_weight_matmul(DType.float16, size[0], size[1], size[2], "FP16")

    print("\n--- FP32 Tests ---")
    for size in sizes:
        test_weight_matmul(DType.float32, size[0], size[1], size[2], "FP32")

    print("\n=== Reference: llama.cpp Performance ===")
    print("llama.cpp FP16: ~50 GFLOPS (llamafile sgemm)")
    print("llama.cpp FP32: ~23 GFLOPS (llamafile sgemm)")
    print("\nNote: llama.cpp uses llamafile_sgemm with formula C = Aᵀ × B")
    print("      Our matmul_weight_cpu uses row×row: y = W @ x")
    print("      Both should have similar memory access patterns.")

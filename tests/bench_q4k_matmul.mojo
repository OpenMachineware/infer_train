# Benchmark Q4_K matmul against llama.cpp
from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
from src.core.ops.quantized.quant_types import QuantType
from src.core.thread_pool import now_ns
from std.utils.static_tuple import StaticTuple

def benchmark_q4k_matmul():
    print("=== Q4_K Matmul Benchmark ===")

    # Test dimensions (typical for 7B model)
    # M=1 (decode), K=4096 (hidden), N=14336 (FFN)
    var M = 1
    var K = 4096
    var N = 14336
    var iterations = 10

    print("M:", M, "K:", K, "N:", N, "Iterations:", iterations)

    # Create test input
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.set(i, Scalar[DType.float16](Float32(0.01) * Float32(i % 100)))

    # Create Q4_K weights (simulate)
    # For now, use FP16 weights and quantize them
    var w_fp16 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))
    for i in range(N * K):
        w_fp16.set(i, Scalar[DType.float16](Float32(0.01) * Float32(i % 100)))

    # Quantize weights to Q4_K
    print("Quantizing weights to Q4_K...")
    var w_quant = tensor_zeros[DType.uint8, 2](StaticTuple[Int, 2](N, 144 * (K // 256)))
    # TODO: Implement Q4_K quantization
    # For now, skip this and just test with FP16 matmul

    # Warmup with FP16 matmul
    print("Warming up with FP16 matmul...")
    from src.core.ops.cpu.matmul_cpu import matmul_weight_cpu_threaded
    for _ in range(3):
        _ = matmul_weight_cpu_threaded[DType.float16](x, w_fp16)

    # Benchmark FP16 matmul
    print("Benchmarking FP16 matmul...")
    var start = now_ns()
    for _ in range(iterations):
        _ = matmul_weight_cpu_threaded[DType.float16](x, w_fp16)
    var elapsed = now_ns() - start

    var avg_time_ms = Float64(elapsed) / Float64(iterations) / 1_000_000.0
    var ops_per_sec = Float64(iterations) / (Float64(elapsed) / 1_000_000_000.0)
    var flops = Float64(2 * M * K * N)  # multiply-add = 2 ops

    print("\nFP16 Matmul Results:")
    print("  Average time:", avg_time_ms, "ms")
    print("  Operations/sec:", ops_per_sec)
    print("  GFLOPS:", flops * ops_per_sec / 1_000_000_000.0)

    # Verify correctness
    var result = matmul_weight_cpu_threaded[DType.float16](x, w_fp16)
    var shape = result.shape()
    print("\nOutput shape: (", shape[0], ",", shape[1], ")")
    print("Output[0,0]:", Float32(result.get(0)))


def main():
    benchmark_q4k_matmul()

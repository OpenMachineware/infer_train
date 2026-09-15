# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# tools/bench_blas.mojo
#
# Benchmark comparing Apple Accelerate BLAS vs hand-written SIMD matmul.

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.cpu.matmul_cpu import matmul_weight_cpu
from src.core.ops.cpu.blas_cpu import matmul_weight_blas
from std.time import perf_counter_ns
from std.sys import argv, num_performance_cores
from std.utils.static_tuple import StaticTuple
from std.math import sqrt
from std.memory.alloc import unsafe_alloc
from std.random import rand

comptime WARMUP = 3
comptime REPEATS = 10


def random_fill(ptr: Pointer[Float32, MutUntrackedOrigin], n: Int):
    """Fill memory with random FP32 values in [-1, 1]."""
    for i in range(n):
        var r = Float32(i % 1000) / 500.0 - 1.0  # deterministic pseudo-random
        ptr.unsafe_offset(i).unsafe_store(val=r)


def benchmark_matmul(M: Int, K: Int, N: Int) raises:
    """Benchmark matmul_weight for given dimensions."""
    print("\n=== matmul_weight [", M, "x", K, "] @ [", N, "x", K, "]^T ===")

    # Create FP32 tensors
    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](M, K))
    var w = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](N, K))

    # Fill with pseudo-random values
    random_fill(x.data(), M * K)
    random_fill(w.data(), N * K)

    # Warmup SIMD kernel
    for _ in range(WARMUP):
        var _ = matmul_weight_cpu[DType.float32](x, w)

    # Benchmark SIMD
    var simd_times = List[Int]()
    for _ in range(REPEATS):
        var t0 = perf_counter_ns()
        var _ = matmul_weight_cpu[DType.float32](x, w)
        var t1 = perf_counter_ns()
        simd_times.append(t1 - t0)

    # Warmup BLAS kernel
    for _ in range(WARMUP):
        var _ = matmul_weight_blas[DType.float32](x, w)

    # Benchmark BLAS
    var blas_times = List[Int]()
    for _ in range(REPEATS):
        var t0 = perf_counter_ns()
        var _ = matmul_weight_blas[DType.float32](x, w)
        var t1 = perf_counter_ns()
        blas_times.append(t1 - t0)

    # Compute stats
    var simd_total = 0
    var simd_min = simd_times[0]
    var simd_max = simd_times[0]
    for t in simd_times:
        simd_total += t
        if t < simd_min:
            simd_min = t
        if t > simd_max:
            simd_max = t

    var blas_total = 0
    var blas_min = blas_times[0]
    var blas_max = blas_times[0]
    for t in blas_times:
        blas_total += t
        if t < blas_min:
            blas_min = t
        if t > blas_max:
            blas_max = t

    var simd_avg = simd_total // REPEATS
    var blas_avg = blas_total // REPEATS

    # Compute GFLOPS
    # matmul_weight: 2 * M * N * K FLOPs
    var flops = 2 * M * N * K
    var simd_gflops = Float64(flops) / (Float64(simd_avg) * 1e-9) / 1e9
    var blas_gflops = Float64(flops) / (Float64(blas_avg) * 1e-9) / 1e9

    print("SIMD:  ", simd_avg / 1000, "ms (min:", simd_min / 1000, "max:", simd_max / 1000, ")  ", simd_gflops, "GFLOPS")
    print("BLAS:  ", blas_avg / 1000, "ms (min:", blas_min / 1000, "max:", blas_max / 1000, ")  ", blas_gflops, "GFLOPS")
    print("Speedup:", Float64(simd_avg) / Float64(blas_avg), "x")


def main() raises:
    print("BLAS vs SIMD Matmul Benchmark")
    print("Apple M1 Max, 4 threads")
    print("Warmup:", WARMUP, "Repeats:", REPEATS)

    # Test various sizes relevant to LLM inference
    # 1. Small projection (Qwen3-0.6B Q projection: 896 x 896)
    benchmark_matmul(1, 896, 896)

    # 2. Medium projection (Qwen3-0.6B FFN gate: 896 x 4864)
    benchmark_matmul(1, 896, 4864)

    # 3. Large projection (DeepSeek-1.5B FFN gate: 1536 x 5376)
    benchmark_matmul(1, 1536, 5376)

    # 4. Batch decode (batch=16, Qwen3-0.6B)
    benchmark_matmul(16, 896, 896)

    # 5. Batch decode (batch=16, DeepSeek-1.5B)
    benchmark_matmul(16, 1536, 5376)

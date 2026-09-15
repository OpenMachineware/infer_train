# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# tools/bench_q8k_vs_fp32.mojo
#
# Benchmark comparing Q8_K + SDOT path vs FP32 SIMD path for quantized matmul.

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.cpu.matmul_cpu import matmul_quantized_cpu, matmul_quantized_cpu_threaded
from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
from src.core.ops.quantized.quant_types import QuantType, block_bytes, block_elems
from src.core.thread_pool import now_ns
from std.utils.static_tuple import StaticTuple
from std.memory.alloc import unsafe_alloc

comptime QK_K = 256
comptime Q4_K_BLOCK_BYTES = 144
comptime REPEATS = 10
comptime WARMUP = 3


def create_q4_k_weights(N: Int, K: Int) -> Tensor[DType.uint8, 2]:
    """Create Q4_K quantized weights with meaningful data."""
    var nb = K // 256
    var total_bytes = N * nb * Q4_K_BLOCK_BYTES
    var data = unsafe_alloc[UInt8](total_bytes)

    # Initialize with a pattern that exercises the quantization properly
    # Scale bytes: non-zero values
    # Quantized values: alternating pattern
    for i in range(total_bytes):
        if i % 144 < 16:
            # Header bytes (scales and d/dmin)
            data.unsafe_offset(i).unsafe_store(val=UInt8(64))  # Non-zero scale
        else:
            # Quantized values
            data.unsafe_offset(i).unsafe_store(val=UInt8((i % 256)))

    var shape = StaticTuple[Int, 2](N, nb * Q4_K_BLOCK_BYTES)
    return Tensor[DType.uint8, 2](shape, data)


def create_fp16_activations(M: Int, K: Int) -> Tensor[DType.float16, 2]:
    """Create FP16 activations with a simple pattern."""
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.data().unsafe_offset(i).unsafe_store(
            val=Scalar[DType.float16](Float32(i % 100 + 1) / 100.0)
        )
    return x


def benchmark_matmul(M: Int, N: Int, K: Int):
    """Compare FP32 SIMD vs Q8_K + SDOT performance."""
    print("\n=== Benchmark [M=", M, ", N=", N, ", K=", K, "] ===")

    # Create weights
    var w = create_q4_k_weights(N, K)
    var dummy_scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))

    # Create activations
    var x = create_fp16_activations(M, K)

    # Warmup
    for _ in range(WARMUP):
        _ = matmul_quantized_cpu[DType.float16, QuantType.Q4_K_M, 32](x, w, dummy_scale)

    # Benchmark FP32 SIMD (single-threaded)
    var times_fp32 = List[Int]()
    for _ in range(REPEATS):
        var t0 = now_ns()
        var out_fp32 = matmul_quantized_cpu[DType.float16, QuantType.Q4_K_M, 32](x, w, dummy_scale)
        var t1 = now_ns()
        times_fp32.append(t1 - t0)
        _ = out_fp32

    # Benchmark FP32 SIMD (threaded, 4 threads)
    var times_fp32_t = List[Int]()
    for _ in range(REPEATS):
        var t0 = now_ns()
        var out_fp32_t = matmul_quantized_cpu_threaded[DType.float16, QuantType.Q4_K_M, 32](
            x, w, dummy_scale, nthreads=4
        )
        var t1 = now_ns()
        times_fp32_t.append(t1 - t0)
        _ = out_fp32_t

    # Benchmark Q8_K + SDOT
    var times_q8k = List[Int]()
    for _ in range(REPEATS):
        var t0 = now_ns()
        var out_q8k = matmul_quantized_q8k[QuantType.Q4_K_M](x, w, dummy_scale)
        var t1 = now_ns()
        times_q8k.append(t1 - t0)
        _ = out_q8k

    # Compute averages
    var avg_fp32 = 0
    for t in times_fp32:
        avg_fp32 += t
    avg_fp32 = avg_fp32 // REPEATS

    var avg_fp32_t = 0
    for t in times_fp32_t:
        avg_fp32_t += t
    avg_fp32_t = avg_fp32_t // REPEATS

    var avg_q8k = 0
    for t in times_q8k:
        avg_q8k += t
    avg_q8k = avg_q8k // REPEATS

    # Report
    print("  FP32 SIMD (1 thread): ", avg_fp32 / 1000, " µs")
    print("  FP32 SIMD (4 threads): ", avg_fp32_t / 1000, " µs")
    print("  Q8_K + SDOT (1 thread): ", avg_q8k / 1000, " µs")
    print("  Ratio (FP32_t/Q8K): ", Float64(avg_fp32_t) / Float64(avg_q8k))

    # Compute GFLOPS
    var flops = Float64(2 * M * N * K)
    var gflops_fp32 = flops / Float64(avg_fp32) * 1000.0
    var gflops_fp32_t = flops / Float64(avg_fp32_t) * 1000.0
    var gflops_q8k = flops / Float64(avg_q8k) * 1000.0
    print("  GFLOPS (FP32): ", gflops_fp32)
    print("  GFLOPS (FP32_t): ", gflops_fp32_t)
    print("  GFLOPS (Q8K): ", gflops_q8k)


def main():
    # Test shapes typical for 7B model
    # Small test
    benchmark_matmul(1, 256, 256)

    # Medium test
    benchmark_matmul(1, 1024, 1024)

    # Large test (like attention projection)
    benchmark_matmul(1, 4096, 4096)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# tools/bench_q8k_detailed.mojo
#
# Detailed benchmark for Q8_K + SDOT path analysis.

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.cpu.matmul_cpu import matmul_quantized_cpu, matmul_quantized_cpu_threaded
from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
from src.core.ops.cpu.simd.simd_neon import vec_dot_q4_k_q8_k
from src.core.ops.quantized.quant_types import QuantType
from src.core.thread_pool import now_ns
from std.utils.static_tuple import StaticTuple
from std.memory.alloc import unsafe_alloc
from std.math import abs

comptime QK_K = 256
comptime Q4_K_BLOCK_BYTES = 144
comptime REPEATS = 10
comptime WARMUP = 3


def create_q4_k_weights(N: Int, K: Int) -> Tensor[DType.uint8, 2]:
    """Create Q4_K quantized weights."""
    var nb = K // 256
    var total_bytes = N * nb * Q4_K_BLOCK_BYTES
    var data = unsafe_alloc[UInt8](total_bytes)
    
    for i in range(total_bytes):
        if i % 144 < 16:
            data.unsafe_offset(i).unsafe_store(val=UInt8(64))
        else:
            data.unsafe_offset(i).unsafe_store(val=UInt8(i % 256))
    
    var shape = StaticTuple[Int, 2](N, nb * Q4_K_BLOCK_BYTES)
    return Tensor[DType.uint8, 2](shape, data)


def create_fp16_activations(M: Int, K: Int) -> Tensor[DType.float16, 2]:
    """Create FP16 activations."""
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.data().unsafe_offset(i).unsafe_store(
            val=Scalar[DType.float16](Float32(i % 100 + 1) / 100.0)
        )
    return x


def benchmark_single_dot():
    """Benchmark a single Q4_K × Q8_K dot product."""
    print("=== Single Dot Product Benchmark ===")
    
    # Create one Q4_K block (144 bytes)
    var w_block = unsafe_alloc[UInt8](144)
    for i in range(144):
        if i < 16:
            w_block.unsafe_offset(i).unsafe_store(val=UInt8(64))
        else:
            w_block.unsafe_offset(i).unsafe_store(val=UInt8(i % 256))
    
    # Create one Q8_K block (292 bytes)
    var q8_block = unsafe_alloc[UInt8](292)
    # Scale
    q8_block.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(
        val=Scalar[DType.float32](1.0)
    )
    # Quantized values
    var qs = q8_block.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
    for i in range(256):
        qs.unsafe_offset(i).unsafe_store(val=Scalar[DType.int8](Int8(i % 256 - 128)))
    # Partial sums
    var bsums = q8_block.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
    for i in range(16):
        var sum = Int16(0)
        for j in range(16):
            sum += Int16(qs.unsafe_offset(i * 16 + j).unsafe_load())
        bsums.unsafe_offset(i).unsafe_store(val=Scalar[DType.int16](sum))
    
    # Benchmark
    var iters = 100000
    var t0 = now_ns()
    for _ in range(iters):
        var result = vec_dot_q4_k_q8_k(w_block, q8_block)
        _ = result
    var t1 = now_ns()
    
    var ns_per_call = (t1 - t0) // iters
    print("  Time per dot product:", ns_per_call, "ns")
    print("  Throughput:", 1000.0 / Float64(ns_per_call), "M dot/s")
    print("  Each dot processes 256 elements = 512 ops")
    print("  Throughput:", Float64(512) * 1000.0 / Float64(ns_per_call), "MOPS")


def benchmark_matmul(M: Int, N: Int, K: Int):
    """Compare all paths."""
    print("\n=== Matmul [M=", M, ", N=", N, ", K=", K, "] ===")
    
    var w = create_q4_k_weights(N, K)
    var dummy_scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))
    var x = create_fp16_activations(M, K)
    
    # FP32 SIMD single-threaded
    var times = List[Int]()
    for _ in range(REPEATS):
        var t0 = now_ns()
        var out = matmul_quantized_cpu[DType.float16, QuantType.Q4_K_M, 32](x, w, dummy_scale)
        times.append(now_ns() - t0)
        _ = out
    var avg_fp32 = 0
    for t in times:
        avg_fp32 += t
    avg_fp32 = avg_fp32 // REPEATS
    
    # FP32 SIMD threaded
    times = List[Int]()
    for _ in range(REPEATS):
        var t0 = now_ns()
        var out = matmul_quantized_cpu_threaded[DType.float16, QuantType.Q4_K_M, 32](
            x, w, dummy_scale, nthreads=4
        )
        times.append(now_ns() - t0)
        _ = out
    var avg_fp32_t = 0
    for t in times:
        avg_fp32_t += t
    avg_fp32_t = avg_fp32_t // REPEATS
    
    # Q8_K + SDOT
    times = List[Int]()
    for _ in range(REPEATS):
        var t0 = now_ns()
        var out = matmul_quantized_q8k[QuantType.Q4_K_M](x, w, dummy_scale)
        times.append(now_ns() - t0)
        _ = out
    var avg_q8k = 0
    for t in times:
        avg_q8k += t
    avg_q8k = avg_q8k // REPEATS
    
    # Report
    var flops = Float64(2 * M * N * K)
    print("  FP32 SIMD (1t):", avg_fp32 / 1000, "µs,", flops / Float64(avg_fp32) * 1000.0, "GFLOPS")
    print("  FP32 SIMD (4t):", avg_fp32_t / 1000, "µs,", flops / Float64(avg_fp32_t) * 1000.0, "GFLOPS")
    print("  Q8_K + SDOT (1t):", avg_q8k / 1000, "µs,", flops / Float64(avg_q8k) * 1000.0, "GFLOPS")
    print("  Q8_K needs", Float64(avg_fp32_t) / Float64(avg_q8k), "x faster to beat FP32_t")


def main():
    benchmark_single_dot()
    benchmark_matmul(1, 1024, 1024)
    benchmark_matmul(1, 4096, 4096)
    benchmark_matmul(1, 8192, 4096)
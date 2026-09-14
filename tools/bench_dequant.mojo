# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# tools/bench_dequant.mojo
#
# Microbenchmark for dequantization and quantized matmul performance.
# Measures:
#   1. Dequantization throughput (blocks/s, elements/s)
#   2. Quantized matmul throughput
#   3. Comparison: dequant-then-matmul vs fused quantized matmul

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.quantized.dequantize import (
    dequantize_blocks,
    dequantize_into,
    dequantize_into_f32,
)
from src.core.ops.quantized.quant_types import QuantType, block_elems, block_bytes
from src.core.ops.cpu.matmul_cpu import (
    matmul_quantized_cpu,
    matmul_weight_cpu,
)
from src.core.ops.quantized.qweight import quant_proj_dispatch
from std.time import perf_counter_ns
from std.sys import argv
from std.utils.static_tuple import StaticTuple
from std.memory.alloc import unsafe_alloc
from std.random import rand

comptime QK_K = 256
comptime Q4_K_BLOCK_BYTES = 144
comptime WARMUP = 3
comptime REPEATS = 10


def create_q4_k_block() -> Pointer[UInt8, MutUntrackedOrigin]:
    """Create a fake Q4_K block with pseudo-random data."""
    var block = unsafe_alloc[UInt8](Q4_K_BLOCK_BYTES)
    # d (fp16)
    block.unsafe_offset(0).unsafe_store(val=UInt8(0x00))
    block.unsafe_offset(1).unsafe_store(val=UInt8(0x3C))  # 1.0 in fp16
    # dmin (fp16)
    block.unsafe_offset(2).unsafe_store(val=UInt8(0x00))
    block.unsafe_offset(3).unsafe_store(val=UInt8(0x3C))
    # scales (12 bytes)
    for i in range(12):
        block.unsafe_offset(4 + i).unsafe_store(val=UInt8(64 + i))
    # qs (128 bytes)
    for i in range(128):
        block.unsafe_offset(16 + i).unsafe_store(val=UInt8(i % 256))
    return block


def bench_dequant_block() raises:
    """Benchmark dequantizing a single Q4_K block."""
    print("\n=== Dequantize Q4_K Block (256 elements) ===")

    var block = create_q4_k_block()
    var scratch = unsafe_alloc[Scalar[DType.float16]](QK_K)

    # Warmup
    for _ in range(WARMUP):
        dequantize_blocks[DType.float16, QuantType.Q4_K_M](block, 0, scratch, 1)

    # Benchmark
    var times = List[Int]()
    for _ in range(REPEATS):
        var t0 = perf_counter_ns()
        dequantize_blocks[DType.float16, QuantType.Q4_K_M](block, 0, scratch, 1)
        var t1 = perf_counter_ns()
        times.append(t1 - t0)

    var total = 0
    for t in times:
        total += t
    var avg_ns = total // REPEATS

    print("  Avg time:", avg_ns, "ns")
    print("  Throughput:", Float64(QK_K) / (Float64(avg_ns) * 1e-9) / 1e6, "M elements/s")
    print("  Throughput:", Float64(1e9) / Float64(avg_ns), "blocks/s")

    scratch.unsafe_free()
    block.unsafe_free()


def bench_dequant_weight(N: Int, K: Int) raises:
    """Benchmark dequantizing a full weight matrix [N, K]."""
    print("\n=== Dequantize Weight [", N, "x", K, "] (Q4_K_M) ===")

    var nb = K // QK_K
    var weight_bytes = N * nb * Q4_K_BLOCK_BYTES
    var weight = unsafe_alloc[UInt8](weight_bytes)

    # Fill with pseudo-random data
    for i in range(weight_bytes):
        weight.unsafe_offset(i).unsafe_store(val=UInt8(i % 256))

    var dst = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](N, K))

    # Warmup
    for _ in range(WARMUP):
        dequantize_into_f32(12, weight, 0, dst, N * K)  # 12 = GGML_Q4_K

    # Benchmark
    var times = List[Int]()
    for _ in range(REPEATS):
        var t0 = perf_counter_ns()
        dequantize_into_f32(12, weight, 0, dst, N * K)
        var t1 = perf_counter_ns()
        times.append(t1 - t0)

    var total = 0
    for t in times:
        total += t
    var avg_ns = total // REPEATS
    var avg_ms = avg_ns // 1000

    print("  Weight size:", N * K, "elements (", N * K // 1024, "K )")
    print("  Packed size:", weight_bytes // 1024, "KB")
    print("  Avg time:", avg_ms, "ms")
    print("  Throughput:", Float64(N * K) / (Float64(avg_ns) * 1e-9) / 1e6, "M elements/s")

    weight.unsafe_free()


def bench_fused_vs_separate(M: Int, N: Int, K: Int) raises:
    """Compare fused quantized matmul vs dequantize-then-matmul."""
    print("\n=== Fused vs Separate [", M, "x", K, "] @ [", N, "x", K, "]^T ===")

    # Create input tensor
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.data().unsafe_offset(i).unsafe_store(
            val=Scalar[DType.float16](Float16(i % 100) / 100.0)
        )

    # Create quantized weight
    var nb = K // QK_K
    var row_bytes = nb * Q4_K_BLOCK_BYTES
    var weight_quant = tensor_zeros[DType.uint8, 2](
        StaticTuple[Int, 2](N, row_bytes)
    )
    for i in range(N * row_bytes):
        weight_quant.data().unsafe_offset(i).unsafe_store(val=UInt8(i % 256))

    # Dummy scale (not used for Q4_K)
    var dummy_scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))

    # Method 1: Fused quantized matmul
    var times_fused = List[Int]()
    for _ in range(REPEATS):
        var t0 = perf_counter_ns()
        _ = matmul_quantized_cpu[
            DType.float16, QuantType.Q4_K_M, 32
        ](x, weight_quant, dummy_scale)
        var t1 = perf_counter_ns()
        times_fused.append(t1 - t0)

    var total_fused = 0
    for t in times_fused:
        total_fused += t
    var avg_fused = total_fused // REPEATS

    # Method 2: Dequantize then matmul
    var w_fp16 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))
    var times_separate = List[Int]()
    for _ in range(REPEATS):
        var t0 = perf_counter_ns()
        dequantize_into(12, weight_quant.data(), 0, w_fp16, N * K)  # to FP16
        _ = matmul_weight_cpu[DType.float16](x, w_fp16)
        var t1 = perf_counter_ns()
        times_separate.append(t1 - t0)

    var total_separate = 0
    for t in times_separate:
        total_separate += t
    var avg_separate = total_separate // REPEATS

    print("  Fused:    ", avg_fused // 1000, "ms")
    print("  Separate: ", avg_separate // 1000, "ms")
    print("  Speedup:  ", Float64(avg_separate) / Float64(avg_fused), "x")

    # Compute GFLOPS
    var flops = 2 * M * N * K
    print("  Fused GFLOPS:   ", Float64(flops) / (Float64(avg_fused) * 1e-9) / 1e9)
    print("  Separate GFLOPS:", Float64(flops) / (Float64(avg_separate) * 1e-9) / 1e9)


def main() raises:
    print("Dequantization Microbenchmark")
    print("Warmup:", WARMUP, "Repeats:", REPEATS)

    # 1. Single block dequantization
    bench_dequant_block()

    # 2. Full weight dequantization (typical layer sizes, K must be multiple of 256)
    bench_dequant_weight(1024, 1024)   # ~Qwen3-0.6B Q projection
    bench_dequant_weight(5376, 1536)   # DeepSeek-1.5B FFN gate

    # 3. Fused vs separate comparison (K must be multiple of 256)
    bench_fused_vs_separate(1, 1024, 1024)
    bench_fused_vs_separate(1, 5376, 1536)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# tools/bench_simd.mojo
#
# Benchmark for SIMD-optimized fused dot product kernels.
# Compares the new SIMD kernels against the original dequantize-then-dot path.

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.cpu.simd import vec_dot_q4_k, vec_dot_q8_0, TARGET_HAS_NEON
from src.core.ops.quantized.dequantize import dequantize_blocks
from src.core.ops.quantized.quant_types import QuantType
from std.time import perf_counter_ns
from std.sys import argv
from std.utils.static_tuple import StaticTuple
from std.memory.alloc import unsafe_alloc

comptime QK_K = 256
comptime Q4_K_BLOCK_BYTES = 144
comptime Q8_0_BLOCK_BYTES = 34
comptime WARMUP = 3
comptime REPEATS = 100
comptime INNER_LOOPS = 10


def create_q4_k_blocks(N: Int, nb: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
    """Create N rows of Q4_K quantized data, each with nb blocks."""
    var total_bytes = N * nb * Q4_K_BLOCK_BYTES
    var data = unsafe_alloc[UInt8](total_bytes)
    for i in range(total_bytes):
        data.unsafe_offset(i).unsafe_store(val=UInt8(i % 256))
    return data


def create_q8_0_blocks(N: Int, nb: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
    """Create N rows of Q8_0 quantized data."""
    var total_bytes = N * nb * Q8_0_BLOCK_BYTES
    var data = unsafe_alloc[UInt8](total_bytes)
    for i in range(total_bytes):
        data.unsafe_offset(i).unsafe_store(val=UInt8(i % 256))
    return data


def bench_q4_k_dot(N: Int, K: Int) raises:
    """Benchmark Q4_K fused dot product."""
    print("\n=== Q4_K Dot Product [N=", N, ", K=", K, "] ===")

    if K % 256 != 0:
        print("  SKIP: K must be multiple of 256")
        return

    var nb = K // 256
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, K))
    for i in range(K):
        x.data().unsafe_offset(i).unsafe_store(val=Scalar[DType.float16](Float16(i % 100) / 100.0))

    var blocks = create_q4_k_blocks(N, nb)

    print("  Platform:", "NEON" if TARGET_HAS_NEON else "Scalar")

    # Benchmark SIMD fused dot product
    var times_simd = List[Int]()
    var sink = Float32(0)  # Prevent optimization
    for _ in range(REPEATS):
        var t0 = perf_counter_ns()
        for _ in range(INNER_LOOPS):
            for j in range(N):
                sink += vec_dot_q4_k[DType.float16](x.data(), blocks.unsafe_offset(j * nb * Q4_K_BLOCK_BYTES), nb)
        var t1 = perf_counter_ns()
        times_simd.append(t1 - t0)
    # Use sink to prevent optimization
    if sink > 1e30:
        print("  [sink check]")

    var total_simd = 0
    for t in times_simd:
        total_simd += t
    var avg_simd = total_simd // REPEATS

    # Benchmark original dequantize-then-dot
    var scratch = unsafe_alloc[Scalar[DType.float16]](256)
    var times_orig = List[Int]()
    var sink2 = Float32(0)  # Prevent optimization
    for _ in range(REPEATS):
        var t0 = perf_counter_ns()
        for _ in range(INNER_LOOPS):
            for j in range(N):
                var acc = Float32(0)
                var k = 0
                for blk in range(nb):
                    dequantize_blocks[DType.float16, QuantType.Q4_K_M](
                        blocks.unsafe_offset(j * nb * Q4_K_BLOCK_BYTES), blk * Q4_K_BLOCK_BYTES, scratch, 1
                    )
                    for i in range(256):
                        var xv = Float32(x.data().unsafe_load(offset=k + i))
                        var wv = Float32(scratch.unsafe_load(offset=i))
                        acc += xv * wv
                    k += 256
                sink2 += acc
        var t1 = perf_counter_ns()
        times_orig.append(t1 - t0)
    # Use sink2 to prevent optimization
    if sink2 > 1e30:
        print("  [sink2 check]")

    var total_orig = 0
    for t in times_orig:
        total_orig += t
    var avg_orig = total_orig // REPEATS

    print("  SIMD fused:  ", Float64(avg_simd) / 1_000_000.0, "ms")
    print("  Dequant+dot: ", Float64(avg_orig) / 1_000_000.0, "ms")
    print("  Speedup:     ", Float64(avg_orig) / Float64(avg_simd), "x")

    # Compute GFLOPS (2 * N * K FLOPs per row * INNER_LOOPS)
    var flops = Float64(2 * N * K * INNER_LOOPS)
    print("  SIMD GFLOPS: ", flops / (Float64(avg_simd) * 1e-9) / 1e9)

    blocks.unsafe_free()
    scratch.unsafe_free()


def bench_q8_0_dot(N: Int, K: Int) raises:
    """Benchmark Q8_0 fused dot product."""
    print("\n=== Q8_0 Dot Product [N=", N, ", K=", K, "] ===")

    if K % 32 != 0:
        print("  SKIP: K must be multiple of 32")
        return

    var nb = K // 32
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, K))
    for i in range(K):
        x.data().unsafe_offset(i).unsafe_store(val=Scalar[DType.float16](Float16(i % 100) / 100.0))

    var blocks = create_q8_0_blocks(N, nb)

    # Benchmark SIMD fused dot product
    var times_simd = List[Int]()
    var sink = Float32(0)  # Prevent optimization
    for _ in range(REPEATS):
        var t0 = perf_counter_ns()
        for _ in range(INNER_LOOPS):
            for j in range(N):
                sink += vec_dot_q8_0[DType.float16](x.data(), blocks.unsafe_offset(j * nb * Q8_0_BLOCK_BYTES), nb)
        var t1 = perf_counter_ns()
        times_simd.append(t1 - t0)
    # Use sink to prevent optimization
    if sink > 1e30:
        print("  [sink check]")

    var total_simd = 0
    for t in times_simd:
        total_simd += t
    var avg_simd = total_simd // REPEATS

    print("  SIMD fused:  ", Float64(avg_simd) / 1_000_000.0, "ms")
    var flops = Float64(2 * N * K * INNER_LOOPS)
    print("  GFLOPS:      ", flops / (Float64(avg_simd) * 1e-9) / 1e9)

    blocks.unsafe_free()


def main() raises:
    print("SIMD Fused Dot Product Benchmark")
    print("Warmup:", WARMUP, "Repeats:", REPEATS)

    # Q4_K benchmarks (K must be multiple of 256)
    bench_q4_k_dot(1024, 1024)
    bench_q4_k_dot(5376, 1536)

    # Q8_0 benchmarks (K must be multiple of 32)
    bench_q8_0_dot(1024, 1024)
    bench_q8_0_dot(5376, 1536)
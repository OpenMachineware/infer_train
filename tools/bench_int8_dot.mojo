# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# tools/bench_int8_dot.mojo
#
# Benchmark for hardware int8 dot product using NEON intrinsics.

from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc
from std.time import perf_counter_ns
from std.sys import argv
from std.sys import llvm_intrinsic

comptime NEON_WIDTH = 8
comptime DOT_WIDTH = 4


def _neon_dotprod_int8[
    width: SIMDLength = DOT_WIDTH
](
    c: SIMD[DType.int32, width],
    a: SIMD[DType.int8, width * 4],
    b: SIMD[DType.int8, width * 4],
) -> SIMD[DType.int32, width]:
    """NEON int8 dot product accumulate (SDOT)."""
    comptime assert width == 4
    return llvm_intrinsic[
        "llvm.aarch64.neon.sdot.v4i32.v16i8",
        SIMD[DType.int32, width],
    ](c, a, b)


def bench_int8_dot[N: Int, K: Int](warmup: Int, repeats: Int):
    """Benchmark int8 dot product."""
    print("=== Int8 Dot Product [N=", N, ", K=", K, "] ===")

    # Allocate int8 data
    var size = N * K
    var a_ptr = unsafe_alloc[Int8](size)
    var b_ptr = unsafe_alloc[Int8](size)

    # Initialize with random-ish data
    for i in range(size):
        a_ptr.unsafe_store(i, Int8(i % 256 - 128))
        b_ptr.unsafe_store(i, Int8((i * 7) % 256 - 128))

    # Warmup
    var sink = 0.0
    for w in range(warmup):
        var acc = SIMD[DType.int32, DOT_WIDTH](0)
        for i in range(N):
            for j in range(K // 16):
                var a = a_ptr.unsafe_load[width=16](offset=i * K + j * 16).cast[DType.int8]()
                var b = b_ptr.unsafe_load[width=16](offset=i * K + j * 16).cast[DType.int8]()
                acc = _neon_dotprod_int8(acc, a, b)
        sink = sink + Float64(acc.reduce_add())

    # Benchmark
    var start = perf_counter_ns()
    for r in range(repeats):
        var acc = SIMD[DType.int32, DOT_WIDTH](0)
        for i in range(N):
            for j in range(K // 16):
                var a = a_ptr.unsafe_load[width=16](offset=i * K + j * 16).cast[DType.int8]()
                var b = b_ptr.unsafe_load[width=16](offset=i * K + j * 16).cast[DType.int8]()
                acc = _neon_dotprod_int8(acc, a, b)
        sink = sink + Float64(acc.reduce_add())
    var end = perf_counter_ns()

    var elapsed_s = Float64(end - start) / 1e9
    var avg_time_ms = Float32(elapsed_s) * 1000.0 / Float32(repeats)

    # Calculate GFLOPS: N * K MACs per iteration
    var macs = Float64(N * K)
    var gflops = macs / (elapsed_s * 1e9) * Float64(repeats)

    print("  Time:       ", avg_time_ms, " ms")
    print("  GFLOPS:     ", Float32(gflops))
    print("  Sink:       ", sink)


def main() raises:
    var args = argv()
    var warmup = 3
    var repeats = 100
    if len(args) > 1:
        warmup = Int(args[1])
    if len(args) > 2:
        repeats = Int(args[2])

    print("Hardware Int8 Dot Product Benchmark")
    print("Warmup:", warmup, "Repeats:", repeats)
    print()

    bench_int8_dot[1024, 1024](warmup, repeats)
    print()
    bench_int8_dot[5376, 1536](warmup, repeats)

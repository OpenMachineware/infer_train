# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# Benchmark for IQ3_XXS × Q8_K kernel

from src.core.ops.cpu.simd.iq3xxs_q8k_neon import vec_dot_iq3xxs_q8k_neon
from std.time import perf_counter_ns
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.origin import MutUntrackedOrigin

comptime QK_K = 256
comptime NBLOCKS = 1000

def main():
    print("=== IQ3_XXS × Q8_K NEON Benchmark ===")
    print("Block size: 98 bytes (3.06 bpw)")
    print("Grid: 256 entries")
    print()

    # Allocate test data
    var x = unsafe_alloc[UInt8](NBLOCKS * 98, alignment=64)
    var y = unsafe_alloc[UInt8](NBLOCKS * 292, alignment=64)

    # Initialize IQ3_XXS data (98 bytes per block)
    for i in range(NBLOCKS):
        var x_base = i * 98
        # d = 1.0 (FP16)
        x.unsafe_offset(x_base).unsafe_bitcast[Scalar[DType.float16]]().unsafe_store(Scalar[DType.float16](1.0))

        # qs[64]: grid indices (random values 0-255)
        for j in range(64):
            x.unsafe_store(offset=x_base + 2 + j, val=UInt8((i + j) % 256))

        # scales_and_signs[32]: packed scales and signs
        for j in range(32):
            # Each byte encodes part of the uint32
            x.unsafe_store(offset=x_base + 66 + j, val=UInt8((i * 31 + j) % 256))

    # Initialize Q8_K data (292 bytes per block)
    for i in range(NBLOCKS):
        var y_base = i * 292
        # d = 1.0 (FP32)
        y.unsafe_offset(y_base).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(Scalar[DType.float32](1.0))
        # qs[256]: random int8 values
        for j in range(256):
            y.unsafe_store(offset=y_base + 4 + j, val=UInt8((i + j) % 256))

    # Warmup
    var _ = vec_dot_iq3xxs_q8k_neon(x, y, NBLOCKS)

    # Benchmark
    var start = perf_counter_ns()
    var sumf = Float32(0)
    for _ in range(10):
        sumf += vec_dot_iq3xxs_q8k_neon(x, y, NBLOCKS)
    var end = perf_counter_ns()

    var elapsed_ns = end - start
    var elapsed_seconds = Float64(elapsed_ns) / 1e9 / 10.0

    # Calculate GFLOPS
    # Each block: 256 multiply-adds = 512 FLOPs
    var flops = Float64(NBLOCKS) * 512.0
    var gflops = flops / elapsed_seconds / 1e9

    print("Blocks:", NBLOCKS)
    print("Time per iteration:", elapsed_seconds * 1000.0, "ms")
    print("GFLOPS:", gflops)
    print("Result:", sumf)

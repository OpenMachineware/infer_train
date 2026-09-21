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

    # Initialize IQ3_XXS data (98 bytes per block, matching llama.cpp exactly)
    for i in range(NBLOCKS):
        var x_base = i * 98
        # d = 1.0 (FP16) - manual store
        # FP16 1.0 = 0x3C00 (little-endian: 0x00, 0x3C)
        x.unsafe_store(offset=x_base, val=UInt8(0x00))
        x.unsafe_store(offset=x_base + 1, val=UInt8(0x3C))

        # qs[96]: match llama.cpp - x[i].qs[j] = (uint8_t)((i + j) % 256)
        for j in range(96):
            x.unsafe_store(offset=x_base + 2 + j, val=UInt8((i + j) % 256))

    # Initialize Q8_K data (292 bytes per block, matching llama.cpp exactly)
    for i in range(NBLOCKS):
        var y_base = i * 292
        # d = 1.0 (FP32)
        y.unsafe_offset(y_base).unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(Scalar[DType.float32](1.0))
        # qs[256]: match llama.cpp - y[i].qs[j] = (int8_t)((j + 1) % 256)
        for j in range(256):
            var val = (j + 1) % 256
            y.unsafe_store(offset=y_base + 4 + j, val=UInt8(val))

        # bsums[16]: match llama.cpp - calculate sum for each group
        for j in range(16):
            var sum = 0
            for k in range(16):
                var idx = j * 16 + k
                var val = (idx + 1) % 256
                var signed_val = val if val < 128 else val - 256
                sum += signed_val
            y.unsafe_offset(y_base + 260 + j * 2).unsafe_bitcast[Scalar[DType.int16]]().unsafe_store(
                Scalar[DType.int16](Int16(sum))
            )

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

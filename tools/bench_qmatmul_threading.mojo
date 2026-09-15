# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# tools/bench_qmatmul_threading.mojo
#
# Benchmark for quantized matmul with different thread counts.
# Measures the actual speedup from pthread parallelization.

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.cpu.matmul_cpu import (
    matmul_quantized_cpu,
    matmul_quantized_cpu_threaded,
)
from src.core.ops.quantized.quant_types import QuantType
from src.core.thread_pool import now_ns, num_pcores
from std.utils.static_tuple import StaticTuple
from std.memory.alloc import unsafe_alloc
from std.sys import argv

comptime QK_K = 256
comptime Q4_K_BLOCK_BYTES = 144
comptime REPEATS = 20
comptime WARMUP = 3


def create_q4_k_weights(N: Int, K: Int) -> Tensor[DType.uint8, 2]:
    """Create Q4_K quantized weights [N, K/256*144 bytes]."""
    var nb = K // 256
    var total_bytes = N * nb * Q4_K_BLOCK_BYTES
    var data = unsafe_alloc[UInt8](total_bytes)
    for i in range(total_bytes):
        data.unsafe_offset(i).unsafe_store(val=UInt8(i % 256))

    # Create tensor view
    var shape = StaticTuple[Int, 2](N, nb * Q4_K_BLOCK_BYTES)
    var tensor = Tensor[DType.uint8, 2](shape, data)
    return tensor


def bench_q4_k_matmul(M: Int, N: Int, K: Int, nthreads: Int) raises:
    """Benchmark Q4_K matmul with threading."""
    print("\n=== Q4_K Matmul [M=", M, ", N=", N, ", K=", K, ", threads=", nthreads, "] ===")

    # Create activation tensor
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    for i in range(M * K):
        x.data().unsafe_offset(i).unsafe_store(
            val=Scalar[DType.float16](Float32(i % 100) / 100.0)
        )

    # Create quantized weights
    var w = create_q4_k_weights(N, K)
    var dummy_scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))

    # Warmup
    for _ in range(WARMUP):
        _ = matmul_quantized_cpu_threaded[DType.float16, QuantType.Q4_K_M, 32](
            x, w, dummy_scale, nthreads=nthreads
        )

    # Benchmark single-threaded
    var times_single = List[Int]()
    for _ in range(REPEATS):
        var t0 = now_ns()
        var out = matmul_quantized_cpu[DType.float16, QuantType.Q4_K_M, 32](x, w, dummy_scale)
        var t1 = now_ns()
        times_single.append(t1 - t0)
        _ = out  # Prevent optimization

    var total_single = 0
    for t in times_single:
        total_single += t
    var avg_single = total_single // REPEATS

    # Benchmark multi-threaded
    var times_threaded = List[Int]()
    for _ in range(REPEATS):
        var t0 = now_ns()
        var out = matmul_quantized_cpu_threaded[DType.float16, QuantType.Q4_K_M, 32](
            x, w, dummy_scale, nthreads=nthreads
        )
        var t1 = now_ns()
        times_threaded.append(t1 - t0)
        _ = out  # Prevent optimization

    var total_threaded = 0
    for t in times_threaded:
        total_threaded += t
    var avg_threaded = total_threaded // REPEATS

    # Report results
    print("  Single-threaded: ", avg_single / 1000, " µs")
    print("  Threaded (", nthreads, "): ", avg_threaded / 1000, " µs")
    print("  Speedup: ", Float64(avg_single) / Float64(avg_threaded))

    # Calculate GFLOPS
    var flops = Float64(2 * M * N * K)  # multiply-add
    var gflops_single = flops / Float64(avg_single) * 1000.0
    var gflops_threaded = flops / Float64(avg_threaded) * 1000.0
    print("  GFLOPS (single): ", gflops_single)
    print("  GFLOPS (threaded): ", gflops_threaded)


def main() raises:
    var arg_list = List[String]()
    for a in argv():
        arg_list.append(String(a))

    var nthreads = num_pcores()  # Default to performance cores
    if len(arg_list) >= 2:
        var b = arg_list[1].as_bytes()
        var parsed = 0
        var ok = True
        for i in range(len(b)):
            var c = Int(b[i])
            if c >= 48 and c <= 57:
                parsed = parsed * 10 + (c - 48)
            else:
                ok = False
        if ok and parsed > 0:
            nthreads = parsed

    print("Performance cores: ", num_pcores())
    print("Testing with ", nthreads, " threads")

    # Test shapes typical for 7B model projections
    # QKV projection: [1, 4096] x [12288, 4096]
    # FFN up/gate: [1, 4096] x [22016, 4096]
    # Output: [1, 4096] x [4096, 4096]

    # Small test (quick)
    bench_q4_k_matmul(1, 1024, 1024, nthreads)

    # Medium test (like attention projection)
    bench_q4_k_matmul(1, 4096, 4096, nthreads)

    # Large test (like FFN projection)
    bench_q4_k_matmul(1, 8192, 4096, nthreads)

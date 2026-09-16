# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# tests/test_gpu_matmul_basic.mojo
#
# Basic GPU matmul test to verify the naive kernel works.

from src.core.device import has_metal_gpu
from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.gpu.matmul_gpu import matmul_gpu_dynamic
from src.core.ops.cpu.matmul_cpu import matmul_cpu_dynamic
from std.utils.static_tuple import StaticTuple


def main() raises:
    if not has_metal_gpu():
        print("SKIP: no Metal GPU")
        return

    print("Testing GPU matmul (naive kernel)...")

    # Create small test matrices
    comptime M = 2
    comptime K = 64
    comptime N = 32

    var a = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, K))
    var b = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](K, N))

    # Fill with test values
    for i in range(M * K):
        a.set(i, Scalar[DType.float16](Float32(i % 10) * 0.1))
    for i in range(K * N):
        b.set(i, Scalar[DType.float16](Float32((i % 7) + 1) * 0.05))

    # CPU reference
    var cpu_out = matmul_cpu_dynamic[DType.float16](a, b)
    print("CPU output shape:", cpu_out.shape()[0], "x", cpu_out.shape()[1])

    # GPU test
    var gpu_out = matmul_gpu_dynamic[DType.float16](a, b)
    print("GPU output shape:", gpu_out.shape()[0], "x", gpu_out.shape()[1])

    # Compare
    var max_diff = Float32(0.0)
    for i in range(M * N):
        var diff = abs(Float32(cpu_out.get(i)) - Float32(gpu_out.get(i)))
        if diff > max_diff:
            max_diff = diff

    print("Max diff:", max_diff)
    if max_diff > Float32(0.001):
        print("FAIL: GPU output differs from CPU")
        raise Error("GPU matmul correctness failed")

    print("PASS: GPU matmul matches CPU output")

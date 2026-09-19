# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/matmul_fp_optimized.mojo
#
# Optimized FP16/FP32 matmul using llama.cpp strategy:
# 1. Transpose B matrix for contiguous memory access
# 2. Row × Row vector dot product with SIMD
# 3. Use FP16 FMA instructions for maximum throughput

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.utils.static_tuple import StaticTuple
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc

comptime W_F16 = 8  # 8 x f16 = 128-bit NEON vector
comptime W_F32 = 4  # 4 x f32 = 128-bit NEON vector


def transpose_f16(
    b: Tensor[DType.float16, 2]
) -> Tensor[DType.float16, 2]:
    """Transpose a FP16 matrix.

    Input: K x N matrix
    Output: N x K matrix (transposed)
    """
    var K = b.shape()[0]
    var N = b.shape()[1]
    var out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))

    # Simple transpose (can be optimized with SIMD later if needed)
    for i in range(K):
        for j in range(N):
            out.set(j * K + i, b.get(i * N + j))

    return out


def transpose_f32(
    b: Tensor[DType.float32, 2]
) -> Tensor[DType.float32, 2]:
    """Transpose a FP32 matrix.

    Input: K x N matrix
    Output: N x K matrix (transposed)
    """
    var K = b.shape()[0]
    var N = b.shape()[1]
    var out = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](N, K))

    for i in range(K):
        for j in range(N):
            out.set(j * K + i, b.get(i * N + j))

    return out


def vec_dot_f16(
    x: Pointer[Scalar[DType.float16], MutUntrackedOrigin],
    y: Pointer[Scalar[DType.float16], MutUntrackedOrigin],
    n: Int,
) -> Float32:
    """FP16 vector dot product using SIMD.

    Matches llama.cpp's ggml_vec_dot_f16 (vec.cpp:264-378).
    Process 32 elements at a time (4 x float16x8_t vectors).
    """
    var np = (n // 32) * 32  # 32 elements at a time

    # 4 accumulators (matching llama.cpp's GGML_F16_ARR = 4)
    var sum0 = SIMD[DType.float16, W_F16](0)
    var sum1 = SIMD[DType.float16, W_F16](0)
    var sum2 = SIMD[DType.float16, W_F16](0)
    var sum3 = SIMD[DType.float16, W_F16](0)

    var i = 0
    while i < np:
        # Load 4 vectors of 8 FP16 each
        var ax0 = x.unsafe_load[width=W_F16](offset=i)
        var ay0 = y.unsafe_load[width=W_F16](offset=i)
        var ax1 = x.unsafe_load[width=W_F16](offset=i + W_F16)
        var ay1 = y.unsafe_load[width=W_F16](offset=i + W_F16)
        var ax2 = x.unsafe_load[width=W_F16](offset=i + 2 * W_F16)
        var ay2 = y.unsafe_load[width=W_F16](offset=i + 2 * W_F16)
        var ax3 = x.unsafe_load[width=W_F16](offset=i + 3 * W_F16)
        var ay3 = y.unsafe_load[width=W_F16](offset=i + 3 * W_F16)

        # FMA (fused multiply-add) - uses vfmaq_f16 instruction
        sum0 = sum0 + ax0 * ay0
        sum1 = sum1 + ax1 * ay1
        sum2 = sum2 + ax2 * ay2
        sum3 = sum3 + ax3 * ay3

        i += 32

    # Reduce: sum0..sum3 to sum0
    sum0 = sum0 + sum1
    sum2 = sum2 + sum3
    sum0 = sum0 + sum2

    # Convert to FP32 and sum
    var sum_f32 = sum0.cast[DType.float32]()
    var result = sum_f32.reduce_add()

    # Handle leftovers
    while i < n:
        var a_val = Float32(x.unsafe_load(offset=i))
        var b_val = Float32(y.unsafe_load(offset=i))
        result += a_val * b_val
        i += 1

    return result


def vec_dot_f32(
    x: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
    y: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
    n: Int,
) -> Float32:
    """FP32 vector dot product using SIMD.

    Process 16 elements at a time (4 x float32x4_t vectors).
    """
    var np = (n // 16) * 16  # 16 elements at a time

    var sum0 = SIMD[DType.float32, W_F32](0)
    var sum1 = SIMD[DType.float32, W_F32](0)
    var sum2 = SIMD[DType.float32, W_F32](0)
    var sum3 = SIMD[DType.float32, W_F32](0)

    var i = 0
    while i < np:
        var ax0 = x.unsafe_load[width=W_F32](offset=i)
        var ay0 = y.unsafe_load[width=W_F32](offset=i)
        var ax1 = x.unsafe_load[width=W_F32](offset=i + W_F32)
        var ay1 = y.unsafe_load[width=W_F32](offset=i + W_F32)
        var ax2 = x.unsafe_load[width=W_F32](offset=i + 2 * W_F32)
        var ay2 = y.unsafe_load[width=W_F32](offset=i + 2 * W_F32)
        var ax3 = x.unsafe_load[width=W_F32](offset=i + 3 * W_F32)
        var ay3 = y.unsafe_load[width=W_F32](offset=i + 3 * W_F32)

        sum0 = sum0 + ax0 * ay0
        sum1 = sum1 + ax1 * ay1
        sum2 = sum2 + ax2 * ay2
        sum3 = sum3 + ax3 * ay3

        i += 16

    sum0 = sum0 + sum1
    sum2 = sum2 + sum3
    sum0 = sum0 + sum2

    var result = sum0.reduce_add()

    while i < n:
        var a_val = x.unsafe_load(offset=i)
        var b_val = y.unsafe_load(offset=i)
        result += a_val * b_val
        i += 1

    return result


def matmul_f16_optimized(
    a: Tensor[DType.float16, 2], b: Tensor[DType.float16, 2]
) -> Tensor[DType.float16, 2]:
    """Optimized FP16 matmul using transpose + row×row dot product.

    Matches llama.cpp's strategy:
    1. Transpose B matrix
    2. Compute row × row dot products (contiguous memory access)
    3. Use SIMD vector dot product with FP16 FMA
    """
    var M = a.shape()[0]
    var K = a.shape()[1]
    var N = b.shape()[1]
    if K != b.shape()[0]:
        unimplemented("matmul_f16_optimized: K mismatch")

    # Transpose B for contiguous memory access
    var b_transposed = transpose_f16(b)  # N x K

    var out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, N))

    # Row × Row dot product
    for i in range(M):
        for j in range(N):
            var sum = vec_dot_f16(
                a.data().unsafe_bitcast[Scalar[DType.float16]]() + i * K,
                b_transposed.data().unsafe_bitcast[Scalar[DType.float16]]() + j * K,
                K,
            )
            out.set(i * N + j, Scalar[DType.float16](sum))

    return out


def matmul_f16_pretransposed(
    a: Tensor[DType.float16, 2], b_transposed: Tensor[DType.float16, 2]
) -> Tensor[DType.float16, 2]:
    """FP16 matmul with pre-transposed B matrix.

    Use this for inference where B (weights) is constant.
    Pre-transpose B once before inference starts.

    Args:
        a: M x K activation matrix
        b_transposed: N x K transposed weight matrix (call transpose_f16 once)
    Returns:
        M x N output matrix
    """
    var M = a.shape()[0]
    var K = a.shape()[1]
    var N = b_transposed.shape()[0]

    var out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, N))

    for i in range(M):
        for j in range(N):
            var sum = vec_dot_f16(
                a.data().unsafe_bitcast[Scalar[DType.float16]]() + i * K,
                b_transposed.data().unsafe_bitcast[Scalar[DType.float16]]() + j * K,
                K,
            )
            out.set(i * N + j, Scalar[DType.float16](sum))

    return out


def matmul_f32_optimized(
    a: Tensor[DType.float32, 2], b: Tensor[DType.float32, 2]
) -> Tensor[DType.float32, 2]:
    """Optimized FP32 matmul using transpose + row×row dot product."""
    var M = a.shape()[0]
    var K = a.shape()[1]
    var N = b.shape()[1]
    if K != b.shape()[0]:
        unimplemented("matmul_f32_optimized: K mismatch")

    var b_transposed = transpose_f32(b)

    var out = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](M, N))

    for i in range(M):
        for j in range(N):
            var sum = vec_dot_f32(
                a.data() + i * K,
                b_transposed.data() + j * K,
                K,
            )
            out.set(i * N + j, Scalar[DType.float32](sum))

    return out


def matmul_f32_pretransposed(
    a: Tensor[DType.float32, 2], b_transposed: Tensor[DType.float32, 2]
) -> Tensor[DType.float32, 2]:
    """FP32 matmul with pre-transposed B matrix.

    Use this for inference where B (weights) is constant.
    """
    var M = a.shape()[0]
    var K = a.shape()[1]
    var N = b_transposed.shape()[0]

    var out = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](M, N))

    for i in range(M):
        for j in range(N):
            var sum = vec_dot_f32(
                a.data() + i * K,
                b_transposed.data() + j * K,
                K,
            )
            out.set(i * N + j, Scalar[DType.float32](sum))

    return out

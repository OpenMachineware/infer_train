# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/blas_cpu.mojo
#
# Apple Accelerate BLAS wrapper for optimized matrix operations.
# Uses cblas_sgemm for FP32 matmul, providing 2-3x speedup over
# hand-written SIMD kernels on Apple Silicon.
#
# Link with -Xlinker "-framework" -Xlinker "Accelerate"

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.utils.static_tuple import StaticTuple
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc, unsafe_free

# Apple Accelerate BLAS constants
comptime CblasRowMajor: Int32 = 101
comptime CblasNoTrans: Int32 = 111
comptime CblasTrans: Int32 = 112

# Tile size for tiled dequantize+BLAS: dequantize this many weight rows at a time
comptime BLAS_TILE_ROWS = 256


def matmul_blas_f32(
    a: Tensor[DType.float32, 2], b: Tensor[DType.float32, 2]
) -> Tensor[DType.float32, 2]:
    """y = a @ b using Accelerate cblas_sgemm.

    Both inputs are row-major [M, K] and [K, N].
    Output is [M, N].
    """
    var M = a.shape()[0]
    var K = a.shape()[1]
    var N = b.shape()[1]
    if K != b.shape()[0]:
        unimplemented("matmul_blas: K mismatch")

    var out = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](M, N))

    # cblas_sgemm: C = alpha * op(A) * op(B) + beta * C
    # For C = A @ B (row-major):
    #   order = RowMajor
    #   transA = NoTrans, transB = NoTrans
    #   A is [M, K], B is [K, N], C is [M, N]
    #   lda = K, ldb = N, ldc = N
    # Note: cblas_sgemm returns void, but external_call requires a RegisterPassable
    # return type, so we use Int32 and ignore the result.
    _ = external_call[
        "cblas_sgemm",
        Int32,
        Int32, Int32, Int32, Int32, Int32, Int32,
        Float32, Pointer[Float32, MutUntrackedOrigin], Int32,
        Pointer[Float32, MutUntrackedOrigin], Int32,
        Float32, Pointer[Float32, MutUntrackedOrigin], Int32,
    ](
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        Int32(M),
        Int32(N),
        Int32(K),
        Float32(1.0),
        a.data(),
        Int32(K),
        b.data(),
        Int32(N),
        Float32(0.0),
        out.data(),
        Int32(N),
    )
    return out


def matmul_weight_blas_f32(
    x: Tensor[DType.float32, 2], w: Tensor[DType.float32, 2]
) -> Tensor[DType.float32, 2]:
    """y = x @ w.T using Accelerate cblas_sgemm.

    Weight layout: w is [N, K] (output dim first).
    Computation: y[i, j] = sum_k x[i, k] * w[j, k]
    This is equivalent to y = x @ w.T where w.T is [K, N].

    Output is [M, N] where M = x.shape()[0], N = w.shape()[0].
    """
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]
    if K != w.shape()[1]:
        unimplemented("matmul_weight_blas: K mismatch")

    var out = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](M, N))

    # y = x @ w.T
    # cblas_sgemm: C = A @ B^T
    #   transA = NoTrans (A = x, [M, K])
    #   transB = Trans (B = w, stored as [N, K], treat as [K, N])
    #   lda = K (x stride)
    #   ldb = K (w stride - K is the leading dim of w before transpose)
    #   ldc = N (output stride)
    _ = external_call[
        "cblas_sgemm",
        Int32,
        Int32, Int32, Int32, Int32, Int32, Int32,
        Float32, Pointer[Float32, MutUntrackedOrigin], Int32,
        Pointer[Float32, MutUntrackedOrigin], Int32,
        Float32, Pointer[Float32, MutUntrackedOrigin], Int32,
    ](
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        Int32(M),
        Int32(N),
        Int32(K),
        Float32(1.0),
        x.data(),
        Int32(K),
        w.data(),
        Int32(K),  # ldb = K for transposed B
        Float32(0.0),
        out.data(),
        Int32(N),
    )
    return out


def matmul_blas[
    dtype: DType
](a: Tensor[dtype, 2], b: Tensor[dtype, 2]) -> Tensor[dtype, 2]:
    """Dispatch to BLAS matmul for supported dtypes."""
    comptime if dtype == DType.float32:
        var a32 = Tensor[DType.float32, 2](
            a.shape(),
            a.data().unsafe_bitcast[Scalar[DType.float32]](),
            a.device(),
        )
        var b32 = Tensor[DType.float32, 2](
            b.shape(),
            b.data().unsafe_bitcast[Scalar[DType.float32]](),
            b.device(),
        )
        var out = matmul_blas_f32(a32, b32)
        return Tensor[dtype, 2](
            out.shape(),
            out.data().unsafe_bitcast[Scalar[dtype]](),
            out.device(),
        )
    else:
        # FP16 not supported by Accelerate BLAS, fall back to SIMD
        unimplemented("matmul_blas: only FP32 supported")
        return tensor_zeros[dtype, 2](StaticTuple[Int, 2](0, 0))


def matmul_weight_blas[
    dtype: DType
](x: Tensor[dtype, 2], w: Tensor[dtype, 2]) -> Tensor[dtype, 2]:
    """Dispatch to BLAS weight-major matmul for supported dtypes."""
    comptime if dtype == DType.float32:
        var x32 = Tensor[DType.float32, 2](
            x.shape(),
            x.data().unsafe_bitcast[Scalar[DType.float32]](),
            x.device(),
        )
        var w32 = Tensor[DType.float32, 2](
            w.shape(),
            w.data().unsafe_bitcast[Scalar[DType.float32]](),
            w.device(),
        )
        var out = matmul_weight_blas_f32(x32, w32)
        return Tensor[dtype, 2](
            out.shape(),
            out.data().unsafe_bitcast[Scalar[dtype]](),
            out.device(),
        )
    else:
        # FP16 not supported by Accelerate BLAS, fall back to SIMD
        unimplemented("matmul_weight_blas: only FP32 supported")
        return tensor_zeros[dtype, 2](StaticTuple[Int, 2](0, 0))
# core/ops/fused/matmul_rms_norm.mojo
#
# M5 fused kernel: matmul + RMSNorm in one kernel.
#
#   fused_matmul_rms_norm(x, w, eps):  y = rms_norm(x @ w^T, eps)
#
# The projection result row is written once, then normalized in place:
# no intermediate tensor is materialized and the output buffer is touched
# twice instead of (alloc + write + read + alloc + write + read).  The dot
# products accumulate in f32 (M3 numerics rule); the norm is computed in
# f32 like the standalone rms_norm kernel.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.utils.static_tuple import StaticTuple
from std.math import sqrt

comptime W_F16 = 8
comptime W_F32 = 4


def _fused_matmul_rms_norm_kernel[dtype: DType](
    x: Tensor[dtype, 2],
    w: Tensor[dtype, 2],
    eps: Float32,
) -> Tensor[dtype, 2]:
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]
    if K != w.shape()[1]:
        unimplemented("fused_matmul_rms_norm: K mismatch")
    var out = tensor_zeros[dtype, 2](StaticTuple[Int, 2](M, N))
    comptime W = W_F16 if dtype == DType.float16 else W_F32
    var k_main = (K // W) * W
    var n_main = (N // W) * W
    for i in range(M):
        # pass 1: y[i, :] = x[i, :] @ w^T (dot products, f32 accumulation)
        var j = 0
        while j < N:
            var acc = SIMD[DType.float32, W](0)
            var k = 0
            while k < k_main:
                var xv = x.data().unsafe_load[width=W](
                    offset=i * K + k
                ).cast[DType.float32]()
                var wv = w.data().unsafe_load[width=W](
                    offset=j * K + k
                ).cast[DType.float32]()
                acc = acc + xv * wv
                k += W
            var total = Float32(acc.reduce_add())
            while k < K:
                total += Float32(x.get(i * K + k)) * Float32(
                    w.get(j * K + k)
                )
                k += 1
            out.set(i * N + j, Scalar[dtype](total))
            j += 1
        # pass 2: RMSNorm the row in place
        var ss = Float32(0)
        j = 0
        while j < n_main:
            var v = out.data().unsafe_load[width=W](
                offset=i * N + j
            ).cast[DType.float32]()
            ss += Float32((v * v).reduce_add())
            j += W
        while j < N:
            var v = Float32(out.get(i * N + j))
            ss += v * v
            j += 1
        var inv = Float32(1.0) / sqrt(ss / Float32(N) + eps)
        j = 0
        while j < n_main:
            var v = out.data().unsafe_load[width=W](
                offset=i * N + j
            ).cast[DType.float32]()
            out.data().unsafe_store(
                i * N + j, (v * SIMD[DType.float32, W](inv)).cast[dtype]()
            )
            j += W
        while j < N:
            var v = Float32(out.get(i * N + j))
            out.set(i * N + j, Scalar[dtype](v * inv))
            j += 1
    return out


def _dispatch[dtype: DType](
    x: Tensor[dtype, 2], w: Tensor[dtype, 2], eps: Float32
) -> Tensor[dtype, 2]:
    comptime if dtype == DType.float16:
        var x16 = Tensor[DType.float16, 2](
            x.shape(),
            x.data().unsafe_bitcast[Scalar[DType.float16]](),
            x.device(),
        )
        var w16 = Tensor[DType.float16, 2](
            w.shape(),
            w.data().unsafe_bitcast[Scalar[DType.float16]](),
            w.device(),
        )
        var out = _fused_matmul_rms_norm_kernel[DType.float16](x16, w16, eps)
        return Tensor[dtype, 2](
            out.shape(),
            out.data().unsafe_bitcast[Scalar[dtype]](),
            out.device(),
        )
    elif dtype == DType.float32:
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
        var out = _fused_matmul_rms_norm_kernel[DType.float32](x32, w32, eps)
        return Tensor[dtype, 2](
            out.shape(),
            out.data().unsafe_bitcast[Scalar[dtype]](),
            out.device(),
        )
    else:
        unimplemented("fused_matmul_rms_norm: unsupported dtype")
        return tensor_zeros[dtype, 2](StaticTuple[Int, 2](0, 0))


def fused_matmul_rms_norm[dtype: DType](
    x: Tensor[dtype, 2],
    w: Tensor[dtype, 2],
    eps: Float32 = Float32(1e-5),
) -> Tensor[dtype, 2]:
    """y = rms_norm(x @ w^T) with w stored [out, in]."""
    return _dispatch[dtype](x, w, eps)

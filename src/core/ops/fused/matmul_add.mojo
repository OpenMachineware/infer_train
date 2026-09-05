# core/ops/fused/matmul_add.mojo
#
# M5 fused kernels: matmul + add in a single pass.
#
#   fused_matmul_add_bias(x, w, bias)
#       y[i, j] = sum_k x[i, k] * w[j, k] + bias[j]
#   fused_matmul_add(x, w, b)
#       y[i, j] = sum_k x[i, k] * w[j, k] + b[i, j]
#
# `w` is weight-major ([out, in], the GGUF layout) so the inner dot product
# walks contiguous rows of both operands.  f16 inputs widen per element and
# accumulate in f32 (the M3 numerics rule); the result casts back to `dtype`.
# The bias/add is folded into the store, saving a full output-tensor pass.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.utils.static_tuple import StaticTuple

comptime W_F16 = 8  # 8 x f16 = 128-bit NEON/SSE vector
comptime W_F32 = 4  # 4 x f32 = 128-bit NEON/SSE vector


def _fused_matmul_add_bias_kernel[
    dtype: DType
](
    x: Tensor[dtype, 2],
    w: Tensor[dtype, 2],
    bias: Tensor[dtype, 1],
) -> Tensor[
    dtype, 2
]:
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]
    if K != w.shape()[1]:
        unimplemented("fused_matmul_add_bias: K mismatch")
    if bias.shape()[0] != N:
        unimplemented("fused_matmul_add_bias: bias length mismatch")
    var out = tensor_zeros[dtype, 2](StaticTuple[Int, 2](M, N))
    comptime W = W_F16 if dtype == DType.float16 else W_F32
    var k_main = (K // W) * W
    for i in range(M):
        for j in range(N):
            var acc = SIMD[DType.float32, W](0)
            var k = 0
            while k < k_main:
                var xv = (
                    x.data()
                    .unsafe_load[width=W](offset=i * K + k)
                    .cast[DType.float32]()
                )
                var wv = (
                    w.data()
                    .unsafe_load[width=W](offset=j * K + k)
                    .cast[DType.float32]()
                )
                acc = acc + xv * wv
                k += W
            var total = Float32(acc.reduce_add())
            while k < K:
                total += Float32(x.get(i * K + k)) * Float32(w.get(j * K + k))
                k += 1
            total += Float32(bias.get(j))
            out.set(i * N + j, Scalar[dtype](total))
    return out


def _fused_matmul_add_kernel[
    dtype: DType
](
    x: Tensor[dtype, 2],
    w: Tensor[dtype, 2],
    b: Tensor[dtype, 2],
) -> Tensor[
    dtype, 2
]:
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]
    if K != w.shape()[1]:
        unimplemented("fused_matmul_add: K mismatch")
    if b.shape()[0] != M or b.shape()[1] != N:
        unimplemented("fused_matmul_add: b shape mismatch")
    var out = tensor_zeros[dtype, 2](StaticTuple[Int, 2](M, N))
    comptime W = W_F16 if dtype == DType.float16 else W_F32
    var k_main = (K // W) * W
    for i in range(M):
        for j in range(N):
            var acc = SIMD[DType.float32, W](0)
            var k = 0
            while k < k_main:
                var xv = (
                    x.data()
                    .unsafe_load[width=W](offset=i * K + k)
                    .cast[DType.float32]()
                )
                var wv = (
                    w.data()
                    .unsafe_load[width=W](offset=j * K + k)
                    .cast[DType.float32]()
                )
                acc = acc + xv * wv
                k += W
            var total = Float32(acc.reduce_add())
            while k < K:
                total += Float32(x.get(i * K + k)) * Float32(w.get(j * K + k))
                k += 1
            total += Float32(b.get(i * N + j))
            out.set(i * N + j, Scalar[dtype](total))
    return out


def _dispatch_matmul_add_bias[
    dtype: DType
](
    x: Tensor[dtype, 2],
    w: Tensor[dtype, 2],
    bias: Tensor[dtype, 1],
) -> Tensor[
    dtype, 2
]:
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
        var b16 = Tensor[DType.float16, 1](
            bias.shape(),
            bias.data().unsafe_bitcast[Scalar[DType.float16]](),
            bias.device(),
        )
        var out = _fused_matmul_add_bias_kernel[DType.float16](x16, w16, b16)
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
        var b32 = Tensor[DType.float32, 1](
            bias.shape(),
            bias.data().unsafe_bitcast[Scalar[DType.float32]](),
            bias.device(),
        )
        var out = _fused_matmul_add_bias_kernel[DType.float32](x32, w32, b32)
        return Tensor[dtype, 2](
            out.shape(),
            out.data().unsafe_bitcast[Scalar[dtype]](),
            out.device(),
        )
    else:
        unimplemented("fused_matmul_add_bias: unsupported dtype")
        return tensor_zeros[dtype, 2](StaticTuple[Int, 2](0, 0))


def _dispatch_matmul_add[
    dtype: DType
](
    x: Tensor[dtype, 2],
    w: Tensor[dtype, 2],
    b: Tensor[dtype, 2],
) -> Tensor[
    dtype, 2
]:
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
        var b16 = Tensor[DType.float16, 2](
            b.shape(),
            b.data().unsafe_bitcast[Scalar[DType.float16]](),
            b.device(),
        )
        var out = _fused_matmul_add_kernel[DType.float16](x16, w16, b16)
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
        var b32 = Tensor[DType.float32, 2](
            b.shape(),
            b.data().unsafe_bitcast[Scalar[DType.float32]](),
            b.device(),
        )
        var out = _fused_matmul_add_kernel[DType.float32](x32, w32, b32)
        return Tensor[dtype, 2](
            out.shape(),
            out.data().unsafe_bitcast[Scalar[dtype]](),
            out.device(),
        )
    else:
        unimplemented("fused_matmul_add: unsupported dtype")
        return tensor_zeros[dtype, 2](StaticTuple[Int, 2](0, 0))


def fused_matmul_add_bias[
    dtype: DType
](
    x: Tensor[dtype, 2],
    w: Tensor[dtype, 2],
    bias: Tensor[dtype, 1],
) -> Tensor[
    dtype, 2
]:
    """Fused linear: y = x @ w^T + bias with w stored [out, in]."""
    return _dispatch_matmul_add_bias[dtype](x, w, bias)


def fused_matmul_add[
    dtype: DType
](
    x: Tensor[dtype, 2],
    w: Tensor[dtype, 2],
    b: Tensor[dtype, 2],
) -> Tensor[
    dtype, 2
]:
    """Fused matmul + elementwise add: y = x @ w^T + b."""
    return _dispatch_matmul_add[dtype](x, w, b)


# M2 placeholder name kept for source compatibility.
def matmul_add_fused[
    dtype: DType
](
    a: Tensor[dtype, 2],
    b: Tensor[dtype, 2],
    bias: Tensor[dtype, 2],
) -> Tensor[
    dtype, 2
]:
    return fused_matmul_add[dtype](a, b, bias)

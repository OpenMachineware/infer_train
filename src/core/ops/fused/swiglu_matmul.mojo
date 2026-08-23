# core/ops/fused/swiglu_matmul.mojo
#
# M5 fused kernel: SwiGLU + matmul in one kernel.
#
#   fused_swiglu_matmul(g, u, w):  y[i, j] = sum_f silu(g[i, f]) * u[i, f] * w[j, f]
#
# This is the Qwen2 FFN down-projection pattern (down(swiglu(gate, up))):
# the SwiGLU activation is never materialized - it is computed on the fly
# inside the dot-product loop.  silu is evaluated in f32 (clamped like the
# standalone swiglu kernel) and the dot products accumulate in f32.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.utils.static_tuple import StaticTuple
from std.math import exp

comptime W_F16 = 8
comptime W_F32 = 4


def _silu_scalar(x: Float32) -> Float32:
    if x < Float32(-20.0):
        return Float32(0.0)
    if x > Float32(20.0):
        return x
    return x / (Float32(1.0) + exp(-x))


def _fused_swiglu_matmul_kernel[dtype: DType](
    gate: Tensor[dtype, 2],
    up: Tensor[dtype, 2],
    w: Tensor[dtype, 2],
) -> Tensor[dtype, 2]:
    var M = gate.shape()[0]
    var F = gate.shape()[1]
    var N = w.shape()[0]
    if up.shape()[0] != M or up.shape()[1] != F:
        unimplemented("fused_swiglu_matmul: gate/up shape mismatch")
    if w.shape()[1] != F:
        unimplemented("fused_swiglu_matmul: w inner dim mismatch")
    var out = tensor_zeros[dtype, 2](StaticTuple[Int, 2](M, N))
    comptime W = W_F16 if dtype == DType.float16 else W_F32
    var f_main = (F // W) * W
    for i in range(M):
        for j in range(N):
            var total = Float32(0)
            var f = 0
            while f < f_main:
                var gv = gate.data().unsafe_load[width=W](
                    offset=i * F + f
                ).cast[DType.float32]()
                var uv = up.data().unsafe_load[width=W](
                    offset=i * F + f
                ).cast[DType.float32]()
                var wv = w.data().unsafe_load[width=W](
                    offset=j * F + f
                ).cast[DType.float32]()
                # silu per lane (scalar exp; lane writes are miscompiled in
                # Mojo 1.0 SIMD, so accumulate the products into `total`)
                for lane in range(W):
                    total += (
                        _silu_scalar(gv[lane]) * uv[lane] * wv[lane]
                    )
                f += W
            while f < F:
                var sg = _silu_scalar(Float32(gate.get(i * F + f)))
                total += sg * Float32(up.get(i * F + f)) * Float32(
                    w.get(j * F + f)
                )
                f += 1
            out.set(i * N + j, Scalar[dtype](total))
    return out


def _dispatch[dtype: DType](
    gate: Tensor[dtype, 2],
    up: Tensor[dtype, 2],
    w: Tensor[dtype, 2],
) -> Tensor[dtype, 2]:
    comptime if dtype == DType.float16:
        var g16 = Tensor[DType.float16, 2](
            gate.shape(),
            gate.data().unsafe_bitcast[Scalar[DType.float16]](),
            gate.device(),
        )
        var u16 = Tensor[DType.float16, 2](
            up.shape(),
            up.data().unsafe_bitcast[Scalar[DType.float16]](),
            up.device(),
        )
        var w16 = Tensor[DType.float16, 2](
            w.shape(),
            w.data().unsafe_bitcast[Scalar[DType.float16]](),
            w.device(),
        )
        var out = _fused_swiglu_matmul_kernel[DType.float16](g16, u16, w16)
        return Tensor[dtype, 2](
            out.shape(),
            out.data().unsafe_bitcast[Scalar[dtype]](),
            out.device(),
        )
    elif dtype == DType.float32:
        var g32 = Tensor[DType.float32, 2](
            gate.shape(),
            gate.data().unsafe_bitcast[Scalar[DType.float32]](),
            gate.device(),
        )
        var u32 = Tensor[DType.float32, 2](
            up.shape(),
            up.data().unsafe_bitcast[Scalar[DType.float32]](),
            up.device(),
        )
        var w32 = Tensor[DType.float32, 2](
            w.shape(),
            w.data().unsafe_bitcast[Scalar[DType.float32]](),
            w.device(),
        )
        var out = _fused_swiglu_matmul_kernel[DType.float32](g32, u32, w32)
        return Tensor[dtype, 2](
            out.shape(),
            out.data().unsafe_bitcast[Scalar[dtype]](),
            out.device(),
        )
    else:
        unimplemented("fused_swiglu_matmul: unsupported dtype")
        return tensor_zeros[dtype, 2](StaticTuple[Int, 2](0, 0))


def fused_swiglu_matmul[dtype: DType](
    gate: Tensor[dtype, 2],
    up: Tensor[dtype, 2],
    w: Tensor[dtype, 2],
) -> Tensor[dtype, 2]:
    """y = silu(gate) * up @ w^T with w stored [out, in]."""
    return _dispatch[dtype](gate, up, w)

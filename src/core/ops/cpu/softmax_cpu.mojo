# core/ops/cpu/softmax_cpu.mojo
#
# CPU stable softmax (P0) along the last axis of a [batch, dim] tensor.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.math import exp


def _softmax_cpu_kernel[
    dtype: DType
](x: Tensor[dtype, 2], dim: Int) -> Tensor[
    dtype, 2
] where dtype.is_floating_point():
    var batch = x.shape()[0]
    var out = tensor_zeros[dtype, 2](x.shape())

    comptime W = 8 if dtype == DType.float16 else 4
    var d_main = (dim // W) * W

    for i in range(batch):
        var base = i * dim

        var mx = Float32(x.get(base))
        var j = 0
        while j < d_main:
            var v = x.data().unsafe_load[width=W](offset=base + j)
            var lane_max = Float32(v.reduce_max())
            if lane_max > mx:
                mx = lane_max
            j += W
        while j < dim:
            var v = Float32(x.get(base + j))
            if v > mx:
                mx = v
            j += 1

        var acc = SIMD[dtype, W](0)
        var total = Float32(0)
        j = 0
        while j < d_main:
            var v = x.data().unsafe_load[width=W](offset=base + j)
            var e = exp(v - SIMD[dtype, W](mx))
            out.data().unsafe_store(base + j, e)
            acc = acc + e
            j += W
        total += Float32(acc.reduce_add())
        while j < dim:
            var v = Float32(x.get(base + j))
            var e = exp(v - mx)
            out.set(base + j, Scalar[dtype](e))
            total += Float32(e)
            j += 1

        var inv = Float32(1) / total
        j = 0
        while j < d_main:
            var v = out.data().unsafe_load[width=W](offset=base + j)
            out.data().unsafe_store(base + j, v * SIMD[dtype, W](inv))
            j += W
        while j < dim:
            var v = Float32(out.get(base + j))
            out.set(base + j, Scalar[dtype](v * inv))
            j += 1
    return out


def softmax_cpu[
    dtype: DType, dim: Int
](x: Tensor[dtype, 2]) -> Tensor[dtype, 2] where dtype.is_floating_point():
    if x.shape()[1] != dim:
        unimplemented("softmax_cpu: static dim mismatch")
    return _softmax_cpu_kernel[dtype](x, dim)


def softmax_cpu_dynamic[
    dtype: DType
](x: Tensor[dtype, 2]) -> Tensor[dtype, 2] where dtype.is_floating_point():
    return _softmax_cpu_kernel[dtype](x, x.shape()[1])


def softmax_cpu_forward_with_saved[
    dtype: DType, dim: Int
](x: Tensor[dtype, 2]) -> Tuple[
    Tensor[dtype, 2], List[Tensor[dtype, 2]]
] where dtype.is_floating_point():
    var out = softmax_cpu[dtype, dim](x)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(out)
    return (out, saved^)


def softmax_cpu_backward[
    dtype: DType, dim: Int
](grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]) -> List[
    Tensor[dtype, 2]
]:
    """Backward for row-wise softmax.

    grad_x[i,j] = p[i,j] * (grad_out[i,j] - sum_k grad_out[i,k] * p[i,k])
    where p = softmax(x) is `saved[0]`.
    """
    var p = saved[0]
    var rows = p.shape()[0]
    var cols = p.shape()[1]
    var grad_x = tensor_zeros[dtype, 2](p.shape())
    for i in range(rows):
        var base = i * cols
        var dot = Float32(0)
        for j in range(cols):
            dot += Float32(grad_out.get(base + j)) * Float32(p.get(base + j))
        for j in range(cols):
            var v = Float32(p.get(base + j)) * (
                Float32(grad_out.get(base + j)) - dot
            )
            grad_x.set(base + j, Scalar[dtype](v))
    var result = List[Tensor[dtype, 2]]()
    result.append(grad_x)
    return result^

# core/ops/cpu/rms_norm_cpu.mojo
#
# CPU RMSNorm (P0).  x is [batch, dim]; the kernel reads `dim` at runtime so
# the registry can dispatch on dynamic shapes, while `rms_norm_cpu` keeps the
# comptime-`dim` signature from the spec.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.math import sqrt


def _rms_norm_cpu_kernel[dtype: DType](
    x: Tensor[dtype, 2], dim: Int, eps: Float32
) -> Tensor[dtype, 2]:
    """out[i, j] = x[i, j] / sqrt(mean(x[i, :]^2) + eps)."""
    var batch = x.shape()[0]
    var out = tensor_zeros[dtype, 2](x.shape())

    comptime W = 8 if dtype == DType.float16 else 4
    var d_main = (dim // W) * W

    for i in range(batch):
        var acc = SIMD[dtype, W](0)
        var base = i * dim
        var j = 0
        while j < d_main:
            var v = x.data().unsafe_load[width=W](offset=base + j)
            acc = acc + v * v
            j += W
        var ss = Float32(acc.reduce_add())
        while j < dim:
            var v = Float32(x.get(base + j))
            ss += v * v
            j += 1

        var rms = sqrt(ss / Float32(dim) + eps)
        var inv = Float32(1) / rms

        j = 0
        while j < d_main:
            var v = x.data().unsafe_load[width=W](offset=base + j)
            out.data().unsafe_store(base + j, v * SIMD[dtype, W](inv))
            j += W
        while j < dim:
            var v = Float32(x.get(base + j))
            out.set(base + j, Scalar[dtype](v * inv))
            j += 1
    return out


def rms_norm_cpu[dtype: DType, dim: Int](
    x: Tensor[dtype, 2], eps: Float32 = Float32(1e-5)
) -> Tensor[dtype, 2]:
    if x.shape()[1] != dim:
        unimplemented("rms_norm_cpu: static dim mismatch")
    return _rms_norm_cpu_kernel[dtype](x, dim, eps)


def rms_norm_cpu_dynamic[dtype: DType](
    x: Tensor[dtype, 2], eps: Float32 = Float32(1e-5)
) -> Tensor[dtype, 2]:
    return _rms_norm_cpu_kernel[dtype](x, x.shape()[1], eps)


def rms_norm_cpu_forward_with_saved[dtype: DType, dim: Int](
    x: Tensor[dtype, 2], eps: Float32 = Float32(1e-5)
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]]:
    var out = rms_norm_cpu[dtype, dim](x, eps)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(x)
    saved.append(out)
    return (out, saved^)


def rms_norm_cpu_backward[dtype: DType, dim: Int](
    grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]
) -> List[Tensor[dtype, 2]]:
    """Backward for unweighted RMSNorm: y = x / sqrt(mean(x^2) + eps).

    With r_i = sqrt(ss_i/N + eps) and s_i = sum_j grad_out[i,j] * y[i,j]:
        grad_x[i,j] = (grad_out[i,j] - y[i,j] * s_i / N) / r_i

    `saved` = [x, y] (the normalized output, captured in forward).
    """
    var x = saved[0]
    var y = saved[1]
    var rows = x.shape()[0]
    var cols = x.shape()[1]
    var eps = Float32(1e-5)
    var grad_x = tensor_zeros[dtype, 2](x.shape())
    for i in range(rows):
        var base = i * cols
        # r_i^2 = mean(x^2) + eps
        var ss = Float32(0)
        for j in range(cols):
            var xv = Float32(x.get(base + j))
            ss += xv * xv
        var r2 = ss / Float32(cols) + eps
        var r = sqrt(r2)
        # s_i = sum_j grad_out * y
        var s = Float32(0)
        for j in range(cols):
            s += Float32(grad_out.get(base + j)) * Float32(y.get(base + j))
        var k = s / Float32(cols)
        for j in range(cols):
            var v = (
                Float32(grad_out.get(base + j))
                - Float32(y.get(base + j)) * k
            ) / r
            grad_x.set(base + j, Scalar[dtype](v))
    var result = List[Tensor[dtype, 2]]()
    result.append(grad_x)
    return result^


def rms_norm_weight_cpu[dtype: DType](
    x: Tensor[dtype, 2],
    weight: Tensor[dtype, 1],
    eps: Float32 = Float32(1e-5),
) -> Tensor[dtype, 2]:
    """Weighted RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight.

    (The norm the Qwen2 transformer layers actually use; `rms_norm_cpu` is
    the unweighted M1 kernel.)
    """
    var rows = x.shape()[0]
    var cols = x.shape()[1]
    if weight.shape()[0] != cols:
        unimplemented("rms_norm_weight_cpu: weight length mismatch")
    var out = tensor_zeros[dtype, 2](x.shape())
    for i in range(rows):
        var base = i * cols
        var ss = Float32(0)
        for j in range(cols):
            var xv = Float32(x.get(base + j))
            ss += xv * xv
        var r = sqrt(ss / Float32(cols) + eps)
        for j in range(cols):
            out.set(
                base + j,
                Scalar[dtype](
                    Float32(x.get(base + j)) / r * Float32(weight.get(j))
                ),
            )
    return out


def rms_norm_weight_cpu_forward_with_saved[dtype: DType](
    x: Tensor[dtype, 2],
    weight: Tensor[dtype, 1],
    eps: Float32 = Float32(1e-5),
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]]:
    var out = rms_norm_weight_cpu[dtype](x, weight, eps)
    # save x, weight, and z = x / r (the unweighted normalized tensor)
    var z = tensor_zeros[dtype, 2](x.shape())
    var rows = x.shape()[0]
    var cols = x.shape()[1]
    for i in range(rows):
        var base = i * cols
        var ss = Float32(0)
        for j in range(cols):
            var xv = Float32(x.get(base + j))
            ss += xv * xv
        var r = sqrt(ss / Float32(cols) + eps)
        for j in range(cols):
            z.set(base + j, Scalar[dtype](Float32(x.get(base + j)) / r))
    var saved = List[Tensor[dtype, 2]]()
    saved.append(x)
    saved.append(z)
    return (out, saved^)


def rms_norm_weight_cpu_backward[dtype: DType](
    grad_out: Tensor[dtype, 2],
    saved: List[Tensor[dtype, 2]],
    weight: Tensor[dtype, 1],
) -> Tuple[Tensor[dtype, 2], Tensor[dtype, 1]]:
    """Backward for weighted RMSNorm: out = z * w with z = x / r.

    grad_w[j] = sum_i grad_out[i,j] * z[i,j]; grad_x applies the unweighted
    backward to grad_z = grad_out * w.  `saved` = [x, z].
    """
    var x = saved[0]
    var z = saved[1]
    var rows = x.shape()[0]
    var cols = x.shape()[1]
    var eps = Float32(1e-5)
    var grad_x = tensor_zeros[dtype, 2](x.shape())
    var grad_w = tensor_zeros[dtype, 1](weight.shape())
    for i in range(rows):
        var base = i * cols
        # r_i^2 = mean(x^2) + eps (recompute from x)
        var ss = Float32(0)
        for j in range(cols):
            var xv = Float32(x.get(base + j))
            ss += xv * xv
        var r2 = ss / Float32(cols) + eps
        var r = sqrt(r2)
        # s_i = sum_j (grad_out * w) * z; grad_w accumulates grad_out * z
        var s = Float32(0)
        for j in range(cols):
            var go = Float32(grad_out.get(base + j))
            var gz = go * Float32(weight.get(j))
            grad_w.set(
                j,
                Scalar[dtype](
                    Float32(grad_w.get(j))
                    + go * Float32(z.get(base + j))
                ),
            )
            s += gz * Float32(z.get(base + j))
        var k = s / Float32(cols)
        for j in range(cols):
            var gz = Float32(grad_out.get(base + j)) * Float32(
                weight.get(j)
            )
            var v = (gz - Float32(z.get(base + j)) * k) / r
            grad_x.set(base + j, Scalar[dtype](v))
    return (grad_x, grad_w)

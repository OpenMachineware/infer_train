# core/ops/gpu/rms_norm_gpu.mojo
#
# GPU RMSNorm: out[i, j] = x[i, j] / sqrt(mean(x[i, :]^2) + eps).
#
# One thread block per row: threads strided-reduce the sum of squares with
# the `max.gpu.primitives.block` collectives, then scale the row.  All math
# is f32; f16 rows are widened element-wise and cast back on store.

from ...tensor import Tensor
from ...utils import unimplemented
from ..cpu.rms_norm_cpu import (
    rms_norm_cpu,
    rms_norm_cpu_dynamic,
    rms_norm_cpu_backward,
)
from .gpu_runtime import (
    download2,
    get_gpu_context,
    gpu_available,
    upload,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.block import sum as block_sum
from std.gpu import block_idx, thread_idx
from std.memory import Pointer
from std.origin import MutAnyOrigin
from std.math import sqrt
from std.utils.static_tuple import StaticTuple

comptime BLOCK = 256


# -- kernels ------------------------------------------------------------------


def _rms_norm_kernel_f32(
    x: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    dim: Int32,
    eps: Float32,
):
    var n = Int(dim)
    var row = block_idx.x
    var base = row * n
    var tid = thread_idx.x

    var ss = Float32(0.0)
    var i = tid
    while i < n:
        var v = x[base + i]
        ss += v * v
        i += BLOCK
    var ss_all = block_sum[block_size=BLOCK](ss)
    var rms = sqrt(ss_all / Float32(n) + eps)
    var inv = Float32(1.0) / rms

    i = tid
    while i < n:
        dst[base + i] = x[base + i] * inv
        i += BLOCK


def _rms_norm_kernel_f16(
    x: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dim: Int32,
    eps: Float32,
):
    var n = Int(dim)
    var row = block_idx.x
    var base = row * n
    var tid = thread_idx.x

    var ss = Float32(0.0)
    var i = tid
    while i < n:
        var v = Float32(x[base + i])
        ss += v * v
        i += BLOCK
    var ss_all = block_sum[block_size=BLOCK](ss)
    var rms = sqrt(ss_all / Float32(n) + eps)
    var inv = Float32(1.0) / rms

    i = tid
    while i < n:
        dst[base + i] = Scalar[DType.float16](Float32(x[base + i]) * inv)
        i += BLOCK


def _rms_norm_bwd_kernel_f32(
    grad_out: Pointer[Float32, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    dim: Int32,
    eps: Float32,
):
    """grad_x[i,j] = (grad_out[i,j] - y[i,j] * s_i / N) / r_i with
    s_i = sum_j grad_out[i,j] * y[i,j], r_i = sqrt(mean(x^2) + eps)."""
    var n = Int(dim)
    var row = block_idx.x
    var base = row * n
    var tid = thread_idx.x

    var ss = Float32(0.0)
    var s = Float32(0.0)
    var i = tid
    while i < n:
        var xv = x[base + i]
        ss += xv * xv
        s += grad_out[base + i] * y[base + i]
        i += BLOCK
    var ss_all = block_sum[block_size=BLOCK](ss)
    var s_all = block_sum[block_size=BLOCK](s)
    var r = sqrt(ss_all / Float32(n) + eps)
    var k = s_all / Float32(n)

    i = tid
    while i < n:
        dst[base + i] = (grad_out[base + i] - y[base + i] * k) / r
        i += BLOCK


def _rms_norm_bwd_kernel_f16(
    grad_out: Pointer[Scalar[DType.float16], MutAnyOrigin],
    x: Pointer[Scalar[DType.float16], MutAnyOrigin],
    y: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dim: Int32,
    eps: Float32,
):
    var n = Int(dim)
    var row = block_idx.x
    var base = row * n
    var tid = thread_idx.x

    var ss = Float32(0.0)
    var s = Float32(0.0)
    var i = tid
    while i < n:
        var xv = Float32(x[base + i])
        ss += xv * xv
        s += Float32(grad_out[base + i]) * Float32(y[base + i])
        i += BLOCK
    var ss_all = block_sum[block_size=BLOCK](ss)
    var s_all = block_sum[block_size=BLOCK](s)
    var r = sqrt(ss_all / Float32(n) + eps)
    var k = s_all / Float32(n)

    i = tid
    while i < n:
        dst[base + i] = Scalar[DType.float16](
            (Float32(grad_out[base + i]) - Float32(y[base + i]) * k) / r
        )
        i += BLOCK


# -- launch helpers -----------------------------------------------------------


def _rms_norm_gpu_launch[dtype: DType](
    ctx: DeviceContext, x: Tensor[dtype, 2], eps: Float32
) raises -> Tensor[dtype, 2]:
    var dim = x.shape()[1]
    var x_buf = upload[dtype, 2](ctx, x)
    var dst_buf = ctx.enqueue_create_buffer[dtype](x.numel())
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_rms_norm_kernel_f16](
            x_buf, dst_buf, Int32(dim), eps,
            grid_dim=x.shape()[0], block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_rms_norm_kernel_f32](
            x_buf, dst_buf, Int32(dim), eps,
            grid_dim=x.shape()[0], block_dim=BLOCK,
        )
    var out = download2[dtype](ctx, dst_buf, x.shape())
    ctx.synchronize()
    return out


def _rms_norm_bwd_gpu_launch[dtype: DType](
    ctx: DeviceContext,
    grad_out: Tensor[dtype, 2],
    x: Tensor[dtype, 2],
    y: Tensor[dtype, 2],
    eps: Float32,
) raises -> Tensor[dtype, 2]:
    var dim = x.shape()[1]
    var go_buf = upload[dtype, 2](ctx, grad_out)
    var x_buf = upload[dtype, 2](ctx, x)
    var y_buf = upload[dtype, 2](ctx, y)
    var dst_buf = ctx.enqueue_create_buffer[dtype](x.numel())
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_rms_norm_bwd_kernel_f16](
            go_buf, x_buf, y_buf, dst_buf, Int32(dim), eps,
            grid_dim=x.shape()[0], block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_rms_norm_bwd_kernel_f32](
            go_buf, x_buf, y_buf, dst_buf, Int32(dim), eps,
            grid_dim=x.shape()[0], block_dim=BLOCK,
        )
    var out = download2[dtype](ctx, dst_buf, x.shape())
    ctx.synchronize()
    return out


# -- public entry points ------------------------------------------------------


def rms_norm_gpu[dtype: DType, dim: Int](
    x: Tensor[dtype, 2], eps: Float32 = Float32(1e-5)
) -> Tensor[dtype, 2]:
    """Comptime-dim GPU RMSNorm (CPU fallback on any GPU error)."""
    if x.shape()[1] != dim:
        unimplemented("rms_norm_gpu: static dim mismatch")
    if not gpu_available[dtype]():
        return rms_norm_cpu[dtype, dim](x, eps)
    try:
        var ctx = get_gpu_context()
        return _rms_norm_gpu_launch[dtype](ctx, x, eps)
    except:
        return rms_norm_cpu[dtype, dim](x, eps)


def rms_norm_gpu_dynamic[dtype: DType](
    x: Tensor[dtype, 2], eps: Float32 = Float32(1e-5)
) -> Tensor[dtype, 2]:
    """Runtime-dim GPU RMSNorm (CPU fallback on any GPU error)."""
    if not gpu_available[dtype]():
        return rms_norm_cpu_dynamic[dtype](x, eps)
    try:
        var ctx = get_gpu_context()
        return _rms_norm_gpu_launch[dtype](ctx, x, eps)
    except:
        return rms_norm_cpu_dynamic[dtype](x, eps)


def rms_norm_gpu_forward_with_saved[dtype: DType, dim: Int](
    x: Tensor[dtype, 2], eps: Float32 = Float32(1e-5)
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]]:
    var out = rms_norm_gpu[dtype, dim](x, eps)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(x)
    saved.append(out)
    return (out, saved^)


def rms_norm_gpu_backward[dtype: DType, dim: Int](
    grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]
) -> List[Tensor[dtype, 2]]:
    """Backward for unweighted RMSNorm.

    With r_i = sqrt(ss_i/N + eps) and s_i = sum_j grad_out[i,j] * y[i,j]:
        grad_x[i,j] = (grad_out[i,j] - y[i,j] * s_i / N) / r_i
    `saved` = [x, y] (y is the normalized output captured in forward).
    """
    var x = saved[0]
    var y = saved[1]
    var eps = Float32(1e-5)
    if not gpu_available[dtype]():
        return rms_norm_cpu_backward[dtype, dim](grad_out, saved)
    try:
        var ctx = get_gpu_context()
        var grad_x = _rms_norm_bwd_gpu_launch[dtype](ctx, grad_out, x, y, eps)
        var result = List[Tensor[dtype, 2]]()
        result.append(grad_x)
        return result^
    except:
        return rms_norm_cpu_backward[dtype, dim](grad_out, saved)

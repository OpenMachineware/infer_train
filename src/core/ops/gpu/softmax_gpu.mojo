# core/ops/gpu/softmax_gpu.mojo
#
# GPU stable softmax along the last axis of a [batch, dim] tensor.
#
# One thread block per row: threads strided-reduce the row max and the sum of
# exps with the `max.gpu.primitives.block` collectives (which manage the
# shared memory and barriers internally), then write the normalized row.
# All math is f32; f16 rows are widened element-wise and cast back on store.

from ...tensor import Tensor
from ...utils import unimplemented
from ..cpu.softmax_cpu import (
    softmax_cpu,
    softmax_cpu_dynamic,
    softmax_cpu_backward,
)
from .gpu_runtime import (
    download2,
    get_gpu_context,
    gpu_available,
    upload,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.block import max as block_max, sum as block_sum
from std.gpu import block_idx, thread_idx
from std.memory import Pointer
from std.origin import MutAnyOrigin
from std.math import exp
from std.utils.static_tuple import StaticTuple

comptime BLOCK = 256


# -- kernels ------------------------------------------------------------------


def _softmax_kernel_f32(
    x: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    dim: Int32,
):
    var n = Int(dim)
    var row = block_idx.x
    var base = row * n
    var tid = thread_idx.x

    var mx = Float32(-3.402823466e38)
    var i = tid
    while i < n:
        var v = x[base + i]
        if v > mx:
            mx = v
        i += BLOCK
    var mx_all = block_max[block_size=BLOCK](mx)

    var total = Float32(0.0)
    i = tid
    while i < n:
        total += exp(x[base + i] - mx_all)
        i += BLOCK
    var total_all = block_sum[block_size=BLOCK](total)
    var inv = Float32(1.0) / total_all

    i = tid
    while i < n:
        dst[base + i] = exp(x[base + i] - mx_all) * inv
        i += BLOCK


def _softmax_kernel_f16(
    x: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dim: Int32,
):
    var n = Int(dim)
    var row = block_idx.x
    var base = row * n
    var tid = thread_idx.x

    var mx = Float32(-3.402823466e38)
    var i = tid
    while i < n:
        var v = Float32(x[base + i])
        if v > mx:
            mx = v
        i += BLOCK
    var mx_all = block_max[block_size=BLOCK](mx)

    var total = Float32(0.0)
    i = tid
    while i < n:
        total += exp(Float32(x[base + i]) - mx_all)
        i += BLOCK
    var total_all = block_sum[block_size=BLOCK](total)
    var inv = Float32(1.0) / total_all

    i = tid
    while i < n:
        dst[base + i] = Scalar[DType.float16](
            exp(Float32(x[base + i]) - mx_all) * inv
        )
        i += BLOCK


def _softmax_bwd_kernel_f32(
    grad_out: Pointer[Float32, MutAnyOrigin],
    p: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    dim: Int32,
):
    var n = Int(dim)
    var row = block_idx.x
    var base = row * n
    var tid = thread_idx.x

    var dot = Float32(0.0)
    var i = tid
    while i < n:
        dot += grad_out[base + i] * p[base + i]
        i += BLOCK
    var dot_all = block_sum[block_size=BLOCK](dot)

    i = tid
    while i < n:
        dst[base + i] = p[base + i] * (grad_out[base + i] - dot_all)
        i += BLOCK


def _softmax_bwd_kernel_f16(
    grad_out: Pointer[Scalar[DType.float16], MutAnyOrigin],
    p: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dim: Int32,
):
    var n = Int(dim)
    var row = block_idx.x
    var base = row * n
    var tid = thread_idx.x

    var dot = Float32(0.0)
    var i = tid
    while i < n:
        dot += Float32(grad_out[base + i]) * Float32(p[base + i])
        i += BLOCK
    var dot_all = block_sum[block_size=BLOCK](dot)

    i = tid
    while i < n:
        dst[base + i] = Scalar[DType.float16](
            Float32(p[base + i]) * (Float32(grad_out[base + i]) - dot_all)
        )
        i += BLOCK


# -- launch helpers -----------------------------------------------------------


def _softmax_gpu_launch[dtype: DType](
    ctx: DeviceContext, x: Tensor[dtype, 2]
) raises -> Tensor[dtype, 2]:
    var batch = x.shape()[0]
    var dim = x.shape()[1]
    var x_buf = upload[dtype, 2](ctx, x)
    var dst_buf = ctx.enqueue_create_buffer[dtype](x.numel())
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_softmax_kernel_f16](
            x_buf, dst_buf, Int32(dim),
            grid_dim=batch, block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_softmax_kernel_f32](
            x_buf, dst_buf, Int32(dim),
            grid_dim=batch, block_dim=BLOCK,
        )
    var out = download2[dtype](ctx, dst_buf, x.shape())
    ctx.synchronize()
    return out


def _softmax_bwd_gpu_launch[dtype: DType](
    ctx: DeviceContext, grad_out: Tensor[dtype, 2], p: Tensor[dtype, 2]
) raises -> Tensor[dtype, 2]:
    var dim = p.shape()[1]
    var go_buf = upload[dtype, 2](ctx, grad_out)
    var p_buf = upload[dtype, 2](ctx, p)
    var dst_buf = ctx.enqueue_create_buffer[dtype](p.numel())
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_softmax_bwd_kernel_f16](
            go_buf, p_buf, dst_buf, Int32(dim),
            grid_dim=p.shape()[0], block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_softmax_bwd_kernel_f32](
            go_buf, p_buf, dst_buf, Int32(dim),
            grid_dim=p.shape()[0], block_dim=BLOCK,
        )
    var out = download2[dtype](ctx, dst_buf, p.shape())
    ctx.synchronize()
    return out


# -- public entry points ------------------------------------------------------


def softmax_gpu[dtype: DType, dim: Int](
    x: Tensor[dtype, 2]
) -> Tensor[dtype, 2] where dtype.is_floating_point():
    """Comptime-dim GPU stable softmax (CPU fallback on any GPU error)."""
    if x.shape()[1] != dim:
        unimplemented("softmax_gpu: static dim mismatch")
    if not gpu_available[dtype]():
        return softmax_cpu[dtype, dim](x)
    try:
        var ctx = get_gpu_context()
        return _softmax_gpu_launch[dtype](ctx, x)
    except:
        return softmax_cpu[dtype, dim](x)


def softmax_gpu_dynamic[dtype: DType](
    x: Tensor[dtype, 2]
) -> Tensor[dtype, 2] where dtype.is_floating_point():
    """Runtime-dim GPU stable softmax (CPU fallback on any GPU error)."""
    if not gpu_available[dtype]():
        return softmax_cpu_dynamic[dtype](x)
    try:
        var ctx = get_gpu_context()
        return _softmax_gpu_launch[dtype](ctx, x)
    except:
        return softmax_cpu_dynamic[dtype](x)


def softmax_gpu_forward_with_saved[dtype: DType, dim: Int](
    x: Tensor[dtype, 2]
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]] where dtype.is_floating_point():
    var out = softmax_gpu[dtype, dim](x)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(out)
    return (out, saved^)


def softmax_gpu_backward[dtype: DType, dim: Int](
    grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]
) -> List[Tensor[dtype, 2]] where dtype.is_floating_point():
    """Backward for row-wise softmax.

    grad_x[i,j] = p[i,j] * (grad_out[i,j] - sum_k grad_out[i,k] * p[i,k])
    where p = softmax(x) is `saved[0]`.
    """
    var p = saved[0]
    if not gpu_available[dtype]():
        return softmax_cpu_backward[dtype, dim](grad_out, saved)
    try:
        var ctx = get_gpu_context()
        var grad_x = _softmax_bwd_gpu_launch[dtype](ctx, grad_out, p)
        var result = List[Tensor[dtype, 2]]()
        result.append(grad_x)
        return result^
    except:
        return softmax_cpu_backward[dtype, dim](grad_out, saved)

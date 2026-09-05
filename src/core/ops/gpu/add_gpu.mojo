# core/ops/gpu/add_gpu.mojo
#
# GPU element-wise add (residual connections) plus the rank-1 bias broadcast
# used by the QKV projections.  Metal kernels via `max.gpu.host.DeviceContext`;
# the host entry points upload, launch, and download (see gpu_runtime.mojo).

from ...device import has_metal_gpu
from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from ..cpu.add_cpu import (
    add_cpu,
    add_cpu_dynamic,
    add_cpu_backward,
    add_row_cpu,
    add_row_cpu_backward,
)
from .gpu_runtime import (
    download2,
    get_gpu_context,
    grid1d,
    gpu_available,
    upload,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import global_idx, grid_dim, block_dim
from std.memory import Pointer
from std.origin import MutAnyOrigin
from std.utils.static_tuple import StaticTuple

comptime BLOCK = 256


# -- kernels ------------------------------------------------------------------


def _add_kernel_f32(
    a: Pointer[Float32, MutAnyOrigin],
    b: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    n: Int32,
):
    var n_i = Int(n)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n_i:
        dst[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]
        i += stride


def _add_kernel_f16(
    a: Pointer[Scalar[DType.float16], MutAnyOrigin],
    b: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    n: Int32,
):
    var n_i = Int(n)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n_i:
        dst[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]
        i += stride


def _add_row_kernel_f32(
    x: Pointer[Float32, MutAnyOrigin],
    bias: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    cols: Int32,
    n: Int32,
):
    var cols_i = Int(cols)
    var n_i = Int(n)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n_i:
        var j = i % cols_i
        dst[unsafe_offset=i] = x[unsafe_offset=i] + bias[unsafe_offset=j]
        i += stride


def _add_row_kernel_f16(
    x: Pointer[Scalar[DType.float16], MutAnyOrigin],
    bias: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    cols: Int32,
    n: Int32,
):
    var cols_i = Int(cols)
    var n_i = Int(n)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n_i:
        var j = i % cols_i
        dst[unsafe_offset=i] = x[unsafe_offset=i] + bias[unsafe_offset=j]
        i += stride


def _copy_kernel_f32(
    src: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    n: Int32,
):
    var n_i = Int(n)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n_i:
        dst[unsafe_offset=i] = src[unsafe_offset=i]
        i += stride


def _copy_kernel_f16(
    src: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    n: Int32,
):
    var n_i = Int(n)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n_i:
        dst[unsafe_offset=i] = src[unsafe_offset=i]
        i += stride


# -- launch helpers -----------------------------------------------------------


def _add_gpu_launch[
    dtype: DType
](
    ctx: DeviceContext, a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) raises -> Tensor[dtype, 2]:
    if a.shape() != b.shape():
        unimplemented("add_gpu: shape mismatch")
    var n = a.numel()
    var a_buf = upload[dtype, 2](ctx, a)
    var b_buf = upload[dtype, 2](ctx, b)
    var dst_buf = ctx.enqueue_create_buffer[dtype](n)
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_add_kernel_f16](
            a_buf,
            b_buf,
            dst_buf,
            Int32(n),
            grid_dim=grid1d(n, BLOCK),
            block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_add_kernel_f32](
            a_buf,
            b_buf,
            dst_buf,
            Int32(n),
            grid_dim=grid1d(n, BLOCK),
            block_dim=BLOCK,
        )
    var out = download2[dtype](ctx, dst_buf, a.shape())
    ctx.synchronize()
    return out


def _add_row_gpu_launch[
    dtype: DType
](
    ctx: DeviceContext, x: Tensor[dtype, 2], bias: Tensor[dtype, 1]
) raises -> Tensor[dtype, 2]:
    var cols = x.shape()[1]
    if bias.shape()[0] != cols:
        unimplemented("add_row_gpu: bias length mismatch")
    var n = x.numel()
    var x_buf = upload[dtype, 2](ctx, x)
    var bias_buf = upload[dtype, 1](ctx, bias)
    var dst_buf = ctx.enqueue_create_buffer[dtype](n)
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_add_row_kernel_f16](
            x_buf,
            bias_buf,
            dst_buf,
            Int32(cols),
            Int32(n),
            grid_dim=grid1d(n, BLOCK),
            block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_add_row_kernel_f32](
            x_buf,
            bias_buf,
            dst_buf,
            Int32(cols),
            Int32(n),
            grid_dim=grid1d(n, BLOCK),
            block_dim=BLOCK,
        )
    var out = download2[dtype](ctx, dst_buf, x.shape())
    ctx.synchronize()
    return out


def _copy_gpu_launch[
    dtype: DType
](ctx: DeviceContext, src: Tensor[dtype, 2]) raises -> Tensor[dtype, 2]:
    var n = src.numel()
    var src_buf = upload[dtype, 2](ctx, src)
    var dst_buf = ctx.enqueue_create_buffer[dtype](n)
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_copy_kernel_f16](
            src_buf,
            dst_buf,
            Int32(n),
            grid_dim=grid1d(n, BLOCK),
            block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_copy_kernel_f32](
            src_buf,
            dst_buf,
            Int32(n),
            grid_dim=grid1d(n, BLOCK),
            block_dim=BLOCK,
        )
    var out = download2[dtype](ctx, dst_buf, src.shape())
    ctx.synchronize()
    return out


# -- public entry points ------------------------------------------------------


def add_gpu[
    dtype: DType, rows: Int, cols: Int
](a: Tensor[dtype, 2], b: Tensor[dtype, 2]) -> Tensor[dtype, 2]:
    """Comptime-shaped GPU element-wise add (CPU fallback on any GPU error)."""
    if a.shape() != StaticTuple[Int, 2](rows, cols):
        unimplemented("add_gpu: static shape mismatch")
    if not gpu_available[dtype]():
        return add_cpu[dtype, rows, cols](a, b)
    try:
        var ctx = get_gpu_context()
        return _add_gpu_launch[dtype](ctx, a, b)
    except:
        return add_cpu[dtype, rows, cols](a, b)


def add_gpu_dynamic[
    dtype: DType
](a: Tensor[dtype, 2], b: Tensor[dtype, 2]) -> Tensor[dtype, 2]:
    """Runtime-shaped GPU element-wise add (CPU fallback on any GPU error)."""
    if not gpu_available[dtype]():
        return add_cpu_dynamic[dtype](a, b)
    try:
        var ctx = get_gpu_context()
        return _add_gpu_launch[dtype](ctx, a, b)
    except:
        return add_cpu_dynamic[dtype](a, b)


def add_row_gpu[
    dtype: DType
](x: Tensor[dtype, 2], bias: Tensor[dtype, 1]) -> Tensor[dtype, 2]:
    """GPU row-broadcast bias add: out[i, j] = x[i, j] + bias[j]."""
    if not gpu_available[dtype]():
        return add_row_cpu[dtype](x, bias)
    try:
        var ctx = get_gpu_context()
        return _add_row_gpu_launch[dtype](ctx, x, bias)
    except:
        return add_row_cpu[dtype](x, bias)


def add_gpu_forward_with_saved[
    dtype: DType, rows: Int, cols: Int
](a: Tensor[dtype, 2], b: Tensor[dtype, 2]) -> Tuple[
    Tensor[dtype, 2], List[Tensor[dtype, 2]]
]:
    var out = add_gpu[dtype, rows, cols](a, b)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(a)
    saved.append(b)
    return (out, saved^)


def add_gpu_backward[
    dtype: DType, rows: Int, cols: Int
](grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]) -> List[
    Tensor[dtype, 2]
]:
    """Backward for element-wise add: both inputs get a copy of grad_out."""
    _ = saved
    if not gpu_available[dtype]():
        return add_cpu_backward[dtype, rows, cols](grad_out, saved)
    try:
        var ctx = get_gpu_context()
        var grad_a = _copy_gpu_launch[dtype](ctx, grad_out)
        var grad_b = _copy_gpu_launch[dtype](ctx, grad_out)
        var result = List[Tensor[dtype, 2]]()
        result.append(grad_a)
        result.append(grad_b)
        return result^
    except:
        return add_cpu_backward[dtype, rows, cols](grad_out, saved)

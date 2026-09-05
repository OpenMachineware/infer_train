# core/ops/gpu/swiglu_gpu.mojo
#
# GPU SwiGLU activation for the Qwen2 FFN: out = silu(gate) * up, where
# silu(x) = x * sigmoid(x).  Computed in f32 and cast back to `dtype`
# (f16 inputs are widened element-wise, matching the CPU kernel).

from ...tensor import Tensor
from ...utils import unimplemented
from ..cpu.swiglu_cpu import (
    swiglu_cpu,
    swiglu_cpu_dynamic,
    swiglu_cpu_backward,
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
from std.math import exp
from std.utils.static_tuple import StaticTuple

comptime BLOCK = 256


# -- kernels ------------------------------------------------------------------


def _swiglu_kernel_f32(
    gate: Pointer[Float32, MutAnyOrigin],
    up: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    n: Int32,
):
    var n_i = Int(n)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n_i:
        var g = gate[unsafe_offset=i]
        var u = up[unsafe_offset=i]
        # silu with overflow guards (mirrors the CPU kernel)
        var silu_g: Float32
        if g < Float32(-20.0):
            silu_g = Float32(0.0)
        elif g > Float32(20.0):
            silu_g = g
        else:
            silu_g = g / (Float32(1.0) + exp(-g))
        dst[unsafe_offset=i] = silu_g * u
        i += stride


def _swiglu_kernel_f16(
    gate: Pointer[Scalar[DType.float16], MutAnyOrigin],
    up: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    n: Int32,
):
    var n_i = Int(n)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n_i:
        var g = Float32(gate[unsafe_offset=i])
        var u = Float32(up[unsafe_offset=i])
        var silu_g: Float32
        if g < Float32(-20.0):
            silu_g = Float32(0.0)
        elif g > Float32(20.0):
            silu_g = g
        else:
            silu_g = g / (Float32(1.0) + exp(-g))
        dst[unsafe_offset=i] = Scalar[DType.float16](silu_g * u)
        i += stride


def _swiglu_bwd_kernel_f32(
    grad_out: Pointer[Float32, MutAnyOrigin],
    gate: Pointer[Float32, MutAnyOrigin],
    up: Pointer[Float32, MutAnyOrigin],
    grad_gate: Pointer[Float32, MutAnyOrigin],
    grad_up: Pointer[Float32, MutAnyOrigin],
    n: Int32,
):
    var n_i = Int(n)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n_i:
        var g = gate[unsafe_offset=i]
        var u = up[unsafe_offset=i]
        var go = grad_out[unsafe_offset=i]
        var silu_g: Float32
        if g < Float32(-20.0):
            silu_g = Float32(0.0)
        elif g > Float32(20.0):
            silu_g = g
        else:
            silu_g = g / (Float32(1.0) + exp(-g))
        # silu'(g) = sigmoid(g) * (1 + g - silu(g))
        var sig: Float32
        if g < Float32(-20.0):
            sig = Float32(0.0)
        elif g > Float32(20.0):
            sig = Float32(1.0)
        else:
            sig = Float32(1.0) / (Float32(1.0) + exp(-g))
        var dsilu = sig * (Float32(1.0) + g - silu_g)
        grad_up[unsafe_offset=i] = go * silu_g
        grad_gate[unsafe_offset=i] = go * u * dsilu
        i += stride


def _swiglu_bwd_kernel_f16(
    grad_out: Pointer[Scalar[DType.float16], MutAnyOrigin],
    gate: Pointer[Scalar[DType.float16], MutAnyOrigin],
    up: Pointer[Scalar[DType.float16], MutAnyOrigin],
    grad_gate: Pointer[Scalar[DType.float16], MutAnyOrigin],
    grad_up: Pointer[Scalar[DType.float16], MutAnyOrigin],
    n: Int32,
):
    var n_i = Int(n)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n_i:
        var g = Float32(gate[unsafe_offset=i])
        var u = Float32(up[unsafe_offset=i])
        var go = Float32(grad_out[unsafe_offset=i])
        var silu_g: Float32
        if g < Float32(-20.0):
            silu_g = Float32(0.0)
        elif g > Float32(20.0):
            silu_g = g
        else:
            silu_g = g / (Float32(1.0) + exp(-g))
        var sig: Float32
        if g < Float32(-20.0):
            sig = Float32(0.0)
        elif g > Float32(20.0):
            sig = Float32(1.0)
        else:
            sig = Float32(1.0) / (Float32(1.0) + exp(-g))
        var dsilu = sig * (Float32(1.0) + g - silu_g)
        grad_up[unsafe_offset=i] = Scalar[DType.float16](go * silu_g)
        grad_gate[unsafe_offset=i] = Scalar[DType.float16](go * u * dsilu)
        i += stride


# -- launch helpers -----------------------------------------------------------


def _swiglu_gpu_launch[
    dtype: DType
](
    ctx: DeviceContext, gate: Tensor[dtype, 2], up: Tensor[dtype, 2]
) raises -> Tensor[dtype, 2]:
    if gate.shape() != up.shape():
        unimplemented("swiglu_gpu: shape mismatch")
    var n = gate.numel()
    var gate_buf = upload[dtype, 2](ctx, gate)
    var up_buf = upload[dtype, 2](ctx, up)
    var dst_buf = ctx.enqueue_create_buffer[dtype](n)
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_swiglu_kernel_f16](
            gate_buf,
            up_buf,
            dst_buf,
            Int32(n),
            grid_dim=grid1d(n, BLOCK),
            block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_swiglu_kernel_f32](
            gate_buf,
            up_buf,
            dst_buf,
            Int32(n),
            grid_dim=grid1d(n, BLOCK),
            block_dim=BLOCK,
        )
    var out = download2[dtype](ctx, dst_buf, gate.shape())
    ctx.synchronize()
    return out


def _swiglu_bwd_gpu_launch[
    dtype: DType
](
    ctx: DeviceContext,
    grad_out: Tensor[dtype, 2],
    gate: Tensor[dtype, 2],
    up: Tensor[dtype, 2],
) raises -> Tuple[Tensor[dtype, 2], Tensor[dtype, 2]]:
    var n = grad_out.numel()
    var go_buf = upload[dtype, 2](ctx, grad_out)
    var gate_buf = upload[dtype, 2](ctx, gate)
    var up_buf = upload[dtype, 2](ctx, up)
    var gg_buf = ctx.enqueue_create_buffer[dtype](n)
    var gu_buf = ctx.enqueue_create_buffer[dtype](n)
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_swiglu_bwd_kernel_f16](
            go_buf,
            gate_buf,
            up_buf,
            gg_buf,
            gu_buf,
            Int32(n),
            grid_dim=grid1d(n, BLOCK),
            block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_swiglu_bwd_kernel_f32](
            go_buf,
            gate_buf,
            up_buf,
            gg_buf,
            gu_buf,
            Int32(n),
            grid_dim=grid1d(n, BLOCK),
            block_dim=BLOCK,
        )
    var grad_gate = download2[dtype](ctx, gg_buf, grad_out.shape())
    var grad_up = download2[dtype](ctx, gu_buf, grad_out.shape())
    ctx.synchronize()
    return (grad_gate, grad_up)


# -- public entry points ------------------------------------------------------


def swiglu_gpu[
    dtype: DType, rows: Int, cols: Int
](gate: Tensor[dtype, 2], up: Tensor[dtype, 2]) -> Tensor[dtype, 2]:
    """Comptime-shaped GPU SwiGLU (CPU fallback on any GPU error)."""
    if gate.shape() != StaticTuple[Int, 2](rows, cols):
        unimplemented("swiglu_gpu: static shape mismatch")
    if not gpu_available[dtype]():
        return swiglu_cpu[dtype, rows, cols](gate, up)
    try:
        var ctx = get_gpu_context()
        return _swiglu_gpu_launch[dtype](ctx, gate, up)
    except:
        return swiglu_cpu[dtype, rows, cols](gate, up)


def swiglu_gpu_dynamic[
    dtype: DType
](gate: Tensor[dtype, 2], up: Tensor[dtype, 2]) -> Tensor[dtype, 2]:
    """Runtime-shaped GPU SwiGLU (CPU fallback on any GPU error)."""
    if not gpu_available[dtype]():
        return swiglu_cpu_dynamic[dtype](gate, up)
    try:
        var ctx = get_gpu_context()
        return _swiglu_gpu_launch[dtype](ctx, gate, up)
    except:
        return swiglu_cpu_dynamic[dtype](gate, up)


def swiglu_gpu_forward_with_saved[
    dtype: DType, rows: Int, cols: Int
](gate: Tensor[dtype, 2], up: Tensor[dtype, 2]) -> Tuple[
    Tensor[dtype, 2], List[Tensor[dtype, 2]]
]:
    var out = swiglu_gpu[dtype, rows, cols](gate, up)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(gate)
    saved.append(up)
    return (out, saved^)


def swiglu_gpu_backward[
    dtype: DType, rows: Int, cols: Int
](grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]) -> List[
    Tensor[dtype, 2]
]:
    """Backward for SwiGLU: out = silu(gate) * up.

    grad_up   = grad_out * silu(gate)
    grad_gate = grad_out * up * silu'(gate)
    `saved` = [gate, up].
    """
    var gate = saved[0]
    var up = saved[1]
    if not gpu_available[dtype]():
        return swiglu_cpu_backward[dtype, rows, cols](grad_out, saved)
    try:
        var ctx = get_gpu_context()
        var (grad_gate, grad_up) = _swiglu_bwd_gpu_launch[dtype](
            ctx, grad_out, gate, up
        )
        var result = List[Tensor[dtype, 2]]()
        result.append(grad_gate)
        result.append(grad_up)
        return result^
    except:
        return swiglu_cpu_backward[dtype, rows, cols](grad_out, saved)

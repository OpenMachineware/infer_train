# core/ops/gpu/rope_gpu.mojo
#
# GPU rotary position embedding, GPT-NeoX pairing (the layout Qwen2 uses).
#
# Pairs are (d, d + head_dim/2) for d in [0, head_dim/2); the frequency for
# pair `d` is theta^(-2d/head_dim).  `x` is [n_heads, T, head_dim]; positions
# are start_pos + row index.  All math is f32; f16 tensors are widened
# element-wise and cast back on store (matching the CPU kernel).
#
# The backward needs `start_pos`/`theta`, which the original typed
# `rope_gpu_backward(grad_out, saved)` signature does not carry; those ride in
# the erased dispatcher's saved list, so the GPU backward is exposed as
# `rope_gpu_backward_pos` and called from `op_autograd.rope_bwd_gpu`.

from ...tensor import Tensor
from ...utils import unimplemented
from ..cpu.rope_cpu import (
    rope_cpu,
    rope_cpu_dynamic,
    rope_cpu_backward,
)
from .gpu_runtime import (
    download3,
    get_gpu_context,
    grid1d,
    gpu_available,
    upload,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import global_idx, grid_dim, block_dim
from std.memory import Pointer
from std.origin import MutAnyOrigin
from std.math import cos, sin, exp, log
from std.utils.static_tuple import StaticTuple

comptime BLOCK = 256


# -- kernels ------------------------------------------------------------------
#
# One thread per (head, token, pair).  The pair index space is
# n_heads * n_tokens * (head_dim/2); within it, d = idx % half and the
# (head, token) index is idx // half.


def _rope_kernel_f32(
    x: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    n_tokens: Int32,
    head_dim: Int32,
    start_pos: Int32,
    ln_theta: Float32,
    n_pairs: Int32,
):
    var half = Int(head_dim) // 2
    var head_dim_i = Int(head_dim)
    var n_tok_i = Int(n_tokens)
    var n_pairs_i = Int(n_pairs)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n_pairs_i:
        var d = i % half
        var ht = i // half
        var t = ht % n_tok_i
        var pos = Float32(Int(start_pos) + t)
        var freq = exp(Float32(-2 * d) / Float32(head_dim_i) * ln_theta)
        var angle = pos * freq
        var c = cos(angle)
        var s = sin(angle)
        var base = ht * head_dim_i
        var x0 = x[unsafe_offset=base + d]
        var x1 = x[unsafe_offset=base + d + half]
        dst[unsafe_offset=base + d] = x0 * c - x1 * s
        dst[unsafe_offset=base + d + half] = x0 * s + x1 * c
        i += stride


def _rope_kernel_f16(
    x: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    n_tokens: Int32,
    head_dim: Int32,
    start_pos: Int32,
    ln_theta: Float32,
    n_pairs: Int32,
):
    var half = Int(head_dim) // 2
    var head_dim_i = Int(head_dim)
    var n_tok_i = Int(n_tokens)
    var n_pairs_i = Int(n_pairs)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n_pairs_i:
        var d = i % half
        var ht = i // half
        var t = ht % n_tok_i
        var pos = Float32(Int(start_pos) + t)
        var freq = exp(Float32(-2 * d) / Float32(head_dim_i) * ln_theta)
        var angle = pos * freq
        var c = cos(angle)
        var s = sin(angle)
        var base = ht * head_dim_i
        var x0 = Float32(x[unsafe_offset=base + d])
        var x1 = Float32(x[unsafe_offset=base + d + half])
        dst[unsafe_offset=base + d] = Scalar[DType.float16](x0 * c - x1 * s)
        dst[unsafe_offset=base + d + half] = Scalar[DType.float16](
            x0 * s + x1 * c
        )
        i += stride


def _rope_bwd_kernel_f32(
    grad_out: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    n_tokens: Int32,
    head_dim: Int32,
    start_pos: Int32,
    ln_theta: Float32,
    n_pairs: Int32,
):
    var half = Int(head_dim) // 2
    var head_dim_i = Int(head_dim)
    var n_tok_i = Int(n_tokens)
    var n_pairs_i = Int(n_pairs)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n_pairs_i:
        var d = i % half
        var ht = i // half
        var t = ht % n_tok_i
        var pos = Float32(Int(start_pos) + t)
        var freq = exp(Float32(-2 * d) / Float32(head_dim_i) * ln_theta)
        var angle = pos * freq
        var c = cos(angle)
        var s = sin(angle)
        var base = ht * head_dim_i
        var g0 = grad_out[unsafe_offset=base + d]
        var g1 = grad_out[unsafe_offset=base + d + half]
        dst[unsafe_offset=base + d] = g0 * c + g1 * s
        dst[unsafe_offset=base + d + half] = -g0 * s + g1 * c
        i += stride


def _rope_bwd_kernel_f16(
    grad_out: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    n_tokens: Int32,
    head_dim: Int32,
    start_pos: Int32,
    ln_theta: Float32,
    n_pairs: Int32,
):
    var half = Int(head_dim) // 2
    var head_dim_i = Int(head_dim)
    var n_tok_i = Int(n_tokens)
    var n_pairs_i = Int(n_pairs)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n_pairs_i:
        var d = i % half
        var ht = i // half
        var t = ht % n_tok_i
        var pos = Float32(Int(start_pos) + t)
        var freq = exp(Float32(-2 * d) / Float32(head_dim_i) * ln_theta)
        var angle = pos * freq
        var c = cos(angle)
        var s = sin(angle)
        var base = ht * head_dim_i
        var g0 = Float32(grad_out[unsafe_offset=base + d])
        var g1 = Float32(grad_out[unsafe_offset=base + d + half])
        dst[unsafe_offset=base + d] = Scalar[DType.float16](g0 * c + g1 * s)
        dst[unsafe_offset=base + d + half] = Scalar[DType.float16](
            -g0 * s + g1 * c
        )
        i += stride


# -- launch helpers -----------------------------------------------------------


def _rope_gpu_launch[
    dtype: DType
](
    ctx: DeviceContext,
    x: Tensor[dtype, 3],
    start_pos: Int,
    theta: Float32,
) raises -> Tensor[dtype, 3]:
    var n_tokens = x.shape()[1]
    var head_dim = x.shape()[2]
    var half = head_dim // 2
    var n_pairs = x.shape()[0] * n_tokens * half
    var x_buf = upload[dtype, 3](ctx, x)
    var dst_buf = ctx.enqueue_create_buffer[dtype](x.numel())
    var ln_theta = log(theta)
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_rope_kernel_f16](
            x_buf,
            dst_buf,
            Int32(n_tokens),
            Int32(head_dim),
            Int32(start_pos),
            ln_theta,
            Int32(n_pairs),
            grid_dim=grid1d(n_pairs, BLOCK),
            block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_rope_kernel_f32](
            x_buf,
            dst_buf,
            Int32(n_tokens),
            Int32(head_dim),
            Int32(start_pos),
            ln_theta,
            Int32(n_pairs),
            grid_dim=grid1d(n_pairs, BLOCK),
            block_dim=BLOCK,
        )
    var out = download3[dtype](ctx, dst_buf, x.shape())
    ctx.synchronize()
    return out


def _rope_bwd_gpu_launch[
    dtype: DType
](
    ctx: DeviceContext,
    grad_out: Tensor[dtype, 3],
    start_pos: Int,
    theta: Float32,
) raises -> Tensor[dtype, 3]:
    var n_tokens = grad_out.shape()[1]
    var head_dim = grad_out.shape()[2]
    var half = head_dim // 2
    var n_pairs = grad_out.shape()[0] * n_tokens * half
    var go_buf = upload[dtype, 3](ctx, grad_out)
    var dst_buf = ctx.enqueue_create_buffer[dtype](grad_out.numel())
    var ln_theta = log(theta)
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_rope_bwd_kernel_f16](
            go_buf,
            dst_buf,
            Int32(n_tokens),
            Int32(head_dim),
            Int32(start_pos),
            ln_theta,
            Int32(n_pairs),
            grid_dim=grid1d(n_pairs, BLOCK),
            block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_rope_bwd_kernel_f32](
            go_buf,
            dst_buf,
            Int32(n_tokens),
            Int32(head_dim),
            Int32(start_pos),
            ln_theta,
            Int32(n_pairs),
            grid_dim=grid1d(n_pairs, BLOCK),
            block_dim=BLOCK,
        )
    var out = download3[dtype](ctx, dst_buf, grad_out.shape())
    ctx.synchronize()
    return out


# -- public entry points ------------------------------------------------------


def rope_gpu[
    dtype: DType, n_heads: Int, head_dim: Int
](
    x: Tensor[dtype, 3], start_pos: Int, theta: Float32 = Float32(10000.0)
) -> Tensor[dtype, 3]:
    """Comptime-shaped GPU RoPE (CPU fallback on any GPU error)."""
    if x.shape()[0] != n_heads or x.shape()[2] != head_dim:
        unimplemented("rope_gpu: static shape mismatch")
    if not gpu_available[dtype]():
        return rope_cpu[dtype, n_heads, head_dim](x, start_pos, theta)
    try:
        var ctx = get_gpu_context()
        return _rope_gpu_launch[dtype](ctx, x, start_pos, theta)
    except:
        return rope_cpu[dtype, n_heads, head_dim](x, start_pos, theta)


def rope_gpu_dynamic[
    dtype: DType
](
    x: Tensor[dtype, 3], start_pos: Int, theta: Float32 = Float32(10000.0)
) -> Tensor[dtype, 3]:
    """Runtime-shaped GPU RoPE (CPU fallback on any GPU error)."""
    if not gpu_available[dtype]():
        return rope_cpu_dynamic[dtype](x, start_pos, theta)
    try:
        var ctx = get_gpu_context()
        return _rope_gpu_launch[dtype](ctx, x, start_pos, theta)
    except:
        return rope_cpu_dynamic[dtype](x, start_pos, theta)


def rope_gpu_backward_pos[
    dtype: DType
](
    grad_out: Tensor[dtype, 3],
    start_pos: Int,
    theta: Float32 = Float32(10000.0),
) -> Tensor[dtype, 3]:
    """GPU RoPE backward (inverse rotation) with explicit position/theta.

    Called from the erased dispatcher (`op_autograd.rope_bwd_gpu`), which has
    the start_pos/theta from the saved list.  CPU fallback on any GPU error.
    """
    if not gpu_available[dtype]():
        return rope_cpu_backward[dtype, 0, 0](
            grad_out, grad_out, start_pos, theta
        )
    try:
        var ctx = get_gpu_context()
        return _rope_bwd_gpu_launch[dtype](ctx, grad_out, start_pos, theta)
    except:
        return rope_cpu_backward[dtype, 0, 0](
            grad_out, grad_out, start_pos, theta
        )


def rope_gpu_forward_with_saved[
    dtype: DType, n_heads: Int, head_dim: Int
](
    x: Tensor[dtype, 3], start_pos: Int, theta: Float32 = Float32(10000.0)
) -> Tuple[Tensor[dtype, 3], List[Tensor[dtype, 3]]]:
    var out = rope_gpu[dtype, n_heads, head_dim](x, start_pos, theta)
    var saved = List[Tensor[dtype, 3]]()
    saved.append(x)
    return (out, saved^)


def rope_gpu_backward[
    dtype: DType, n_heads: Int, head_dim: Int
](grad_out: Tensor[dtype, 3], saved: List[Tensor[dtype, 3]]) -> List[
    Tensor[dtype, 3]
]:
    """Original typed backward signature; not on the registry path.

    The registry routes the RoPE backward through the erased dispatcher
    (`op_autograd.rope_bwd_gpu`), which has the start_pos/theta and calls
    `rope_gpu_backward_pos`.  This entry is kept for signature compatibility.
    """
    _ = grad_out
    _ = saved
    _ = n_heads
    _ = head_dim
    unimplemented("rope_gpu_backward: use the erased dispatcher")
    return List[Tensor[dtype, 3]]()

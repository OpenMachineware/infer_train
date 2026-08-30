# core/ops/gpu/fused_gpu.mojo
#
# GPU versions of the M5 fused kernels in core/ops/fused.  Each fuses a
# matmul (weight-major `w [out, in]`, the GGUF layout) with an epilogue so
# the intermediate tensor is never materialized:
#
#   fused_matmul_add_bias_gpu(x, w, bias)  y[i,j] = sum_k x[i,k]*w[j,k] + bias[j]
#   fused_matmul_add_gpu(x, w, b)          y[i,j] = sum_k x[i,k]*w[j,k] + b[i,j]
#   fused_matmul_rms_norm_gpu(x, w, eps)   y[i,:] = rms_norm(x[i,:] @ w^T, eps)
#   fused_swiglu_matmul_gpu(g, u, w)       y[i,j] = sum_f silu(g[i,f])*u[i,f]*w[j,f]
#
# The first three epilogues (bias / add / swiglu) are one thread per output
# element with a grid-stride K loop.  The RMSNorm epilogue is one block per
# row: the row is projected into the output buffer, a block barrier makes the
# writes visible, a `max.gpu.primitives.block` sum reduces the row sum of
# squares, then the row is scaled in place.
#
# All dot products accumulate in f32 (the M3 numerics rule); f16 operands are
# widened per element and cast back on store.  Every entry point falls back to
# the corresponding CPU fused kernel when the GPU is unavailable or any GPU
# step raises, so the registry dispatchers keep their non-`raises` signature.

from ...tensor import Tensor
from ...utils import unimplemented
from ..fused.matmul_add import fused_matmul_add_bias, fused_matmul_add
from ..fused.matmul_rms_norm import fused_matmul_rms_norm
from ..fused.swiglu_matmul import fused_swiglu_matmul
from .gpu_runtime import (
    download2,
    get_gpu_context,
    grid1d,
    gpu_available,
    upload,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier
from std.gpu import global_idx, grid_dim, block_dim, block_idx, thread_idx
from std.memory import Pointer
from std.origin import MutAnyOrigin
from std.math import exp, sqrt
from std.utils.static_tuple import StaticTuple

comptime BLOCK = 256


# -- silu (clamped, mirrors the CPU fused kernel) -----------------------------


def _silu_f32(x: Float32) -> Float32:
    if x < Float32(-20.0):
        return Float32(0.0)
    if x > Float32(20.0):
        return x
    return x / (Float32(1.0) + exp(-x))


# -- kernels: one thread per output element -----------------------------------


def _fused_matmul_add_bias_kernel_f32(
    x: Pointer[Float32, MutAnyOrigin],
    w: Pointer[Float32, MutAnyOrigin],
    bias: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    M: Int32,
    K: Int32,
    N: Int32,
):
    var K_i = Int(K)
    var N_i = Int(N)
    var n = Int(M) * N_i
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n:
        var row = i // N_i
        var col = i % N_i
        var acc = Float32(0.0)
        var k = 0
        while k < K_i:
            acc += x[row * K_i + k] * w[col * K_i + k]
            k += 1
        dst[i] = acc + bias[col]
        i += stride


def _fused_matmul_add_bias_kernel_f16(
    x: Pointer[Scalar[DType.float16], MutAnyOrigin],
    w: Pointer[Scalar[DType.float16], MutAnyOrigin],
    bias: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    M: Int32,
    K: Int32,
    N: Int32,
):
    var K_i = Int(K)
    var N_i = Int(N)
    var n = Int(M) * N_i
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n:
        var row = i // N_i
        var col = i % N_i
        var acc = Float32(0.0)
        var k = 0
        while k < K_i:
            acc += Float32(x[row * K_i + k]) * Float32(w[col * K_i + k])
            k += 1
        dst[i] = Scalar[DType.float16](acc + Float32(bias[col]))
        i += stride


def _fused_matmul_add_kernel_f32(
    x: Pointer[Float32, MutAnyOrigin],
    w: Pointer[Float32, MutAnyOrigin],
    b: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    M: Int32,
    K: Int32,
    N: Int32,
):
    var K_i = Int(K)
    var N_i = Int(N)
    var n = Int(M) * N_i
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n:
        var row = i // N_i
        var col = i % N_i
        var acc = Float32(0.0)
        var k = 0
        while k < K_i:
            acc += x[row * K_i + k] * w[col * K_i + k]
            k += 1
        dst[i] = acc + b[i]
        i += stride


def _fused_matmul_add_kernel_f16(
    x: Pointer[Scalar[DType.float16], MutAnyOrigin],
    w: Pointer[Scalar[DType.float16], MutAnyOrigin],
    b: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    M: Int32,
    K: Int32,
    N: Int32,
):
    var K_i = Int(K)
    var N_i = Int(N)
    var n = Int(M) * N_i
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n:
        var row = i // N_i
        var col = i % N_i
        var acc = Float32(0.0)
        var k = 0
        while k < K_i:
            acc += Float32(x[row * K_i + k]) * Float32(w[col * K_i + k])
            k += 1
        dst[i] = Scalar[DType.float16](acc + Float32(b[i]))
        i += stride


def _fused_swiglu_matmul_kernel_f32(
    gate: Pointer[Float32, MutAnyOrigin],
    up: Pointer[Float32, MutAnyOrigin],
    w: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    M: Int32,
    F: Int32,
    N: Int32,
):
    var F_i = Int(F)
    var N_i = Int(N)
    var n = Int(M) * N_i
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n:
        var row = i // N_i
        var col = i % N_i
        var acc = Float32(0.0)
        var f = 0
        while f < F_i:
            acc += _silu_f32(gate[row * F_i + f]) * up[row * F_i + f] * w[col * F_i + f]
            f += 1
        dst[i] = acc
        i += stride


def _fused_swiglu_matmul_kernel_f16(
    gate: Pointer[Scalar[DType.float16], MutAnyOrigin],
    up: Pointer[Scalar[DType.float16], MutAnyOrigin],
    w: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    M: Int32,
    F: Int32,
    N: Int32,
):
    var F_i = Int(F)
    var N_i = Int(N)
    var n = Int(M) * N_i
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n:
        var row = i // N_i
        var col = i % N_i
        var acc = Float32(0.0)
        var f = 0
        while f < F_i:
            var g = Float32(gate[row * F_i + f])
            var u = Float32(up[row * F_i + f])
            var wv = Float32(w[col * F_i + f])
            acc += _silu_f32(g) * u * wv
            f += 1
        dst[i] = Scalar[DType.float16](acc)
        i += stride


# -- kernel: matmul + rms_norm (one block per row) ----------------------------


def _fused_matmul_rms_norm_kernel_f32(
    x: Pointer[Float32, MutAnyOrigin],
    w: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    M: Int32,
    K: Int32,
    N: Int32,
    eps: Float32,
):
    var K_i = Int(K)
    var N_i = Int(N)
    var row = block_idx.x
    var tid = thread_idx.x
    # pass 1: project the row into dst
    var j = tid
    while j < N_i:
        var acc = Float32(0.0)
        var k = 0
        while k < K_i:
            acc += x[row * K_i + k] * w[j * K_i + k]
            k += 1
        dst[row * N_i + j] = acc
        j += BLOCK
    barrier()
    # pass 2: row sum of squares
    var ss = Float32(0.0)
    j = tid
    while j < N_i:
        var v = dst[row * N_i + j]
        ss += v * v
        j += BLOCK
    var ss_all = block_sum[block_size=BLOCK](ss)
    var inv = Float32(1.0) / sqrt(ss_all / Float32(N_i) + eps)
    # pass 3: normalize in place
    j = tid
    while j < N_i:
        dst[row * N_i + j] *= inv
        j += BLOCK


def _fused_matmul_rms_norm_kernel_f16(
    x: Pointer[Scalar[DType.float16], MutAnyOrigin],
    w: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    M: Int32,
    K: Int32,
    N: Int32,
    eps: Float32,
):
    var K_i = Int(K)
    var N_i = Int(N)
    var row = block_idx.x
    var tid = thread_idx.x
    # pass 1: project the row into dst (f32 accumulation, f16 store)
    var j = tid
    while j < N_i:
        var acc = Float32(0.0)
        var k = 0
        while k < K_i:
            acc += Float32(x[row * K_i + k]) * Float32(w[j * K_i + k])
            k += 1
        dst[row * N_i + j] = Scalar[DType.float16](acc)
        j += BLOCK
    barrier()
    # pass 2: row sum of squares (widen the f16 row back to f32)
    var ss = Float32(0.0)
    j = tid
    while j < N_i:
        var v = Float32(dst[row * N_i + j])
        ss += v * v
        j += BLOCK
    var ss_all = block_sum[block_size=BLOCK](ss)
    var inv = Float32(1.0) / sqrt(ss_all / Float32(N_i) + eps)
    # pass 3: normalize in place
    j = tid
    while j < N_i:
        dst[row * N_i + j] = Scalar[DType.float16](Float32(dst[row * N_i + j]) * inv)
        j += BLOCK


# -- launch helpers -----------------------------------------------------------


def _fused_matmul_add_bias_gpu_launch[dtype: DType](
    ctx: DeviceContext,
    x: Tensor[dtype, 2],
    w: Tensor[dtype, 2],
    bias: Tensor[dtype, 1],
) raises -> Tensor[dtype, 2]:
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]
    if K != w.shape()[1] or bias.shape()[0] != N:
        unimplemented("fused_matmul_add_bias_gpu: shape mismatch")
    var x_buf = upload[dtype, 2](ctx, x)
    var w_buf = upload[dtype, 2](ctx, w)
    var b_buf = upload[dtype, 1](ctx, bias)
    var dst_buf = ctx.enqueue_create_buffer[dtype](M * N)
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_fused_matmul_add_bias_kernel_f16](
            x_buf, w_buf, b_buf, dst_buf, Int32(M), Int32(K), Int32(N),
            grid_dim=grid1d(M * N, BLOCK), block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_fused_matmul_add_bias_kernel_f32](
            x_buf, w_buf, b_buf, dst_buf, Int32(M), Int32(K), Int32(N),
            grid_dim=grid1d(M * N, BLOCK), block_dim=BLOCK,
        )
    var out = download2[dtype](ctx, dst_buf, StaticTuple[Int, 2](M, N))
    ctx.synchronize()
    return out


def _fused_matmul_add_gpu_launch[dtype: DType](
    ctx: DeviceContext,
    x: Tensor[dtype, 2],
    w: Tensor[dtype, 2],
    b: Tensor[dtype, 2],
) raises -> Tensor[dtype, 2]:
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]
    if K != w.shape()[1] or b.shape() != StaticTuple[Int, 2](M, N):
        unimplemented("fused_matmul_add_gpu: shape mismatch")
    var x_buf = upload[dtype, 2](ctx, x)
    var w_buf = upload[dtype, 2](ctx, w)
    var b_buf = upload[dtype, 2](ctx, b)
    var dst_buf = ctx.enqueue_create_buffer[dtype](M * N)
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_fused_matmul_add_kernel_f16](
            x_buf, w_buf, b_buf, dst_buf, Int32(M), Int32(K), Int32(N),
            grid_dim=grid1d(M * N, BLOCK), block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_fused_matmul_add_kernel_f32](
            x_buf, w_buf, b_buf, dst_buf, Int32(M), Int32(K), Int32(N),
            grid_dim=grid1d(M * N, BLOCK), block_dim=BLOCK,
        )
    var out = download2[dtype](ctx, dst_buf, StaticTuple[Int, 2](M, N))
    ctx.synchronize()
    return out


def _fused_matmul_rms_norm_gpu_launch[dtype: DType](
    ctx: DeviceContext, x: Tensor[dtype, 2], w: Tensor[dtype, 2], eps: Float32
) raises -> Tensor[dtype, 2]:
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w.shape()[0]
    if K != w.shape()[1]:
        unimplemented("fused_matmul_rms_norm_gpu: K mismatch")
    var x_buf = upload[dtype, 2](ctx, x)
    var w_buf = upload[dtype, 2](ctx, w)
    var dst_buf = ctx.enqueue_create_buffer[dtype](M * N)
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_fused_matmul_rms_norm_kernel_f16](
            x_buf, w_buf, dst_buf, Int32(M), Int32(K), Int32(N), eps,
            grid_dim=M, block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_fused_matmul_rms_norm_kernel_f32](
            x_buf, w_buf, dst_buf, Int32(M), Int32(K), Int32(N), eps,
            grid_dim=M, block_dim=BLOCK,
        )
    var out = download2[dtype](ctx, dst_buf, StaticTuple[Int, 2](M, N))
    ctx.synchronize()
    return out


def _fused_swiglu_matmul_gpu_launch[dtype: DType](
    ctx: DeviceContext,
    gate: Tensor[dtype, 2],
    up: Tensor[dtype, 2],
    w: Tensor[dtype, 2],
) raises -> Tensor[dtype, 2]:
    var M = gate.shape()[0]
    var F = gate.shape()[1]
    var N = w.shape()[0]
    if up.shape() != StaticTuple[Int, 2](M, F) or w.shape()[1] != F:
        unimplemented("fused_swiglu_matmul_gpu: shape mismatch")
    var g_buf = upload[dtype, 2](ctx, gate)
    var u_buf = upload[dtype, 2](ctx, up)
    var w_buf = upload[dtype, 2](ctx, w)
    var dst_buf = ctx.enqueue_create_buffer[dtype](M * N)
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_fused_swiglu_matmul_kernel_f16](
            g_buf, u_buf, w_buf, dst_buf, Int32(M), Int32(F), Int32(N),
            grid_dim=grid1d(M * N, BLOCK), block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_fused_swiglu_matmul_kernel_f32](
            g_buf, u_buf, w_buf, dst_buf, Int32(M), Int32(F), Int32(N),
            grid_dim=grid1d(M * N, BLOCK), block_dim=BLOCK,
        )
    var out = download2[dtype](ctx, dst_buf, StaticTuple[Int, 2](M, N))
    ctx.synchronize()
    return out


# -- public entry points (CPU fallback on any GPU error) ----------------------


def fused_matmul_add_bias_gpu[dtype: DType](
    x: Tensor[dtype, 2], w: Tensor[dtype, 2], bias: Tensor[dtype, 1]
) -> Tensor[dtype, 2]:
    """Fused linear on GPU: y = x @ w^T + bias with w stored [out, in]."""
    if not gpu_available[dtype]():
        return fused_matmul_add_bias[dtype](x, w, bias)
    try:
        var ctx = get_gpu_context()
        return _fused_matmul_add_bias_gpu_launch[dtype](ctx, x, w, bias)
    except:
        return fused_matmul_add_bias[dtype](x, w, bias)


def fused_matmul_add_gpu[dtype: DType](
    x: Tensor[dtype, 2], w: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    """Fused matmul + elementwise add on GPU: y = x @ w^T + b."""
    if not gpu_available[dtype]():
        return fused_matmul_add[dtype](x, w, b)
    try:
        var ctx = get_gpu_context()
        return _fused_matmul_add_gpu_launch[dtype](ctx, x, w, b)
    except:
        return fused_matmul_add[dtype](x, w, b)


def fused_matmul_rms_norm_gpu[dtype: DType](
    x: Tensor[dtype, 2], w: Tensor[dtype, 2], eps: Float32 = Float32(1e-5)
) -> Tensor[dtype, 2]:
    """Fused matmul + RMSNorm on GPU: y = rms_norm(x @ w^T, eps)."""
    if not gpu_available[dtype]():
        return fused_matmul_rms_norm[dtype](x, w, eps)
    try:
        var ctx = get_gpu_context()
        return _fused_matmul_rms_norm_gpu_launch[dtype](ctx, x, w, eps)
    except:
        return fused_matmul_rms_norm[dtype](x, w, eps)


def fused_swiglu_matmul_gpu[dtype: DType](
    gate: Tensor[dtype, 2], up: Tensor[dtype, 2], w: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    """Fused SwiGLU + matmul on GPU: y = silu(gate)*up @ w^T, w [out, in]."""
    if not gpu_available[dtype]():
        return fused_swiglu_matmul[dtype](gate, up, w)
    try:
        var ctx = get_gpu_context()
        return _fused_swiglu_matmul_gpu_launch[dtype](ctx, gate, up, w)
    except:
        return fused_swiglu_matmul[dtype](gate, up, w)

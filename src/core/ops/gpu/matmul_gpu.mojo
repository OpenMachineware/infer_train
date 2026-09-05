# core/ops/gpu/matmul_gpu.mojo
#
# GPU matrix multiplication: out = a @ b, a [M, K], b [K, N].
#
# Naive kernel: one thread per output element, looping over K.  Dot products
# always accumulate in f32 (f16 inputs are widened per element) and cast back
# to the element dtype on store - the same policy as the CPU kernel, because
# f16 accumulation over K ~ 1536 terms injects ~4% rounding noise per dot
# product.  Tiled/shared-memory GEMM is the planned performance upgrade.

from ...tensor import Tensor
from ...utils import unimplemented
from ..cpu.matmul_cpu import (
    matmul_cpu,
    matmul_cpu_dynamic,
    matmul_cpu_backward,
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


def _matmul_kernel_f32(
    a: Pointer[Float32, MutAnyOrigin],
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
            acc += (
                a[unsafe_offset=row * K_i + k] * b[unsafe_offset=k * N_i + col]
            )
            k += 1
        dst[unsafe_offset=i] = acc
        i += stride


def _matmul_kernel_f16(
    a: Pointer[Scalar[DType.float16], MutAnyOrigin],
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
            acc += Float32(a[unsafe_offset=row * K_i + k]) * Float32(
                b[unsafe_offset=k * N_i + col]
            )
            k += 1
        dst[unsafe_offset=i] = Scalar[DType.float16](acc)
        i += stride


def _matmul_bwd_a_kernel_f32(
    grad_out: Pointer[Float32, MutAnyOrigin],
    b: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    M: Int32,
    K: Int32,
    N: Int32,
):
    """grad_a[i, k] = sum_j grad_out[i, j] * b[k, j]."""
    var N_i = Int(N)
    var n = Int(M) * Int(K)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n:
        var row = i // Int(K)
        var k = i % Int(K)
        var acc = Float32(0.0)
        var j = 0
        while j < N_i:
            acc += (
                grad_out[unsafe_offset=row * N_i + j]
                * b[unsafe_offset=k * N_i + j]
            )
            j += 1
        dst[unsafe_offset=i] = acc
        i += stride


def _matmul_bwd_a_kernel_f16(
    grad_out: Pointer[Scalar[DType.float16], MutAnyOrigin],
    b: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    M: Int32,
    K: Int32,
    N: Int32,
):
    var N_i = Int(N)
    var n = Int(M) * Int(K)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n:
        var row = i // Int(K)
        var k = i % Int(K)
        var acc = Float32(0.0)
        var j = 0
        while j < N_i:
            acc += Float32(grad_out[unsafe_offset=row * N_i + j]) * Float32(
                b[unsafe_offset=k * N_i + j]
            )
            j += 1
        dst[unsafe_offset=i] = Scalar[DType.float16](acc)
        i += stride


def _matmul_bwd_b_kernel_f32(
    a: Pointer[Float32, MutAnyOrigin],
    grad_out: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    M: Int32,
    K: Int32,
    N: Int32,
):
    """grad_b[k, j] = sum_i a[i, k] * grad_out[i, j]."""
    var n = Int(K) * Int(N)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n:
        var k = i // Int(N)
        var j = i % Int(N)
        var acc = Float32(0.0)
        var row = 0
        while row < Int(M):
            acc += (
                a[unsafe_offset=row * Int(K) + k]
                * grad_out[unsafe_offset=row * Int(N) + j]
            )
            row += 1
        dst[unsafe_offset=i] = acc
        i += stride


def _matmul_bwd_b_kernel_f16(
    a: Pointer[Scalar[DType.float16], MutAnyOrigin],
    grad_out: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    M: Int32,
    K: Int32,
    N: Int32,
):
    var n = Int(K) * Int(N)
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < n:
        var k = i // Int(N)
        var j = i % Int(N)
        var acc = Float32(0.0)
        var row = 0
        while row < Int(M):
            acc += Float32(a[unsafe_offset=row * Int(K) + k]) * Float32(
                grad_out[unsafe_offset=row * Int(N) + j]
            )
            row += 1
        dst[unsafe_offset=i] = Scalar[DType.float16](acc)
        i += stride


# -- launch helpers -----------------------------------------------------------


def _matmul_gpu_launch[
    dtype: DType
](
    ctx: DeviceContext, a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) raises -> Tensor[dtype, 2]:
    var M = a.shape()[0]
    var K = a.shape()[1]
    if K != b.shape()[0]:
        unimplemented("matmul_gpu: K mismatch between a and b")
    var N = b.shape()[1]
    var a_buf = upload[dtype, 2](ctx, a)
    var b_buf = upload[dtype, 2](ctx, b)
    var n = M * N
    var dst_buf = ctx.enqueue_create_buffer[dtype](n)
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_matmul_kernel_f16](
            a_buf,
            b_buf,
            dst_buf,
            Int32(M),
            Int32(K),
            Int32(N),
            grid_dim=grid1d(n, BLOCK),
            block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_matmul_kernel_f32](
            a_buf,
            b_buf,
            dst_buf,
            Int32(M),
            Int32(K),
            Int32(N),
            grid_dim=grid1d(n, BLOCK),
            block_dim=BLOCK,
        )
    var out = download2[dtype](ctx, dst_buf, StaticTuple[Int, 2](M, N))
    ctx.synchronize()
    return out


def _matmul_bwd_gpu_launch[
    dtype: DType
](
    ctx: DeviceContext,
    grad_out: Tensor[dtype, 2],
    a: Tensor[dtype, 2],
    b: Tensor[dtype, 2],
) raises -> Tuple[Tensor[dtype, 2], Tensor[dtype, 2]]:
    var M = a.shape()[0]
    var K = a.shape()[1]
    var N = b.shape()[1]
    var go_buf = upload[dtype, 2](ctx, grad_out)
    var a_buf = upload[dtype, 2](ctx, a)
    var b_buf = upload[dtype, 2](ctx, b)
    var ga_buf = ctx.enqueue_create_buffer[dtype](M * K)
    var gb_buf = ctx.enqueue_create_buffer[dtype](K * N)
    comptime if dtype == DType.float16:
        ctx.enqueue_function[_matmul_bwd_a_kernel_f16](
            go_buf,
            b_buf,
            ga_buf,
            Int32(M),
            Int32(K),
            Int32(N),
            grid_dim=grid1d(M * K, BLOCK),
            block_dim=BLOCK,
        )
        ctx.enqueue_function[_matmul_bwd_b_kernel_f16](
            a_buf,
            go_buf,
            gb_buf,
            Int32(M),
            Int32(K),
            Int32(N),
            grid_dim=grid1d(K * N, BLOCK),
            block_dim=BLOCK,
        )
    else:
        ctx.enqueue_function[_matmul_bwd_a_kernel_f32](
            go_buf,
            b_buf,
            ga_buf,
            Int32(M),
            Int32(K),
            Int32(N),
            grid_dim=grid1d(M * K, BLOCK),
            block_dim=BLOCK,
        )
        ctx.enqueue_function[_matmul_bwd_b_kernel_f32](
            a_buf,
            go_buf,
            gb_buf,
            Int32(M),
            Int32(K),
            Int32(N),
            grid_dim=grid1d(K * N, BLOCK),
            block_dim=BLOCK,
        )
    var grad_a = download2[dtype](ctx, ga_buf, StaticTuple[Int, 2](M, K))
    var grad_b = download2[dtype](ctx, gb_buf, StaticTuple[Int, 2](K, N))
    ctx.synchronize()
    return (grad_a, grad_b)


# -- public entry points ------------------------------------------------------


def matmul_gpu[
    dtype: DType, M: Int, N: Int, K: Int
](a: Tensor[dtype, 2], b: Tensor[dtype, 2]) -> Tensor[dtype, 2]:
    """Comptime-shaped GPU matmul (CPU fallback on any GPU error)."""
    if a.shape() != StaticTuple[Int, 2](M, K):
        unimplemented("matmul_gpu: static shape mismatch")
    if not gpu_available[dtype]():
        return matmul_cpu[dtype, M, N, K](a, b)
    try:
        var ctx = get_gpu_context()
        return _matmul_gpu_launch[dtype](ctx, a, b)
    except:
        return matmul_cpu[dtype, M, N, K](a, b)


def matmul_gpu_dynamic[
    dtype: DType
](a: Tensor[dtype, 2], b: Tensor[dtype, 2]) -> Tensor[dtype, 2]:
    """Runtime-shaped GPU matmul (CPU fallback on any GPU error)."""
    if not gpu_available[dtype]():
        return matmul_cpu_dynamic[dtype](a, b)
    try:
        var ctx = get_gpu_context()
        return _matmul_gpu_launch[dtype](ctx, a, b)
    except:
        return matmul_cpu_dynamic[dtype](a, b)


def matmul_gpu_forward_with_saved[
    dtype: DType
](a: Tensor[dtype, 2], b: Tensor[dtype, 2]) -> Tuple[
    Tensor[dtype, 2], List[Tensor[dtype, 2]]
]:
    var out = matmul_gpu_dynamic[dtype](a, b)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(a)
    saved.append(b)
    return (out, saved^)


def matmul_gpu_backward[
    dtype: DType
](grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]) -> List[
    Tensor[dtype, 2]
]:
    """Backward for out = a @ b.

    grad_a = grad_out @ b^T; grad_b = a^T @ grad_out.  `saved` = [a, b].
    """
    var a = saved[0]
    var b = saved[1]
    if not gpu_available[dtype]():
        return matmul_cpu_backward[dtype](grad_out, saved)
    try:
        var ctx = get_gpu_context()
        var (grad_a, grad_b) = _matmul_bwd_gpu_launch[dtype](
            ctx, grad_out, a, b
        )
        var result = List[Tensor[dtype, 2]]()
        result.append(grad_a)
        result.append(grad_b)
        return result^
    except:
        return matmul_cpu_backward[dtype](grad_out, saved)

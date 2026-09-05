# core/ops/gpu/gpu_pipeline.mojo
#
# Async GPU/CPU pipeline built on a *persistent* `DeviceContext`.
#
# The per-op `*_gpu` entry points are deliberately synchronous: each one does
# HtoD -> kernel -> DtoH -> synchronize, so the CPU sits idle while the GPU
# works and every op pays a full round trip + sync.  `GpuPipeline` removes
# both costs for a *sequence* of ops:
#
#   1. Constant weights are uploaded ONCE and kept in device buffers, so a
#      sequence (or repeated iterations) reuses them instead of re-uploading
#      per op.
#   2. Activations stay on the device across ops - no per-op DtoH + HtoD
#      round trip - so the CPU is not blocked on a sync between every op.
#   3. Every op is enqueued asynchronously on the context's stream; the CPU
#      keeps enqueuing (and can do other work) while the GPU executes, and we
#      synchronize exactly once, when a result is actually needed on the host.
#
# The GPU-resident ops (`linear`, `add2`, `swiglu`) take and return
# `DeviceBuffer`s and never synchronize.  `to_device`/`to_host` are the only
# host<->device boundaries; `to_host` is the single sync point.
#
# Numerics match the standalone kernels: dot products accumulate in f32 and
# f16 operands are widened per element.

from ...tensor import Tensor
from ...utils import unimplemented
from .gpu_runtime import download2, grid1d, gpu_available, upload
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import global_idx, grid_dim, block_dim
from std.memory import Pointer
from std.origin import MutAnyOrigin
from std.math import exp
from std.utils.static_tuple import StaticTuple

comptime BLOCK = 256


def _silu_f32(x: Float32) -> Float32:
    if x < Float32(-20.0):
        return Float32(0.0)
    if x > Float32(20.0):
        return x
    return x / (Float32(1.0) + exp(-x))


# -- GPU-resident kernels (device pointers in and out) ------------------------


def _pipe_linear_kernel_f32(
    x: Pointer[Float32, MutAnyOrigin],
    w: Pointer[Float32, MutAnyOrigin],
    bias: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    M: Int32,
    K: Int32,
    N: Int32,
):
    """dst[i,j] = sum_k x[i,k]*w[j,k] + bias[j]  (w weight-major [N,K])."""
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
                x[unsafe_offset=row * K_i + k] * w[unsafe_offset=col * K_i + k]
            )
            k += 1
        dst[unsafe_offset=i] = acc + bias[unsafe_offset=col]
        i += stride


def _pipe_linear_kernel_f16(
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
            acc += Float32(x[unsafe_offset=row * K_i + k]) * Float32(
                w[unsafe_offset=col * K_i + k]
            )
            k += 1
        dst[unsafe_offset=i] = Scalar[DType.float16](
            acc + Float32(bias[unsafe_offset=col])
        )
        i += stride


def _pipe_add_kernel_f32(
    a: Pointer[Float32, MutAnyOrigin],
    b: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    n: Int32,
):
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(n):
        dst[i] = a[i] + b[i]
        i += stride


def _pipe_add_kernel_f16(
    a: Pointer[Scalar[DType.float16], MutAnyOrigin],
    b: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    n: Int32,
):
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(n):
        dst[i] = Scalar[DType.float16](Float32(a[i]) + Float32(b[i]))
        i += stride


def _pipe_swiglu_kernel_f32(
    gate: Pointer[Float32, MutAnyOrigin],
    up: Pointer[Float32, MutAnyOrigin],
    dst: Pointer[Float32, MutAnyOrigin],
    n: Int32,
):
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(n):
        dst[i] = _silu_f32(gate[i]) * up[i]
        i += stride


def _pipe_swiglu_kernel_f16(
    gate: Pointer[Scalar[DType.float16], MutAnyOrigin],
    up: Pointer[Scalar[DType.float16], MutAnyOrigin],
    dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
    n: Int32,
):
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(n):
        var g = Float32(gate[i])
        var u = Float32(up[i])
        dst[i] = Scalar[DType.float16](_silu_f32(g) * u)
        i += stride


# -- the pipeline -------------------------------------------------------------


struct GpuPipeline:
    """A persistent GPU context that enqueues work asynchronously.

    Create one pipeline and reuse it across a sequence of ops (or across
    repeated inference iterations).  Weights uploaded with `to_device` stay on
    the device and can be passed to the GPU-resident ops without re-uploading.
    Call `sync` (or `to_host`) to wait for all enqueued work.
    """

    var ctx: DeviceContext

    def __init__(out self) raises:
        self.ctx = DeviceContext()

    def name(self) -> String:
        return self.ctx.name()

    def available[dtype: DType]() -> Bool:
        return gpu_available[dtype]()

    def sync(self) raises:
        self.ctx.synchronize()

    # -- host <-> device boundaries ---------------------------------------

    def to_device2[
        dtype: DType
    ](self, t: Tensor[dtype, 2]) raises -> DeviceBuffer[dtype]:
        """Upload a rank-2 tensor (async, no sync)."""
        return upload[dtype, 2](self.ctx, t)

    def to_device1[
        dtype: DType
    ](self, t: Tensor[dtype, 1]) raises -> DeviceBuffer[dtype]:
        """Upload a rank-1 tensor (async, no sync)."""
        return upload[dtype, 1](self.ctx, t)

    def to_host2[
        dtype: DType
    ](
        self, buf: DeviceBuffer[dtype], shape: StaticTuple[Int, 2]
    ) raises -> Tensor[dtype, 2]:
        """Download a rank-2 buffer and synchronize (the single sync point)."""
        var t = download2[dtype](self.ctx, buf, shape)
        self.ctx.synchronize()
        return t

    # -- GPU-resident ops (enqueue only, no sync) -------------------------

    def linear[
        dtype: DType
    ](
        self,
        x: DeviceBuffer[dtype],
        w: DeviceBuffer[dtype],
        bias: DeviceBuffer[dtype],
        M: Int,
        K: Int,
        N: Int,
    ) raises -> DeviceBuffer[dtype]:
        """y = x @ w^T + bias with w weight-major [N, K] (async)."""
        var dst = self.ctx.enqueue_create_buffer[dtype](M * N)
        comptime if dtype == DType.float16:
            self.ctx.enqueue_function[_pipe_linear_kernel_f16](
                x,
                w,
                bias,
                dst,
                Int32(M),
                Int32(K),
                Int32(N),
                grid_dim=grid1d(M * N, BLOCK),
                block_dim=BLOCK,
            )
        else:
            self.ctx.enqueue_function[_pipe_linear_kernel_f32](
                x,
                w,
                bias,
                dst,
                Int32(M),
                Int32(K),
                Int32(N),
                grid_dim=grid1d(M * N, BLOCK),
                block_dim=BLOCK,
            )
        return dst

    def add2[
        dtype: DType
    ](
        self, a: DeviceBuffer[dtype], b: DeviceBuffer[dtype], n: Int
    ) raises -> DeviceBuffer[dtype]:
        """y = a + b elementwise (async)."""
        var dst = self.ctx.enqueue_create_buffer[dtype](n)
        comptime if dtype == DType.float16:
            self.ctx.enqueue_function[_pipe_add_kernel_f16](
                a,
                b,
                dst,
                Int32(n),
                grid_dim=grid1d(n, BLOCK),
                block_dim=BLOCK,
            )
        else:
            self.ctx.enqueue_function[_pipe_add_kernel_f32](
                a,
                b,
                dst,
                Int32(n),
                grid_dim=grid1d(n, BLOCK),
                block_dim=BLOCK,
            )
        return dst

    def swiglu[
        dtype: DType
    ](
        self, gate: DeviceBuffer[dtype], up: DeviceBuffer[dtype], n: Int
    ) raises -> DeviceBuffer[dtype]:
        """y = silu(gate) * up elementwise (async)."""
        var dst = self.ctx.enqueue_create_buffer[dtype](n)
        comptime if dtype == DType.float16:
            self.ctx.enqueue_function[_pipe_swiglu_kernel_f16](
                gate,
                up,
                dst,
                Int32(n),
                grid_dim=grid1d(n, BLOCK),
                block_dim=BLOCK,
            )
        else:
            self.ctx.enqueue_function[_pipe_swiglu_kernel_f32](
                gate,
                up,
                dst,
                Int32(n),
                grid_dim=grid1d(n, BLOCK),
                block_dim=BLOCK,
            )
        return dst

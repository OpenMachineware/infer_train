# core/ops/gpu/gpu_runtime.mojo
#
# Shared host-side plumbing for the Metal GPU kernels in this package.
#
# Design (M7):
#   * Every `*_gpu` op is a *host-side entry point*: it uploads the input
#     tensors to device buffers, launches the per-dtype kernel, downloads the
#     result into a fresh host `Tensor`, and synchronizes.  The returned
#     tensor is therefore always host-resident (`Device.CPU`), which keeps the
#     rest of the framework (autograd, optimizer, registry) unchanged.
#   * Kernels are written once per concrete element dtype (f16 / f32) because
#     `DeviceContext.enqueue_function[func]` compiles a *concrete* function;
#     a comptime-generic kernel cannot be passed as the kernel parameter.
#   * `DType.float64` is never run on the GPU: Metal has no f64, so the ops
#     fall back to the CPU kernels (see `gpu_available`).
#
# Mojo 1.0 note on context lifetime: the language has no supported mechanism
# for a process-wide cached object (no `static var`, module-level globals are
# rejected), so each op call constructs its own `DeviceContext`.  Steady-state
# construction costs a few hundred microseconds; the per-op HtoD/DtoH traffic
# dominates for all but the smallest tensors.  Threading a shared context
# through the registry is the planned optimization path.

from ...device import has_metal_gpu
from ...tensor import Tensor
from max.gpu.host import DeviceBuffer, DeviceContext
from std.utils.static_tuple import StaticTuple


def gpu_available[dtype: DType]() -> Bool:
    """True when the op can run on the GPU for `dtype`.

    Metal has no f64, so double-precision always stays on the CPU.
    """
    comptime if dtype == DType.float64:
        return False
    return has_metal_gpu()


def get_gpu_context() raises -> DeviceContext:
    """Create a fresh `DeviceContext` for the default accelerator."""
    return DeviceContext()


def grid1d(n: Int, block: Int) -> Int:
    """Number of 1-D blocks of `block` threads covering `n` elements."""
    if n < 1:
        return 1
    return (n + block - 1) // block


def upload[
    dtype: DType, rank: Int
](ctx: DeviceContext, t: Tensor[dtype, rank]) raises -> DeviceBuffer[dtype]:
    """Create a device buffer and asynchronously copy `t` into it."""
    var buf = ctx.enqueue_create_buffer[dtype](t.numel())
    ctx.enqueue_copy[dtype](buf, t.data())
    return buf


def download1[
    dtype: DType
](
    ctx: DeviceContext, buf: DeviceBuffer[dtype], shape: StaticTuple[Int, 1]
) raises -> Tensor[dtype, 1]:
    """Create a host tensor of `shape` and asynchronously copy `buf` into it."""
    var t = Tensor[dtype, 1](shape)
    ctx.enqueue_copy[dtype](t.data(), buf)
    return t


def download2[
    dtype: DType
](
    ctx: DeviceContext, buf: DeviceBuffer[dtype], shape: StaticTuple[Int, 2]
) raises -> Tensor[dtype, 2]:
    """Create a host tensor of `shape` and asynchronously copy `buf` into it."""
    var t = Tensor[dtype, 2](shape)
    ctx.enqueue_copy[dtype](t.data(), buf)
    return t


def download3[
    dtype: DType
](
    ctx: DeviceContext, buf: DeviceBuffer[dtype], shape: StaticTuple[Int, 3]
) raises -> Tensor[dtype, 3]:
    """Create a host tensor of `shape` and asynchronously copy `buf` into it."""
    var t = Tensor[dtype, 3](shape)
    ctx.enqueue_copy[dtype](t.data(), buf)
    return t

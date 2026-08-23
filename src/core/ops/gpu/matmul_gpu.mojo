# core/ops/gpu/matmul_gpu.mojo
#
# GPU matrix multiplication (P0).
#
# Metal does not support `DType.float64`, so that case always falls back to
# CPU.  On M1 the pure-Mojo host GPU runtime is not yet linked (the `max.gpu`
# `DeviceContext`/`enqueue_function` API ships in a separate package), so the
# forward path delegates to the CPU kernel while keeping the dispatch
# structure in place.  When the Metal backend is wired in, only the body of
# these functions changes.

from ...device import has_metal_gpu
from ...tensor import Tensor
from ...utils import unimplemented
from ..cpu.matmul_cpu import (
    matmul_cpu,
    matmul_cpu_dynamic,
)


def _gpu_available[dtype: DType]() -> Bool:
    """True when an accelerator is present and the dtype is Metal-safe."""
    comptime if dtype == DType.float64:
        return False
    return has_metal_gpu()


def matmul_gpu[dtype: DType, M: Int, N: Int, K: Int](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    """Comptime-shaped GPU matmul (M1: delegates to CPU kernel)."""
    _ = _gpu_available[dtype]()
    return matmul_cpu[dtype, M, N, K](a, b)


def matmul_gpu_dynamic[dtype: DType](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    """Runtime-shaped GPU matmul (M1: delegates to CPU kernel)."""
    _ = _gpu_available[dtype]()
    return matmul_cpu_dynamic[dtype](a, b)


def matmul_gpu_forward_with_saved[dtype: DType](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]]:
    var out = matmul_gpu_dynamic[dtype](a, b)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(a)
    saved.append(b)
    return (out, saved^)


def matmul_gpu_backward[dtype: DType](
    grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]
) -> List[Tensor[dtype, 2]]:
    _ = grad_out
    _ = saved
    unimplemented("matmul_gpu_backward")
    return List[Tensor[dtype, 2]]()

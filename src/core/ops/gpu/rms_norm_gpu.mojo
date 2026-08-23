# core/ops/gpu/rms_norm_gpu.mojo
#
# GPU RMSNorm (P0).  Same signature as the CPU version; M1 delegates to the
# CPU kernel until the Metal reduction (via `parallel_reduce`-style device
# code) is wired in.

from ...tensor import Tensor
from ...utils import unimplemented
from ..cpu.rms_norm_cpu import rms_norm_cpu, rms_norm_cpu_dynamic


def rms_norm_gpu[dtype: DType, dim: Int](
    x: Tensor[dtype, 2], eps: Float32 = Float32(1e-5)
) -> Tensor[dtype, 2]:
    return rms_norm_cpu[dtype, dim](x, eps)


def rms_norm_gpu_dynamic[dtype: DType](
    x: Tensor[dtype, 2], eps: Float32 = Float32(1e-5)
) -> Tensor[dtype, 2]:
    return rms_norm_cpu_dynamic[dtype](x, eps)


def rms_norm_gpu_forward_with_saved[dtype: DType, dim: Int](
    x: Tensor[dtype, 2], eps: Float32 = Float32(1e-5)
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]]:
    var out = rms_norm_gpu[dtype, dim](x, eps)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(x)
    return (out, saved^)


def rms_norm_gpu_backward[dtype: DType, dim: Int](
    grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]
) -> List[Tensor[dtype, 2]]:
    _ = grad_out
    _ = saved
    unimplemented("rms_norm_gpu_backward")
    return List[Tensor[dtype, 2]]()

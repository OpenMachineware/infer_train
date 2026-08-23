# core/ops/gpu/softmax_gpu.mojo
#
# GPU stable softmax (P0).  M1 delegates to the CPU kernel until the Metal
# reduction path is wired in.

from ...tensor import Tensor
from ...utils import unimplemented
from ..cpu.softmax_cpu import softmax_cpu, softmax_cpu_dynamic


def softmax_gpu[dtype: DType, dim: Int](
    x: Tensor[dtype, 2]
) -> Tensor[dtype, 2] where dtype.is_floating_point():
    return softmax_cpu[dtype, dim](x)


def softmax_gpu_dynamic[dtype: DType](
    x: Tensor[dtype, 2]
) -> Tensor[dtype, 2] where dtype.is_floating_point():
    return softmax_cpu_dynamic[dtype](x)


def softmax_gpu_forward_with_saved[dtype: DType, dim: Int](
    x: Tensor[dtype, 2]
) -> Tuple[
    Tensor[dtype, 2], List[Tensor[dtype, 2]]
] where dtype.is_floating_point():
    var out = softmax_gpu[dtype, dim](x)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(out)
    return (out, saved^)


def softmax_gpu_backward[dtype: DType, dim: Int](
    grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]
) -> List[Tensor[dtype, 2]]:
    _ = grad_out
    _ = saved
    unimplemented("softmax_gpu_backward")
    return List[Tensor[dtype, 2]]()

# core/ops/gpu/swiglu_gpu.mojo
#
# GPU SwiGLU (M3: delegates to the CPU kernel).

from ...device import has_metal_gpu
from ...tensor import Tensor
from ...utils import unimplemented
from ..cpu.swiglu_cpu import (
    swiglu_cpu,
    swiglu_cpu_dynamic,
)


def _gpu_available[dtype: DType]() -> Bool:
    comptime if dtype == DType.float64:
        return False
    return has_metal_gpu()


def swiglu_gpu[dtype: DType, rows: Int, cols: Int](
    gate: Tensor[dtype, 2], up: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    _ = _gpu_available[dtype]()
    return swiglu_cpu[dtype, rows, cols](gate, up)


def swiglu_gpu_dynamic[dtype: DType](
    gate: Tensor[dtype, 2], up: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    _ = _gpu_available[dtype]()
    return swiglu_cpu_dynamic[dtype](gate, up)


def swiglu_gpu_forward_with_saved[dtype: DType, rows: Int, cols: Int](
    gate: Tensor[dtype, 2], up: Tensor[dtype, 2]
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]]:
    var out = swiglu_gpu[dtype, rows, cols](gate, up)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(gate)
    saved.append(up)
    return (out, saved^)


def swiglu_gpu_backward[dtype: DType, rows: Int, cols: Int](
    grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]
) -> List[Tensor[dtype, 2]]:
    _ = grad_out
    _ = saved
    unimplemented("swiglu_gpu_backward")
    return List[Tensor[dtype, 2]]()

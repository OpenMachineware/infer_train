# core/ops/gpu/add_gpu.mojo
#
# GPU element-wise add (M3: delegates to the CPU kernel).

from ...device import has_metal_gpu
from ...tensor import Tensor
from ...utils import unimplemented
from ..cpu.add_cpu import (
    add_cpu,
    add_cpu_dynamic,
    add_row_cpu,
)


def _gpu_available[dtype: DType]() -> Bool:
    comptime if dtype == DType.float64:
        return False
    return has_metal_gpu()


def add_gpu[dtype: DType, rows: Int, cols: Int](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    _ = _gpu_available[dtype]()
    return add_cpu[dtype, rows, cols](a, b)


def add_gpu_dynamic[dtype: DType](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    _ = _gpu_available[dtype]()
    return add_cpu_dynamic[dtype](a, b)


def add_row_gpu[dtype: DType](
    x: Tensor[dtype, 2], bias: Tensor[dtype, 1]
) -> Tensor[dtype, 2]:
    _ = _gpu_available[dtype]()
    return add_row_cpu[dtype](x, bias)


def add_gpu_forward_with_saved[dtype: DType, rows: Int, cols: Int](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]]:
    var out = add_gpu[dtype, rows, cols](a, b)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(a)
    saved.append(b)
    return (out, saved^)


def add_gpu_backward[dtype: DType, rows: Int, cols: Int](
    grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]
) -> List[Tensor[dtype, 2]]:
    _ = grad_out
    _ = saved
    unimplemented("add_gpu_backward")
    return List[Tensor[dtype, 2]]()

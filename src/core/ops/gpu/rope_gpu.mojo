# core/ops/gpu/rope_gpu.mojo
#
# GPU RoPE (M3: delegates to the CPU kernel; Metal wiring lands with the
# host GPU runtime).

from ...device import has_metal_gpu
from ...tensor import Tensor
from ...utils import unimplemented
from ..cpu.rope_cpu import (
    rope_cpu,
    rope_cpu_dynamic,
)


def _gpu_available[dtype: DType]() -> Bool:
    comptime if dtype == DType.float64:
        return False
    return has_metal_gpu()


def rope_gpu[dtype: DType, n_heads: Int, head_dim: Int](
    x: Tensor[dtype, 3], start_pos: Int, theta: Float32 = Float32(10000.0)
) -> Tensor[dtype, 3]:
    _ = _gpu_available[dtype]()
    return rope_cpu[dtype, n_heads, head_dim](x, start_pos, theta)


def rope_gpu_dynamic[dtype: DType](
    x: Tensor[dtype, 3], start_pos: Int, theta: Float32 = Float32(10000.0)
) -> Tensor[dtype, 3]:
    _ = _gpu_available[dtype]()
    return rope_cpu_dynamic[dtype](x, start_pos, theta)


def rope_gpu_forward_with_saved[dtype: DType, n_heads: Int, head_dim: Int](
    x: Tensor[dtype, 3], start_pos: Int, theta: Float32 = Float32(10000.0)
) -> Tuple[Tensor[dtype, 3], List[Tensor[dtype, 3]]]:
    var out = rope_gpu[dtype, n_heads, head_dim](x, start_pos, theta)
    var saved = List[Tensor[dtype, 3]]()
    saved.append(x)
    return (out, saved^)


def rope_gpu_backward[dtype: DType, n_heads: Int, head_dim: Int](
    grad_out: Tensor[dtype, 3], saved: List[Tensor[dtype, 3]]
) -> List[Tensor[dtype, 3]]:
    _ = grad_out
    _ = saved
    unimplemented("rope_gpu_backward")
    return List[Tensor[dtype, 3]]()

# core/ops/gpu/embedding_gpu.mojo
#
# GPU token embedding (M3: delegates to the CPU kernel; the Metal path is
# wired in with the host GPU runtime - same note as matmul_gpu.mojo).

from ...device import has_metal_gpu
from ...tensor import Tensor
from ...utils import unimplemented
from ..cpu.embedding_cpu import (
    embedding_cpu,
    embedding_cpu_dynamic,
)


def _gpu_available[dtype: DType]() -> Bool:
    comptime if dtype == DType.float64:
        return False
    return has_metal_gpu()


def embedding_gpu[dtype: DType, hidden: Int, vocab: Int](
    tokens: Tensor[DType.int32, 1], table: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    _ = _gpu_available[dtype]()
    return embedding_cpu[dtype, hidden, vocab](tokens, table)


def embedding_gpu_dynamic[dtype: DType](
    tokens: Tensor[DType.int32, 1], table: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    _ = _gpu_available[dtype]()
    return embedding_cpu_dynamic[dtype](tokens, table)


def embedding_gpu_forward_with_saved[dtype: DType, hidden: Int, vocab: Int](
    tokens: Tensor[DType.int32, 1], table: Tensor[dtype, 2]
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]]:
    var out = embedding_gpu[dtype, hidden, vocab](tokens, table)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(table)
    return (out, saved^)


def embedding_gpu_backward[dtype: DType, hidden: Int, vocab: Int](
    grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]
) -> List[Tensor[dtype, 2]]:
    _ = grad_out
    _ = saved
    unimplemented("embedding_gpu_backward")
    return List[Tensor[dtype, 2]]()

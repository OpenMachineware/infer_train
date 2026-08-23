# core/ops/cpu/add_cpu.mojo
#
# Element-wise addition (residual connections) plus the rank-1 bias
# broadcast used by the QKV projections.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from ...simd_utils import W_F16, W_F32
from std.utils.static_tuple import StaticTuple


def _add_cpu_kernel[dtype: DType](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    if a.shape() != b.shape():
        unimplemented("add_cpu: shape mismatch")
    var out = tensor_zeros[dtype, 2](a.shape())
    var n = a.numel()
    comptime W = W_F16 if dtype == DType.float16 else W_F32
    var n_main = (n // W) * W
    var i = 0
    while i < n_main:
        var av = a.data().unsafe_load[width=W](offset=i)
        var bv = b.data().unsafe_load[width=W](offset=i)
        out.data().unsafe_store(i, av + bv)
        i += W
    while i < n:
        out.set(i, a.get(i) + b.get(i))
        i += 1
    return out


def _add_row_cpu_kernel[dtype: DType](
    x: Tensor[dtype, 2], bias: Tensor[dtype, 1]
) -> Tensor[dtype, 2]:
    """out[i, j] = x[i, j] + bias[j]."""
    var rows = x.shape()[0]
    var cols = x.shape()[1]
    if bias.shape()[0] != cols:
        unimplemented("add_row_cpu: bias length mismatch")
    var out = tensor_zeros[dtype, 2](x.shape())
    comptime W = W_F16 if dtype == DType.float16 else W_F32
    var cols_main = (cols // W) * W
    for i in range(rows):
        var base = i * cols
        var j = 0
        while j < cols_main:
            var xv = x.data().unsafe_load[width=W](offset=base + j)
            var bv = bias.data().unsafe_load[width=W](offset=j)
            out.data().unsafe_store(base + j, xv + bv)
            j += W
        while j < cols:
            out.set(base + j, x.get(base + j) + bias.get(j))
            j += 1
    return out


def add_cpu[dtype: DType, rows: Int, cols: Int](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    """Comptime-shaped element-wise add."""
    if a.shape() != StaticTuple[Int, 2](rows, cols):
        unimplemented("add_cpu: static shape mismatch")
    return _add_cpu_kernel[dtype](a, b)


def add_cpu_dynamic[dtype: DType](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tensor[dtype, 2]:
    """Runtime-shaped element-wise add."""
    return _add_cpu_kernel[dtype](a, b)


def add_row_cpu[dtype: DType](
    x: Tensor[dtype, 2], bias: Tensor[dtype, 1]
) -> Tensor[dtype, 2]:
    """Broadcast a rank-1 bias across the rows of `x`."""
    return _add_row_cpu_kernel[dtype](x, bias)


def add_cpu_forward_with_saved[dtype: DType, rows: Int, cols: Int](
    a: Tensor[dtype, 2], b: Tensor[dtype, 2]
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]]:
    var out = add_cpu[dtype, rows, cols](a, b)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(a)
    saved.append(b)
    return (out, saved^)


def add_cpu_backward[dtype: DType, rows: Int, cols: Int](
    grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]
) -> List[Tensor[dtype, 2]]:
    """Backward for element-wise add: both inputs get a copy of grad_out.

    `saved` = [a, b] (unused here - add is linear with identity Jacobian).
    """
    _ = saved
    var grad_a = tensor_zeros[dtype, 2](grad_out.shape())
    var grad_b = tensor_zeros[dtype, 2](grad_out.shape())
    var n = grad_out.numel()
    comptime W = W_F16 if dtype == DType.float16 else W_F32
    var n_main = (n // W) * W
    var i = 0
    while i < n_main:
        var v = grad_out.data().unsafe_load[width=W](offset=i)
        grad_a.data().unsafe_store(i, v)
        grad_b.data().unsafe_store(i, v)
        i += W
    while i < n:
        var v = grad_out.get(i)
        grad_a.set(i, v)
        grad_b.set(i, v)
        i += 1
    var result = List[Tensor[dtype, 2]]()
    result.append(grad_a)
    result.append(grad_b)
    return result^


def add_row_cpu_backward[dtype: DType](
    grad_out: Tensor[dtype, 2], bias: Tensor[dtype, 1]
) -> Tuple[Tensor[dtype, 2], Tensor[dtype, 1]]:
    """Backward for `add_row_cpu` (row broadcast bias add).

    grad_x = grad_out; grad_bias[j] = sum_i grad_out[i, j].
    """
    var rows = grad_out.shape()[0]
    var cols = grad_out.shape()[1]
    var grad_x = tensor_zeros[dtype, 2](grad_out.shape())
    var grad_bias = tensor_zeros[dtype, 1](bias.shape())
    for i in range(rows):
        for j in range(cols):
            var v = grad_out.get(i * cols + j)
            grad_x.set(i * cols + j, v)
            grad_bias.set(
                j, Scalar[dtype](Float32(grad_bias.get(j)) + Float32(v))
            )
    return (grad_x, grad_bias)

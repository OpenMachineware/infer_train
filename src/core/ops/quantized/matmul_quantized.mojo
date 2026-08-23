# core/ops/quantized/matmul_quantized.mojo
#
# Quantized matrix multiplication (CPU + GPU).
#
# The strategy (format / granularity / group size / symmetry) is comptime; the
# scale and zero point are runtime tensors.  Both backends dequantize the
# weight and then call the dense matmul.  (A fused int4-space GEMM is a later
# milestone; see the GPU comment for the API hook point.)
#
# Note: `format` is included here even though the original sketch omitted it,
# because `dequantize` requires it to select the quantization range.

from ...quantization import (
    QuantFormat,
    QuantGranularity,
    dequantize,
)
from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from ..cpu.matmul_cpu import matmul_cpu_dynamic
from ..gpu.matmul_gpu import matmul_gpu_dynamic


def _cast[dst: DType, src: DType](
    tensor: Tensor[src, 2]
) -> Tensor[dst, 2]:
    """Elementwise dtype cast (used when weight dtype != activation dtype)."""
    var out = tensor_zeros[dst, 2](tensor.shape())
    for i in range(tensor.numel()):
        out.set(i, Scalar[dst](Float32(tensor.get(i))))
    return out


def matmul_quantized_cpu[
    dtype: DType,
    weight_type: DType,
    format: QuantFormat,
    granularity: QuantGranularity,
    group_size: Int,
    is_symmetric: Bool,
](
    a: Tensor[dtype, 2],
    b_quant: Tensor[weight_type, 2],
    scale: Tensor[weight_type, 1],
    zero_point: Optional[Tensor[weight_type, 1]] = None,
) -> Tensor[dtype, 2]:
    """Dequantize the weight, then run the dense CPU matmul."""
    var b_deq = dequantize[
        weight_type, 2, format, granularity, group_size, is_symmetric
    ](b_quant, scale, zero_point)
    var b_cast = _cast[dtype, weight_type](b_deq)
    return matmul_cpu_dynamic[dtype](a, b_cast)


def matmul_quantized_gpu[
    dtype: DType,
    weight_type: DType,
    format: QuantFormat,
    granularity: QuantGranularity,
    group_size: Int,
    is_symmetric: Bool,
](
    a: Tensor[dtype, 2],
    b_quant: Tensor[weight_type, 2],
    scale: Tensor[weight_type, 1],
    zero_point: Optional[Tensor[weight_type, 1]] = None,
) -> Tensor[dtype, 2]:
    """GPU quantized matmul.

    If `std.gpu` later exposes a `matmul_gpu_qint4_impl`-style API, compute in
    quantized space here; otherwise dequantize and run the dense GPU matmul.
    M1 delegates to the CPU path via `matmul_gpu_dynamic`.
    """
    var b_deq = dequantize[
        weight_type, 2, format, granularity, group_size, is_symmetric
    ](b_quant, scale, zero_point)
    var b_cast = _cast[dtype, weight_type](b_deq)
    return matmul_gpu_dynamic[dtype](a, b_cast)


def matmul_quantized_forward_with_saved[
    dtype: DType,
    weight_type: DType,
    format: QuantFormat,
    granularity: QuantGranularity,
    group_size: Int,
    is_symmetric: Bool,
](
    a: Tensor[dtype, 2],
    b_quant: Tensor[weight_type, 2],
    scale: Tensor[weight_type, 1],
    zero_point: Optional[Tensor[weight_type, 1]] = None,
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]]:
    var out = matmul_quantized_cpu[
        dtype, weight_type, format, granularity, group_size, is_symmetric
    ](a, b_quant, scale, zero_point)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(a)
    return (out, saved^)


def matmul_quantized_backward[
    dtype: DType,
    weight_type: DType,
    format: QuantFormat,
    granularity: QuantGranularity,
    group_size: Int,
    is_symmetric: Bool,
](
    grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]
) -> List[Tensor[dtype, 2]]:
    _ = grad_out
    _ = saved
    _ = format
    _ = granularity
    _ = group_size
    _ = is_symmetric
    _ = weight_type
    unimplemented("matmul_quantized_backward")
    return List[Tensor[dtype, 2]]()

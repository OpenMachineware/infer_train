# core/ops/quantized/rms_norm_quantized.mojo
#
# Quantized RMSNorm (CPU + GPU): dequantize the input, then normalize.

from ...quantization import (
    QuantFormat,
    QuantGranularity,
    dequantize,
)
from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from ..cpu.rms_norm_cpu import rms_norm_cpu_dynamic
from ..gpu.rms_norm_gpu import rms_norm_gpu_dynamic


def rms_norm_quantized_cpu[
    dtype: DType,
    dim: Int,
    format: QuantFormat,
    granularity: QuantGranularity,
    group_size: Int,
    is_symmetric: Bool,
](
    x_quant: Tensor[dtype, 2],
    scale: Tensor[dtype, 1],
    zero_point: Optional[Tensor[dtype, 1]] = None,
    eps: Float32 = Float32(1e-5),
) -> Tensor[dtype, 2]:
    _ = dim
    var x_deq = dequantize[
        dtype, 2, format, granularity, group_size, is_symmetric
    ](x_quant, scale, zero_point)
    return rms_norm_cpu_dynamic[dtype](x_deq, eps)


def rms_norm_quantized_gpu[
    dtype: DType,
    dim: Int,
    format: QuantFormat,
    granularity: QuantGranularity,
    group_size: Int,
    is_symmetric: Bool,
](
    x_quant: Tensor[dtype, 2],
    scale: Tensor[dtype, 1],
    zero_point: Optional[Tensor[dtype, 1]] = None,
    eps: Float32 = Float32(1e-5),
) -> Tensor[dtype, 2]:
    _ = dim
    var x_deq = dequantize[
        dtype, 2, format, granularity, group_size, is_symmetric
    ](x_quant, scale, zero_point)
    return rms_norm_gpu_dynamic[dtype](x_deq, eps)


def rms_norm_quantized_forward_with_saved[
    dtype: DType,
    dim: Int,
    format: QuantFormat,
    granularity: QuantGranularity,
    group_size: Int,
    is_symmetric: Bool,
](
    x_quant: Tensor[dtype, 2],
    scale: Tensor[dtype, 1],
    zero_point: Optional[Tensor[dtype, 1]] = None,
    eps: Float32 = Float32(1e-5),
) -> Tuple[Tensor[dtype, 2], List[Tensor[dtype, 2]]]:
    var out = rms_norm_quantized_cpu[
        dtype, dim, format, granularity, group_size, is_symmetric
    ](x_quant, scale, zero_point, eps)
    var saved = List[Tensor[dtype, 2]]()
    saved.append(x_quant)
    return (out, saved^)


def rms_norm_quantized_backward[
    dtype: DType,
    dim: Int,
    format: QuantFormat,
    granularity: QuantGranularity,
    group_size: Int,
    is_symmetric: Bool,
](grad_out: Tensor[dtype, 2], saved: List[Tensor[dtype, 2]]) -> List[
    Tensor[dtype, 2]
]:
    _ = grad_out
    _ = saved
    _ = format
    _ = granularity
    _ = group_size
    _ = is_symmetric
    unimplemented("rms_norm_quantized_backward")
    return List[Tensor[dtype, 2]]()

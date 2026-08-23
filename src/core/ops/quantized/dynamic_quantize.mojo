# core/ops/quantized/dynamic_quantize.mojo
#
# M6 Phase 7: dynamic quantization operators (the quantize/dequantize pair
# the training path can use for quantization-aware training).
#
#   * symmetric:   scale = max(|x|) / (2^(bits-1) - 1), zero_point = 0
#   * asymmetric:  scale = (max - min) / (2^bits - 1), zero_point = min
#
# The quantized payload uses `Tensor[DType.int32, 2]` storage (the engine's
# ABI dtype codes only cover f32/f16/i32, see infer_train_bindings.mojo);
# each element holds one int8 code in [-128, 127].

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.utils.static_tuple import StaticTuple


def _round_i(v: Float32) -> Int:
    """Round half away from zero to Int."""
    if v >= Float32(0.0):
        return Int(v + Float32(0.5))
    return Int(v - Float32(0.5))


def dynamic_quantize_symmetric[dtype: DType](
    x: Tensor[dtype, 2], bits: Int = 8
) -> Tuple[Tensor[DType.int32, 2], Float32]:
    """Symmetric quantization: scale = max(|x|) / (2^(bits-1) - 1)."""
    var numel = x.numel()
    var mx = Float32(0)
    for i in range(numel):
        var v = Float32(x.get(i))
        if v < 0:
            v = -v
        if v > mx:
            mx = v
    var qmax = 1
    for i in range(bits - 1):
        qmax = qmax * 2
    qmax -= 1
    var scale = mx / Float32(qmax)
    if scale < Float32(1e-12):
        scale = Float32(1.0)
    var q = tensor_zeros[DType.int32, 2](x.shape())
    for i in range(numel):
        var v = Float32(x.get(i)) / scale
        var qv = _round_i(v)
        if qv > qmax:
            qv = qmax
        if qv < -qmax:
            qv = -qmax
        q.set(i, Scalar[DType.int32](qv))
    return (q, scale)


def dynamic_quantize_asymmetric[dtype: DType](
    x: Tensor[dtype, 2], bits: Int = 8
) -> Tuple[Tensor[DType.int32, 2], Float32, Float32]:
    """Asymmetric quantization: scale = (max-min)/(2^bits-1), zero_point
    = min (quantized code 0 maps to the minimum)."""
    var numel = x.numel()
    var mn = Float32(0)
    var mx = Float32(0)
    if numel > 0:
        mn = Float32(x.get(0))
        mx = mn
    for i in range(numel):
        var v = Float32(x.get(i))
        if v > mx:
            mx = v
        if v < mn:
            mn = v
    var qmax = 1
    for i in range(bits):
        qmax = qmax * 2
    qmax -= 1
    var scale = (mx - mn) / Float32(qmax)
    if scale < Float32(1e-12):
        scale = Float32(1.0)
    var q = tensor_zeros[DType.int32, 2](x.shape())
    for i in range(numel):
        var qv = _round_i((Float32(x.get(i)) - mn) / scale) - 128
        if qv > 127:
            qv = 127
        if qv < -128:
            qv = -128
        q.set(i, Scalar[DType.int32](qv))
    return (q, scale, mn)


def dynamic_dequantize(
    q: Tensor[DType.int32, 2], scale: Float32, zero_point: Float32
) -> Tensor[DType.float32, 2]:
    """x = scale * (q + 128) + zero_point (matches the asymmetric int8
    code range; symmetric quantization has zero_point == 0)."""
    var numel = q.numel()
    var out = tensor_zeros[DType.float32, 2](q.shape())
    for i in range(numel):
        out.set(
            i,
            Scalar[DType.float32](
                scale * Float32(Int(q.get(i)) + 128) + zero_point
            ),
        )
    return out

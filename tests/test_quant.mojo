from src.core.quantization import (
    QuantFormat, QuantGranularity, quantize_dynamic, dequantize
)
from src.core.tensor import Tensor, tensor_zeros
from std.utils.static_tuple import StaticTuple

def main():
    var x = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](2, 4))
    for i in range(8):
        x.set(i, Scalar[DType.float32](Float32(i) - 3.0))
    var (q, info) = quantize_dynamic[
        DType.float32, 2, QuantFormat.Q8_0,
        QuantGranularity.PerTensor, 0, True
    ](x)
    print("q numel:", q.numel(), "scale len:", info.scale.numel())
    var xr = dequantize[
        DType.float32, 2, QuantFormat.Q8_0,
        QuantGranularity.PerTensor, 0, True
    ](q, info.scale, info.zero_point)
    print("reconstructed[0]:", xr.get(0), "orig:", x.get(0))
    print("OK")

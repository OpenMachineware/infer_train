# tests/test_requantize.mojo
#
# M7 re-quantization: fp16 -> Q4_K / Q8_0 / NF4, round-trip error bounds,
# and block-level dequant consistency.

from src.core.tensor import tensor_zeros, Tensor
from src.core.ops.quantized.requantize import (
    requantize,
    requantize_q8_0,
    requantize_q4_k,
    requantize_nf4,
    QuantizedWeights,
)
from src.core.ops.quantized.dequantize import dequantize_into
from std.utils.static_tuple import StaticTuple
from std.math import sin


def main():
    # build a smooth weight-like signal: 512 fp16 values in [-1, 1]
    var n = 512
    var src = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](n))
    for i in range(n):
        var v = sin(Float32(i) * Float32(0.13)) * Float32(0.8)
        src.set(i, Scalar[DType.float16](v))

    # round-trip through each format (quantize -> dequantize -> compare)
    var q8 = requantize_q8_0(src, n)
    check(q8.ggml_type == 8 and len(q8.data) == (n // 32) * 34, "q8_0 layout")
    check_roundtrip(src, q8, Float32(0.02), "q8_0")

    var q4 = requantize_q4_k(src, n)
    check(
        q4.ggml_type == 12 and len(q4.data) == (n // 256) * 144, "q4_k layout"
    )
    check_roundtrip(src, q4, Float32(0.10), "q4_k")

    var nf = requantize_nf4(src, n)
    check(nf.ggml_type == 30 and len(nf.data) == (n // 64) * 34, "nf4 layout")
    check_roundtrip(src, nf, Float32(0.16), "nf4")

    # dispatcher

    var via = requantize(src, n, "Q4_K_M")
    check(via.ggml_type == 12, "dispatcher Q4_K_M")

    var via2 = requantize(src, n, "NF4")
    check(via2.ggml_type == 30, "dispatcher NF4")
    print("test_requantize OK")


def check_roundtrip(
    src: Tensor[DType.float16, 1],
    quant: QuantizedWeights,
    tol: Float32,
    name: String,
):
    var n = src.numel()
    var dst = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, n))
    # serialize the byte list into a buffer for dequantize_into
    var buf = tensor_zeros[DType.uint8, 1](StaticTuple[Int, 1](len(quant.data)))
    for i in range(len(quant.data)):
        buf.set(i, Scalar[DType.uint8](quant.data[i]))

    dequantize_into(
        quant.ggml_type, buf.data().unsafe_bitcast[UInt8](), 0, dst, n
    )

    var max_err = Float32(0)
    for i in range(n):
        var err = Float32(dst.get(i)) - Float32(src.get(i))
        if err < 0:
            err = -err
        if err > max_err:
            max_err = err

    check(
        max_err < tol,
        name + " roundtrip tolerance (max " + String(max_err) + ")",
    )


def check(cond: Bool, name: String):
    if not cond:
        print("FAIL:", name)
        abort()


def abort():
    from std.os.os import abort as _abort

    _abort()

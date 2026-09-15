# Debug output projection
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf, find_tensor
from src.core.transformer import load_config, TransformerModel
from src.core.tensor import tensor_zeros
from std.utils.static_tuple import StaticTuple


def check_output_proj() raises:
    """Compare output projection from quantized vs dequantized."""
    var model_path = "Hy-MT2-7B-Q4_K_M.gguf"

    # Load quantized model
    print("Loading quantized model...")
    var ctx1 = load_gguf(model_path)
    var config = load_config(ctx1)
    var m1 = TransformerModel(config, ctx1^, 512, quant_resident=True)

    # Load dequantized model
    print("Loading dequantized model...")
    var ctx2 = load_gguf(model_path)
    var config2 = load_config(ctx2)
    var m2 = TransformerModel(config2, ctx2^, 512, quant_resident=False)

    # Get hidden state (use a fixed input)
    var hidden = config.hidden
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, hidden))
    for d in range(hidden):
        var val = Float16(0.01) * Float16(d % 100)
        x.set(d, Scalar[DType.float16](val))

    # Run output projection
    print("Running output projection...")
    var out1 = m1._output_proj(x)
    var out2 = m2._output_proj(x)

    var vocab = out1.shape()[1]
    print("Vocab size:", vocab)
    print("Output shape:", out1.shape()[0], out1.shape()[1])

    # Compare
    var max_diff = Float32(0.0)
    var max_idx = 0
    for i in range(min(100, vocab)):
        var v1 = Float32(out1.get(i))
        var v2 = Float32(out2.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
            max_idx = i
        if diff > 1.0:
            print("Logit", i, "quant:", v1, "dequant:", v2, "diff:", diff)

    print("Max diff:", max_diff, "at index", max_idx)


def main() raises:
    check_output_proj()
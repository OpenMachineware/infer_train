# Debug first layer processing
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf, find_tensor
from src.core.transformer import load_config, TransformerModel
from src.core.tensor import tensor_zeros, Tensor
from std.utils.static_tuple import StaticTuple


def compare_layer() raises:
    """Compare first layer processing between quantized and dequantized."""
    var model_path = "Hy-MT2-7B-Q4_K_M.gguf"

    # Load quantized model
    print("Loading quantized model...")
    var ctx1 = load_gguf(model_path)
    var config1 = load_config(ctx1)
    var m1 = TransformerModel(config1, ctx1^, 512, quant_resident=True)

    # Load dequantized model
    print("Loading dequantized model...")
    var ctx2 = load_gguf(model_path)
    var config2 = load_config(ctx2)
    var m2 = TransformerModel(config2, ctx2^, 512, quant_resident=False)

    # Get embedding
    var token = 100
    var emb1 = m1.embed(token)
    var emb2 = m2.embed(token)

    print("Embedding shape:", emb1.shape()[0], emb1.shape()[1])

    # Run first layer
    print("\nRunning first layer...")
    var x1 = m1._layer_forward_attn(0, emb1, 0)
    var x2 = m2._layer_forward_attn(0, emb2, 0)

    print("After first layer:")
    var max_diff = Float32(0.0)
    var max_idx = 0
    for i in range(min(100, config1.hidden)):
        var v1 = Float32(x1.get(i))
        var v2 = Float32(x2.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
            max_idx = i
        if diff > 0.5:
            print("  x[", i, "]: quant=", v1, " dequant=", v2, " diff=", diff)
    print("  Max diff:", max_diff, "at index", max_idx)


def main() raises:
    compare_layer()

# Debug layer values
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config, TransformerModel
from std.utils.static_tuple import StaticTuple


def compare_layer_values() raises:
    """Compare actual values after each layer."""
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

    print("Embedding check:")
    for i in range(5):
        var v1 = Float32(emb1.get(i))
        var v2 = Float32(emb2.get(i))
        print("  emb[", i, "]: quant=", v1, " dequant=", v2)

    # Run all layers
    var x1 = emb1
    var x2 = emb2

    for layer in range(config1.n_layers):
        x1 = m1._layer_forward_attn(layer, x1, 0)
        x2 = m2._layer_forward_attn(layer, x2, 0)

    print("\nFinal layer output (first 20 values):")
    for i in range(20):
        var v1 = Float32(x1.get(i))
        var v2 = Float32(x2.get(i))
        var diff = abs(v1 - v2)
        print("  x[", i, "]: quant=", v1, " dequant=", v2, " diff=", diff)

    # Also run forward_hidden for comparison
    print("\nForward_hidden output (first 20 values):")
    var h1 = m1.forward_hidden(token, 0)
    var h2 = m2.forward_hidden(token, 0)
    for i in range(20):
        var v1 = Float32(h1.get(i))
        var v2 = Float32(h2.get(i))
        var diff = abs(v1 - v2)
        print("  h[", i, "]: quant=", v1, " dequant=", v2, " diff=", diff)


def main() raises:
    compare_layer_values()

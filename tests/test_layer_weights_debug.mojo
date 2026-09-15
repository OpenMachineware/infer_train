# Debug layer weights comparison
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config, TransformerModel
from std.utils.static_tuple import StaticTuple


def check_layer_weights() raises:
    """Compare first layer weights between paths."""
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

    # Get layer 0 views
    var lv1 = m1.layer_view(0)
    var lv2 = m2.layer_view(0)

    # Check attention norm weights
    print("\nLayer 0 attn_norm_w comparison:")
    var max_diff = Float32(0.0)
    for i in range(min(20, config1.hidden)):
        var v1 = Float32(lv1.attn_norm_w.get(i))
        var v2 = Float32(lv2.attn_norm_w.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
        if diff > 0.001:
            print("  norm[", i, "]: quant=", v1, " dequant=", v2, " diff=", diff)
    print("Max diff:", max_diff)

    # Check Q weight
    print("\nLayer 0 Q weight comparison:")
    # For quantized: check if it's quantized
    print("  Quantized Q weight is quantized:", lv1.q_w.quantized)
    print("  Dequantized Q weight is quantized:", lv2.q_w.quantized)


def main() raises:
    check_layer_weights()

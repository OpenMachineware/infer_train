# Debug attention computation
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config, TransformerModel, rms_norm_weight
from std.utils.static_tuple import StaticTuple


def check_attention() raises:
    """Compare attention output between paths."""
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

    # Apply first layer norm
    var lv1 = m1.layer_view(0)
    var lv2 = m2.layer_view(0)

    var norm1 = rms_norm_weight[DType.float16](emb1, lv1.attn_norm_w, config1.norm_eps)
    var norm2 = rms_norm_weight[DType.float16](emb2, lv2.attn_norm_w, config2.norm_eps)

    print("After first layer norm (first 5 values):")
    for i in range(5):
        var v1 = Float32(norm1.get(i))
        var v2 = Float32(norm2.get(i))
        print("  norm[", i, "]: quant=", v1, " dequant=", v2, " diff=", abs(v1 - v2))

    # Get the MHA result
    # Note: _layer_forward_attn also applies ffn, so let's just check the raw attention
    # Actually, let me just check one full layer and see the intermediate values

    print("\nRunning layer 0 forward...")
    var x1 = m1._layer_forward_attn(0, emb1, 0)
    var x2 = m2._layer_forward_attn(0, emb2, 0)

    print("After layer 0 (first 10 values):")
    for i in range(10):
        var v1 = Float32(x1.get(i))
        var v2 = Float32(x2.get(i))
        print("  x[", i, "]: quant=", v1, " dequant=", v2, " diff=", abs(v1 - v2))


def main() raises:
    check_attention()

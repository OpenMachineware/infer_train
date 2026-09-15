# Debug token 100 embedding
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config, TransformerModel
from std.utils.static_tuple import StaticTuple


def check_token_embedding() raises:
    """Check embedding for token 100."""
    var model_path = "Hy-MT2-7B-Q4_K_M.gguf"
    
    # Load quantized model
    print("Loading quantized model...")
    var ctx1 = load_gguf(model_path)
    var config1 = load_config(ctx1)
    var m1 = TransformerModel(config1, ctx1^, 512)
    
    # Load dequantized model
    print("Loading dequantized model...")
    var ctx2 = load_gguf(model_path)
    var config2 = load_config(ctx2)
    var m2 = TransformerModel(config2, ctx2^, 512, quant_resident=False)
    
    # Get embedding for token 100
    var emb1 = m1.embed(100)
    var emb2 = m2.embed(100)
    
    print("Token 100 embedding (first 20 values):")
    for i in range(20):
        var v1 = Float32(emb1.get(i))
        var v2 = Float32(emb2.get(i))
        var diff = abs(v1 - v2)
        print("  emb[", i, "]: quant=", v1, " dequant=", v2, " diff=", diff)
    
    # Check if all embedding values match
    var max_diff = Float32(0.0)
    var hidden = config1.hidden
    for i in range(hidden):
        var v1 = Float32(emb1.get(i))
        var v2 = Float32(emb2.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
    print("Max embedding diff:", max_diff)


def main() raises:
    check_token_embedding()

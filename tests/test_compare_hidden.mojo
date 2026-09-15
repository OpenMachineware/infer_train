# Compare hidden states from different methods
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config, TransformerModel
from std.utils.static_tuple import StaticTuple


def compare_hidden() raises:
    """Compare hidden states from forward_hidden vs manual layer processing."""
    var model_path = "Hy-MT2-7B-Q4_K_M.gguf"
    
    # Load quantized model
    print("Loading quantized model...")
    var ctx1 = load_gguf(model_path)
    var config1 = load_config(ctx1)
    var m1 = TransformerModel(config1, ctx1^, 512, quant_resident=True)
    
    # Get embedding
    var token = 100
    var emb1 = m1.embed(token)
    
    # Method 1: Manual layer processing (like test_all_layers_debug)
    print("\nMethod 1: Manual layer processing")
    var x1_manual = emb1
    for layer in range(config1.n_layers):
        x1_manual = m1._layer_forward_attn(layer, x1_manual, 0)
    
    # Method 2: forward_hidden
    print("Method 2: forward_hidden")
    var x1_forward = m1.forward_hidden(token, 0)
    
    # Compare
    print("\nComparing manual vs forward_hidden for quantized model:")
    var max_diff = Float32(0.0)
    for i in range(min(20, config1.hidden)):
        var v1 = Float32(x1_manual.get(i))
        var v2 = Float32(x1_forward.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
        if diff > 0.001:
            print("  x[", i, "]: manual=", v1, " forward=", v2, " diff=", diff)
    print("Max diff:", max_diff)


def main() raises:
    compare_hidden()

# Debug output norm weights
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf, find_tensor
from src.core.transformer import load_config, TransformerModel
from std.utils.static_tuple import StaticTuple


def check_output_norm() raises:
    """Compare output norm weights between paths."""
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
    
    # Get output norm weights
    var w1 = m1._output_norm_w()
    var w2 = m2._output_norm_w()
    
    print("Output norm shape:", w1.shape()[0])
    
    # Compare weights
    var max_diff = Float32(0.0)
    var max_idx = 0
    for i in range(min(50, config1.hidden)):
        var v1 = Float32(w1.get(i))
        var v2 = Float32(w2.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
            max_idx = i
        if diff > 0.001:
            print("  w[", i, "]: quant=", v1, " dequant=", v2, " diff=", diff)
    print("Max diff:", max_diff, "at index", max_idx)


def main() raises:
    check_output_norm()

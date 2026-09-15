# Debug forward mismatch
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import TransformerModel, load_config
from src.core.tensor import Tensor
from std.utils.static_tuple import StaticTuple


def check_forward_match_debug() raises:
    """Debug the forward mismatch."""
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
    
    # Run forward
    print("Running forward...")
    var logits1 = m1.forward(100, 0)
    var logits2 = m2.forward(100, 0)
    
    # Find max diff
    var max_diff = Float32(0.0)
    var max_idx = 0
    var vocab = logits1.shape()[0]
    
    for i in range(min(100, vocab)):  # Check first 100 logits
        var v1 = Float32(logits1.get(i))
        var v2 = Float32(logits2.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
            max_idx = i
        if diff > 1.0:
            print("Logit", i, "quant:", v1, "dequant:", v2, "diff:", diff)
    
    print("Max diff:", max_diff, "at index", max_idx)
    print("Sample logits:")
    print("  quant[0]:", Float32(logits1.get(0)), "dequant[0]:", Float32(logits2.get(0)))
    print("  quant[1]:", Float32(logits1.get(1)), "dequant[1]:", Float32(logits2.get(1)))
    print("  quant[2]:", Float32(logits1.get(2)), "dequant[2]:", Float32(logits2.get(2)))


def main() raises:
    check_forward_match_debug()
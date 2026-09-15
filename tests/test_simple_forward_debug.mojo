# Simple forward debug
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config, TransformerModel
from std.utils.static_tuple import StaticTuple


def check_simple_forward() raises:
    """Simple single-token forward comparison."""
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
    
    # Single forward call
    print("Running single forward...")
    var logits1 = m1.forward(100, 0)
    var logits2 = m2.forward(100, 0)
    
    # Compare
    var max_diff = Float32(0.0)
    var max_idx = 0
    var vocab = config1.vocab
    
    for i in range(vocab):
        var v1 = Float32(logits1.get(i))
        var v2 = Float32(logits2.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
            max_idx = i
    
    print("Max diff:", max_diff, "at index", max_idx)
    print("logits1[", max_idx, "]=", Float32(logits1.get(max_idx)))
    print("logits2[", max_idx, "]=", Float32(logits2.get(max_idx)))


def main() raises:
    check_simple_forward()

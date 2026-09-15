# Debug hidden state and output projection
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf, find_tensor
from src.core.transformer import load_config, TransformerModel, rms_norm_weight
from src.core.tensor import tensor_zeros, Tensor
from std.utils.static_tuple import StaticTuple


def compare_hidden_proj() raises:
    """Compare hidden state + output projection between paths."""
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
    
    # Get hidden states through forward_hidden
    var token = 100
    print("Getting hidden states...")
    var h1 = m1.forward_hidden(token, 0)
    var h2 = m2.forward_hidden(token, 0)
    
    print("Hidden state shape:", h1.shape()[0], h1.shape()[1])
    
    # Compare hidden states
    var max_diff = Float32(0.0)
    for i in range(min(20, config1.hidden)):
        var v1 = Float32(h1.get(i))
        var v2 = Float32(h2.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
        if diff > 0.5:
            print("  h[", i, "]: quant=", v1, " dequant=", v2, " diff=", diff)
    print("Hidden max diff:", max_diff)
    
    # Apply output norm
    var norm1 = rms_norm_weight[DType.float16](h1, m1._output_norm_w(), config1.norm_eps)
    var norm2 = rms_norm_weight[DType.float16](h2, m2._output_norm_w(), config2.norm_eps)
    
    print("\nAfter output norm:")
    max_diff = Float32(0.0)
    for i in range(min(20, config1.hidden)):
        var v1 = Float32(norm1.get(i))
        var v2 = Float32(norm2.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
        if diff > 0.5:
            print("  norm[", i, "]: quant=", v1, " dequant=", v2, " diff=", diff)
    print("Norm max diff:", max_diff)
    
    # Apply output projection
    print("\nApplying output projection...")
    var logits1 = m1._output_proj(norm1)
    var logits2 = m2._output_proj(norm2)
    
    print("Logits shape:", logits1.shape()[0], logits1.shape()[1])
    
    # Compare logits
    max_diff = Float32(0.0)
    var max_idx = 0
    for i in range(min(50, config1.vocab)):
        var v1 = Float32(logits1.get(i))
        var v2 = Float32(logits2.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
            max_idx = i
        if diff > 2.0:
            print("  logit[", i, "]: quant=", v1, " dequant=", v2, " diff=", diff)
    print("Logits max diff:", max_diff, "at index", max_idx)


def main() raises:
    compare_hidden_proj()

# Analyze logit differences
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config, TransformerModel
from std.utils.static_tuple import StaticTuple
from std.math import sqrt


def analyze_logits() raises:
    """Analyze logit differences across vocab."""
    var model_path = "Hy-MT2-7B-Q4_K_M.gguf"
    
    # Load models
    print("Loading models...")
    var ctx1 = load_gguf(model_path)
    var config1 = load_config(ctx1)
    var m1 = TransformerModel(config1, ctx1^, 512)
    
    var ctx2 = load_gguf(model_path)
    var config2 = load_config(ctx2)
    var m2 = TransformerModel(config2, ctx2^, 512, quant_resident=False)
    
    # Run forward
    var logits1 = m1.forward(100, 0)
    var logits2 = m2.forward(100, 0)
    
    var vocab = config1.vocab
    
    # Count differences by magnitude
    var count_0_1 = 0
    var count_1_5 = 0
    var count_5_10 = 0
    var count_10_plus = 0
    var max_diff = Float32(0.0)
    var max_idx = 0
    
    for i in range(vocab):
        var v1 = Float32(logits1.get(i))
        var v2 = Float32(logits2.get(i))
        var diff = abs(v1 - v2)
        
        if diff > max_diff:
            max_diff = diff
            max_idx = i
        
        if diff < 1.0:
            count_0_1 += 1
        elif diff < 5.0:
            count_1_5 += 1
        elif diff < 10.0:
            count_5_10 += 1
        else:
            count_10_plus += 1
    
    print("\nLogit diff distribution:")
    print("  < 1.0:", count_0_1)
    print("  1-5:", count_1_5)
    print("  5-10:", count_5_10)
    print("  10+:", count_10_plus)
    print("Max diff:", max_diff, "at index", max_idx)
    print("Total vocab:", vocab)
    
    # Check correlation between the two sets
    var sum1 = Float32(0.0)
    var sum2 = Float32(0.0)
    var sum_sq1 = Float32(0.0)
    var sum_sq2 = Float32(0.0)
    var sum_prod = Float32(0.0)
    
    for i in range(vocab):
        var v1 = Float32(logits1.get(i))
        var v2 = Float32(logits2.get(i))
        sum1 = sum1 + v1
        sum2 = sum2 + v2
        sum_sq1 = sum_sq1 + v1 * v1
        sum_sq2 = sum_sq2 + v2 * v2
        sum_prod = sum_prod + v1 * v2
    
    var mean1 = sum1 / Float32(vocab)
    var mean2 = sum2 / Float32(vocab)
    var var1 = sum_sq1 / Float32(vocab) - mean1 * mean1
    var var2 = sum_sq2 / Float32(vocab) - mean2 * mean2
    var covar = sum_prod / Float32(vocab) - mean1 * mean2
    var corr = covar / (sqrt(var1) * sqrt(var2))
    
    print("\nLogit statistics:")
    print("  Mean - quant:", mean1, " dequant:", mean2)
    print("  Std - quant:", sqrt(var1), " dequant:", sqrt(var2))
    print("  Correlation:", corr)


def main() raises:
    analyze_logits()

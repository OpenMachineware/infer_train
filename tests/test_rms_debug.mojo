# Debug RMS values
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config, TransformerModel
from std.utils.static_tuple import StaticTuple
from std.math import sqrt


def check_rms() raises:
    """Compare RMS values of hidden states."""
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
    
    # Get hidden states
    var token = 100
    var h1 = m1.forward_hidden(token, 0)
    var h2 = m2.forward_hidden(token, 0)
    
    # Compute RMS
    var sum1 = Float32(0.0)
    var sum2 = Float32(0.0)
    var hidden = config1.hidden
    
    for i in range(hidden):
        var v1 = Float32(h1.get(i))
        var v2 = Float32(h2.get(i))
        sum1 = sum1 + v1 * v1
        sum2 = sum2 + v2 * v2
    
    var rms1 = sqrt(sum1 / Float32(hidden))
    var rms2 = sqrt(sum2 / Float32(hidden))
    
    print("Hidden RMS - quant:", rms1, " dequant:", rms2, " diff:", abs(rms1 - rms2))
    
    # Check if normalization factor is very different
    var norm_factor1 = 1.0 / rms1
    var norm_factor2 = 1.0 / rms2
    
    print("Norm factor - quant:", norm_factor1, " dequant:", norm_factor2)
    
    # Check hidden values at a few indices
    print("\nSample hidden values:")
    for i in range(0, hidden, 500):
        var v1 = Float32(h1.get(i))
        var v2 = Float32(h2.get(i))
        print("  h[", i, "]: quant=", v1, " dequant=", v2, " diff=", abs(v1 - v2))


def main() raises:
    check_rms()

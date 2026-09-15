# Debug KV cache
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config, TransformerModel
from std.utils.static_tuple import StaticTuple
from std.math import sqrt


def check_cache() raises:
    """Check if KV cache is being used correctly."""
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

    # Run forward_hidden and check cache
    print("Running forward_hidden...")
    var h1 = m1.forward_hidden(100, 0)
    var h2 = m2.forward_hidden(100, 0)

    # Check if the issue is in how the cache is used
    # For position 0, there should be no cached values
    print("\nChecking first forward_hidden output:")
    print("Hidden shape:", h1.shape()[0], h1.shape()[1])

    # Check RMS
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

    print("RMS - quant:", rms1, " dequant:", rms2)

    # Check some specific values
    print("\nSample hidden values:")
    for i in [0, 100, 500, 1000, 2000, 3000]:
        var v1 = Float32(h1.get(i))
        var v2 = Float32(h2.get(i))
        print("  h[", i, "]: quant=", v1, " dequant=", v2, " diff=", abs(v1 - v2))


def main() raises:
    check_cache()

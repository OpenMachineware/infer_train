# Debug quantized vs dequantized forward

from src.core.gguf_loader import load_gguf
from src.core.transformer import TransformerModel, load_config
from src.core.tensor import Tensor
from std.utils.static_tuple import StaticTuple


def main() raises:
    # Test token
    var token = 100

    # Quantized-resident mode
    print("Loading model in quantized-resident mode...")
    var ctx1 = load_gguf("Qwen3-0.6B-UD-Q4_K_XL.gguf")
    var config1 = load_config(ctx1)
    var m1 = TransformerModel(config1, ctx1^, 512)
    var logits1 = m1.forward(token, 0)

    print("Quantized logits num dim:", logits1.rank, "numel:", logits1.numel())
    print("First 10 values:", logits1.get(0), logits1.get(1), logits1.get(2), logits1.get(3), logits1.get(4), logits1.get(5), logits1.get(6), logits1.get(7), logits1.get(8), logits1.get(9))

    # Dequantized mode
    print("\nLoading model in dequantized mode...")
    var ctx2 = load_gguf("Qwen3-0.6B-UD-Q4_K_XL.gguf")
    var config2 = load_config(ctx2)
    var m2 = TransformerModel(config2, ctx2^, 512, quant_resident=False)
    var logits2 = m2.forward(token, 0)

    print("Dequantized logits num dim:", logits2.rank, "numel:", logits2.numel())
    print("First 10 values:", logits2.get(0), logits2.get(1), logits2.get(2), logits2.get(3), logits2.get(4), logits2.get(5), logits2.get(6), logits2.get(7), logits2.get(8), logits2.get(9))

    # Compare
    var n = logits1.numel()
    var max_diff = Float32(0)
    var sum_diff = Float32(0)
    for i in range(n):
        var d = abs(Float32(logits1.get(i)) - Float32(logits2.get(i)))
        sum_diff += d
        if d > max_diff:
            max_diff = d

    print("\nComparison:")
    print("  Max diff:", max_diff)
    print("  Mean diff:", sum_diff / Float32(n))

# Compare token embedding between quantized and dequantized modes

from src.core.gguf_loader import load_gguf, GGUFTensor
from src.core.transformer import TransformerModel, load_config, embedding_row_quantized
from src.core.tensor import Tensor, tensor_zeros
from std.utils.static_tuple import StaticTuple
from std.math import abs


def main() raises:
    print("Comparing token embedding...")

    # Load model in quantized mode
    var ctx1 = load_gguf("Qwen3-0.6B-UD-Q4_K_XL.gguf")
    var config = load_config(ctx1)
    var m1 = TransformerModel(config, ctx1^, 512)

    # Load model in dequantized mode
    var ctx2 = load_gguf("Qwen3-0.6B-UD-Q4_K_XL.gguf")
    var m2 = TransformerModel(config, ctx2^, 512, quant_resident=False)

    var token = 100

    # Get token embedding from quantized mode
    var toks = Tensor[DType.int32, 1](StaticTuple[Int, 1](1))
    toks.set(0, Scalar[DType.int32](token))
    var emb1 = embedding_row_quantized(toks, m1.qparams.token_embd)

    # Get token embedding from dequantized mode
    var emb2_row = m2.params.token_embd.data().unsafe_offset(token * config.hidden)
    var emb2 = Tensor[DType.float16, 2](StaticTuple[Int, 2](1, config.hidden), emb2_row)

    # Compare
    var max_diff = Float32(0)
    var sum_diff = Float32(0)
    for i in range(config.hidden):
        var d = abs(Float32(emb1.get(i)) - Float32(emb2.get(i)))
        sum_diff += d
        if d > max_diff:
            max_diff = d

    print("Embedding comparison:")
    print("  Max diff:", max_diff)
    print("  Mean diff:", sum_diff / Float32(config.hidden))

    # Print first 10 values
    print("\nFirst 10 values:")
    for i in range(10):
        print("  i=", i, " quant=", Float32(emb1.get(i)), " dequant=", Float32(emb2.get(i)))

    # Check if token_embd is quantized
    print("\nToken embedding info:")
    print("  quantized:", m1.qparams.token_embd.quantized)
    print("  ggml_type:", m1.qparams.token_embd.ggml_type)

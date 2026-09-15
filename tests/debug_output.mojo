# Compare output layer (lm_head) between quantized and dequantized modes

from src.core.gguf_loader import load_gguf
from src.core.transformer import TransformerModel, load_config
from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.quantized.qweight import qweight_from_fp16
from std.utils.static_tuple import StaticTuple
from std.math import abs


def main() raises:
    print("Comparing output layer (lm_head)...")

    # Load model in quantized mode
    var ctx1 = load_gguf("Qwen3-0.6B-UD-Q4_K_XL.gguf")
    var config = load_config(ctx1)
    var m1 = TransformerModel(config, ctx1^, 512)

    # Load model in dequantized mode
    var ctx2 = load_gguf("Qwen3-0.6B-UD-Q4_K_XL.gguf")
    var m2 = TransformerModel(config, ctx2^, 512, quant_resident=False)

    # Create a test input (hidden state)
    var hidden = config.hidden
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, hidden))
    for i in range(hidden):
        x.set(i, Scalar[DType.float16](Float16(0.1)))

    # Get output weights
    var w1 = m1.qparams.output_w
    var w2 = qweight_from_fp16(m2.params.output_w)

    print("Output weight info:")
    print("  w1 quantized:", w1.quantized)
    print("  w1 ggml_type:", w1.ggml_type)
    print("  w1 n_out:", w1.n_out)
    print("  w1 n_in:", w1.n_in)

    # Compute output using quantized path
    var dummy_scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))
    var out1 = w1.proj(x, dummy_scale)

    # Compute output using dequantized path
    var out2 = w2.proj(x, dummy_scale)

    # Compare
    var max_diff = Float32(0)
    var sum_diff = Float32(0)
    var n = out1.numel()
    for i in range(min(n, 1000)):  # Only check first 1000 for speed
        var d = abs(Float32(out1.get(i)) - Float32(out2.get(i)))
        sum_diff += d
        if d > max_diff:
            max_diff = d

    print("\nOutput comparison (first 1000):")
    print("  Max diff:", max_diff)
    print("  Mean diff:", sum_diff / Float32(min(n, 1000)))

    # Print first 10 values
    print("\nFirst 10 values:")
    for i in range(10):
        print("  i=", i, " quant=", Float32(out1.get(i)), " dequant=", Float32(out2.get(i)))

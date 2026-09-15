# Debug Q6_K matmul
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf, find_tensor
from src.core.transformer import load_config, TransformerModel
from src.core.tensor import tensor_zeros
from std.utils.static_tuple import StaticTuple


def check_q6k_matmul() raises:
    """Compare Q6_K matmul between quantized and dequantized."""
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
    
    # Get layer 0 V weight (uses Q6_K)
    var lv1 = m1.layer_view(0)
    var lv2 = m2.layer_view(0)
    
    print("V weight quantized:", lv1.v_w.quantized)
    print("V weight ggml_type:", lv1.v_w.ggml_type)
    
    # Create a test input
    var hidden = config1.hidden
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, hidden))
    for i in range(hidden):
        x.set(i, Scalar[DType.float16](Float16(0.01)))
    
    # Run matmul
    print("\nRunning V weight matmul...")
    var dummy_scale = m1._dummy_scale
    var out1 = lv1.v_w.proj(x, dummy_scale)
    var out2 = lv2.v_w.proj(x, dummy_scale)
    
    print("Output shape:", out1.shape()[0], out1.shape()[1])
    
    # Compare
    var max_diff = Float32(0.0)
    var max_idx = 0
    var n_heads = config1.n_heads
    var head_dim = hidden // n_heads
    
    for i in range(min(100, n_heads * head_dim)):
        var v1 = Float32(out1.get(i))
        var v2 = Float32(out2.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
            max_idx = i
        if diff > 0.1:
            print("  out[", i, "]: quant=", v1, " dequant=", v2, " diff=", diff)
    print("Max diff:", max_diff, "at index", max_idx)


def main() raises:
    check_q6k_matmul()

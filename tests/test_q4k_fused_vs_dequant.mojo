# Compare fused Q4_K kernel vs dequantize-then-dot for real weights
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config, TransformerModel
from src.core.tensor import tensor_zeros
from src.core.ops.quantized.dequantize import dequantize_into
from std.utils.static_tuple import StaticTuple


def check_fused_vs_dequant() raises:
    """Compare fused Q4_K dot product with dequantize-then-dot."""
    var model_path = "Hy-MT2-7B-Q4_K_M.gguf"
    
    # Load quantized model
    print("Loading quantized model...")
    var ctx = load_gguf(model_path)
    var config = load_config(ctx)
    var m = TransformerModel(config, ctx^, 512, quant_resident=True)
    
    # Get layer 0 Q weight (Q4_K)
    var lv = m.layer_view(0)
    var q_data = lv.q_w.data
    
    print("Q weight shape:", q_data.shape()[0], q_data.shape()[1])
    print("Q weight ggml_type:", lv.q_w.ggml_type)
    
    # Create a test input
    var hidden = config.hidden
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, hidden))
    for i in range(hidden):
        x.set(i, Scalar[DType.float16](Float16(0.01 * Float16(i % 100))))
    
    # Method 1: Dequantize first, then compute dot product
    print("\nMethod 1: Dequantize-then-dot...")
    var w_dequant = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](lv.q_w.n_out, lv.q_w.n_in))
    dequantize_into(lv.q_w.ggml_type, q_data.data(), 0, w_dequant, lv.q_w.n_out * lv.q_w.n_in)
    
    # Compute dot product for first row
    var dot1 = Float32(0.0)
    for i in range(lv.q_w.n_in):
        var wv = Float32(w_dequant.get(i))
        var xv = Float32(x.get(i))
        dot1 = dot1 + wv * xv
    
    print("Dequantize-then-dot result:", dot1)
    
    # Method 2: Fused SIMD kernel
    print("\nMethod 2: Fused SIMD kernel...")
    from src.core.ops.cpu.simd import vec_dot_q4_k
    var dot2 = vec_dot_q4_k[DType.float16](x.data(), q_data.data(), q_data.shape()[1] // 144)
    
    print("Fused SIMD kernel result:", dot2)
    print("Difference:", abs(dot1 - dot2))


def main() raises:
    check_fused_vs_dequant()

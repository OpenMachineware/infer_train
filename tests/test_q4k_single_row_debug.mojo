# Debug single row Q4_K kernel
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config, TransformerModel
from src.core.tensor import tensor_zeros
from src.core.ops.quantized.dequantize import dequantize_into
from std.utils.static_tuple import StaticTuple


def check_single_row() raises:
    """Compare single row Q4_K kernel vs dequantize-then-dot."""
    var model_path = "Hy-MT2-7B-Q4_K_M.gguf"
    
    # Load quantized model
    print("Loading quantized model...")
    var ctx = load_gguf(model_path)
    var config = load_config(ctx)
    var m = TransformerModel(config, ctx^, 512, quant_resident=True)
    
    # Get layer 0 Q weight
    var lv = m.layer_view(0)
    var q_data = lv.q_w.data
    
    print("Q weight shape:", q_data.shape()[0], q_data.shape()[1])  # [4096, 2304]
    print("Q weight ggml_type:", lv.q_w.ggml_type)  # 12
    print("Q weight n_out:", lv.q_w.n_out, "n_in:", lv.q_w.n_in)  # 4096, 4096
    
    # Number of blocks per row
    var nb = q_data.shape()[1] // 144  # 2304 // 144 = 16
    print("Blocks per row:", nb)  # Should be 16 (4096 // 256 = 16)
    
    # Create a test input (K=4096)
    var hidden = config.hidden
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, hidden))
    for i in range(hidden):
        x.set(i, Scalar[DType.float16](Float16(0.01 * Float16(i % 100))))
    
    # Method 1: Dequantize first row, then compute dot product
    print("\nMethod 1: Dequantize-then-dot for first row...")
    var row0_bytes = nb * 144  # 2304 bytes
    var row0_ptr = q_data.data()  # First row
    
    var w_dequant = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, lv.q_w.n_in))
    dequantize_into(lv.q_w.ggml_type, row0_ptr, 0, w_dequant, lv.q_w.n_in)
    
    var dot1 = Float32(0.0)
    for i in range(lv.q_w.n_in):
        var wv = Float32(w_dequant.get(i))
        var xv = Float32(x.get(i))
        dot1 = dot1 + wv * xv
    
    print("Dequantize-then-dot result:", dot1)
    
    # Method 2: Fused SIMD kernel for first row
    print("\nMethod 2: Fused SIMD kernel for first row...")
    from src.core.ops.cpu.simd import vec_dot_q4_k
    var dot2 = vec_dot_q4_k[DType.float16](x.data(), row0_ptr, nb)
    
    print("Fused SIMD kernel result:", dot2)
    print("Difference:", abs(dot1 - dot2))
    
    # Also check the dequantized weight values
    print("\nDequantized weight values (first 10):")
    for i in range(10):
        print("  w_dequant[", i, "]=", Float32(w_dequant.get(i)))


def main() raises:
    check_single_row()

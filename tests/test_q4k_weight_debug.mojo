# Test Q4_K weight in model
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config, TransformerModel
from src.core.tensor import tensor_zeros
from std.utils.static_tuple import StaticTuple


def check_q4k_weight() raises:
    """Test Q weight (uses Q4_K) matmul."""
    var model_path = "Hy-MT2-7B-Q4_K_M.gguf"
    
    # Load both models
    print("Loading models...")
    var ctx1 = load_gguf(model_path)
    var config1 = load_config(ctx1)
    var m1 = TransformerModel(config1, ctx1^, 512, quant_resident=True)
    
    var ctx2 = load_gguf(model_path)
    var config2 = load_config(ctx2)
    var m2 = TransformerModel(config2, ctx2^, 512, quant_resident=False)
    
    # Get layer 0 Q weight (uses Q4_K)
    var lv1 = m1.layer_view(0)
    var lv2 = m2.layer_view(0)
    
    print("Q weight ggml_type:", lv1.q_w.ggml_type)
    print("Q weight quantized:", lv1.q_w.quantized)
    
    # Create test input
    var hidden = config1.hidden
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, hidden))
    for i in range(hidden):
        x.set(i, Scalar[DType.float16](Float16(0.01 * Float16(i % 100))))
    
    # Run Q projection
    print("\nRunning Q projection...")
    var dummy_scale = m1._dummy_scale
    var q1 = lv1.q_w.proj(x, dummy_scale)
    var q2 = lv2.q_w.proj(x, dummy_scale)
    
    print("Q output shape:", q1.shape()[0], q1.shape()[1])
    
    # Compare
    var n_heads = config1.n_heads
    var head_dim = hidden // n_heads
    var max_diff = Float32(0.0)
    var max_idx = 0
    
    for i in range(n_heads * head_dim):
        var v1 = Float32(q1.get(i))
        var v2 = Float32(q2.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
            max_idx = i
        if diff > 0.1:
            print("  q[", i, "]: quant=", v1, " dequant=", v2, " diff=", diff)
    
    print("Max diff:", max_diff, "at index", max_idx)
    
    # Check correlation
    var sum1 = Float32(0.0)
    var sum2 = Float32(0.0)
    var sum_prod = Float32(0.0)
    var n = n_heads * head_dim
    
    for i in range(n):
        var v1 = Float32(q1.get(i))
        var v2 = Float32(q2.get(i))
        sum1 = sum1 + v1
        sum2 = sum2 + v2
        sum_prod = sum_prod + v1 * v2
    
    var mean1 = sum1 / Float32(n)
    var mean2 = sum2 / Float32(n)
    var covar = sum_prod / Float32(n) - mean1 * mean2
    print("Q output correlation numerator:", covar)


def main() raises:
    check_q4k_weight()

# Verify dequantization correctness
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config, TransformerModel
from src.core.tensor import tensor_zeros
from src.core.ops.quantized.dequantize import dequantize_into
from std.utils.static_tuple import StaticTuple


def check_dequant() raises:
    """Verify dequantization matches between two paths."""
    var model_path = "Hy-MT2-7B-Q4_K_M.gguf"

    # Load both models
    print("Loading models...")
    var ctx1 = load_gguf(model_path)
    var config1 = load_config(ctx1)
    var m1 = TransformerModel(config1, ctx1^, 512, quant_resident=True)

    var ctx2 = load_gguf(model_path)
    var config2 = load_config(ctx2)
    var m2 = TransformerModel(config2, ctx2^, 512, quant_resident=False)

    # Get Q weight
    var lv1 = m1.layer_view(0)
    var lv2 = m2.layer_view(0)

    print("Q weight (quantized):")
    print("  ggml_type:", lv1.q_w.ggml_type)
    print("  n_out:", lv1.q_w.n_out, "n_in:", lv1.q_w.n_in)

    print("Q weight (dequantized FP16):")
    var q_fp16 = lv2.q_w.fp16
    print("  shape:", q_fp16.shape()[0], q_fp16.shape()[1])

    # Dequantize the first row of the quantized weight
    var row_bytes = lv1.q_w.data.shape()[1]
    var row_ptr = lv1.q_w.data.data()

    var w_dequant = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, lv1.q_w.n_in))
    dequantize_into(lv1.q_w.ggml_type, row_ptr, 0, w_dequant, lv1.q_w.n_in)

    # Compare with FP16 version
    print("\nComparing first row dequantized vs FP16:")
    var max_diff = Float32(0.0)
    for i in range(min(20, lv1.q_w.n_in)):
        var v1 = Float32(w_dequant.get(i))
        var v2 = Float32(q_fp16.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff
        print("  w[", i, "]: dequant=", v1, " fp16=", v2, " diff=", diff)
    print("Max diff:", max_diff)


def main() raises:
    check_dequant()

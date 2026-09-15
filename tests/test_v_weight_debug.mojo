# Debug V weight dequantization
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf, find_tensor
from src.core.transformer import load_config, TransformerModel
from src.core.tensor import tensor_zeros
from std.utils.static_tuple import StaticTuple


def check_v_weight() raises:
    """Compare V weight dequantization."""
    var model_path = "Hy-MT2-7B-Q4_K_M.gguf"

    # Load dequantized model
    print("Loading dequantized model...")
    var ctx2 = load_gguf(model_path)
    var config2 = load_config(ctx2)
    var m2 = TransformerModel(config2, ctx2^, 512, quant_resident=False)

    # Load quantized model
    print("Loading quantized model...")
    var ctx1 = load_gguf(model_path)
    var config1 = load_config(ctx1)
    var m1 = TransformerModel(config1, ctx1^, 512, quant_resident=True)

    # Get layer 0 V weights
    var lv1 = m1.layer_view(0)
    var lv2 = m2.layer_view(0)

    print("V weight quantized (quantized model):", lv1.v_w.quantized)
    print("V weight quantized (dequantized model):", lv2.v_w.quantized)
    print("V weight ggml_type (quantized):", lv1.v_w.ggml_type)

    # Check the FP16 V weight from dequantized model
    var v_fp16 = lv2.v_w.fp16
    print("V FP16 shape:", v_fp16.shape()[0], v_fp16.shape()[1])

    # Sample some values from the dequantized V weight
    print("\nSample V weight values (dequantized FP16):")
    for i in [0, 100, 500, 1000, 2000]:
        print("  v[", i, "]:", Float32(v_fp16.get(i)))

    # Dequantize a block from quantized model and compare
    # The V weight is stored as Q6_K, let's dequantize the first block
    var v_data = lv1.v_w.data
    print("\nQuantized V data shape:", v_data.shape()[0], v_data.shape()[1])


def main() raises:
    check_v_weight()

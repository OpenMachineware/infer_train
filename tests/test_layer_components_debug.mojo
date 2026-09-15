# Debug layer components step by step
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config, TransformerModel, rms_norm_weight
from src.core.ops.attention.mha import mha_forward_v2, MHAOptions
from std.utils.static_tuple import StaticTuple


def check_layer_components() raises:
    """Step through layer 0 components."""
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

    # Get embedding
    var token = 100
    var emb1 = m1.embed(token)
    var emb2 = m2.embed(token)

    # Get layer 0 view
    var lv1 = m1.layer_view(0)
    var lv2 = m2.layer_view(0)

    # Step 1: Apply attention norm
    var norm1 = rms_norm_weight[DType.float16](emb1, lv1.attn_norm_w, config1.norm_eps)
    var norm2 = rms_norm_weight[DType.float16](emb2, lv2.attn_norm_w, config2.norm_eps)

    print("After attention norm (first 5):")
    for i in range(5):
        var v1 = Float32(norm1.get(i))
        var v2 = Float32(norm2.get(i))
        print("  norm[", i, "]: quant=", v1, " dequant=", v2, " diff=", abs(v1 - v2))

    # Step 2: Run attention
    var opts = MHAOptions()
    opts.q_norm = config1.has_qk_norm
    opts.k_norm = config1.has_qk_norm
    opts.norm_before_rope = config1.norm_before_rope
    opts.gate = config1.has_gate
    opts.n_rot = config1.n_rot
    opts.norm_eps = config1.norm_eps

    var attn1 = mha_forward_v2(
        norm1, lv1.q_w, lv1.k_w, lv1.v_w, lv1.o_w,
        lv1.q_b, lv1.k_b, lv1.v_b,
        lv1.attn_q_norm, lv1.attn_k_norm,
        0, 0, opts
    )
    var attn2 = mha_forward_v2(
        norm2, lv2.q_w, lv2.k_w, lv2.v_w, lv2.o_w,
        lv2.q_b, lv2.k_b, lv2.v_b,
        lv2.attn_q_norm, lv2.attn_k_norm,
        0, 0, opts
    )

    print("\nAfter attention (first 5):")
    for i in range(5):
        var v1 = Float32(attn1.get(i))
        var v2 = Float32(attn2.get(i))
        print("  attn[", i, "]: quant=", v1, " dequant=", v2, " diff=", abs(v1 - v2))

    # Step 3: Residual
    var res1 = emb1
    var res2 = emb2
    # Add residual
    for i in range(config1.hidden):
        var v1 = Float32(attn1.get(i)) + Float32(res1.get(i))
        var v2 = Float32(attn2.get(i)) + Float32(res2.get(i))
        attn1.set(i, Scalar[DType.float16](Float16(v1)))
        attn2.set(i, Scalar[DType.float16](Float16(v2)))

    print("\nAfter residual (first 5):")
    for i in range(5):
        var v1 = Float32(attn1.get(i))
        var v2 = Float32(attn2.get(i))
        print("  res[", i, "]: quant=", v1, " dequant=", v2, " diff=", abs(v1 - v2))


def main() raises:
    check_layer_components()

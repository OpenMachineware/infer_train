# Debug quantization types for all weights
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf, find_tensor
from src.core.transformer import load_config, TransformerModel
from std.utils.static_tuple import StaticTuple


def check_quant_types() raises:
    """Check quantization types for all tensors."""
    var model_path = "Hy-MT2-7B-Q4_K_M.gguf"
    
    # Load model
    print("Loading model metadata...")
    var ctx = load_gguf(model_path)
    var config = load_config(ctx)
    
    # Check a few key tensors
    var tensors = [
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ]
    
    for name in tensors:
        var t_opt = find_tensor(ctx, name)
        if t_opt:
            var t = t_opt.value()
            print(name, "ggml_type:", t.ggml_type)
        else:
            print(name, "NOT FOUND")


def main() raises:
    check_quant_types()

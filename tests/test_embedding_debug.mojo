# Debug embedding layer
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf, find_tensor
from src.core.transformer import load_config, TransformerModel
from src.core.tensor import tensor_zeros, Tensor
from std.utils.static_tuple import StaticTuple


def check_embedding() raises:
    """Check embedding tensor quantization type."""
    var model_path = "Hy-MT2-7B-Q4_K_M.gguf"
    
    # Load model metadata
    var ctx = load_gguf(model_path)
    var config = load_config(ctx)
    
    # Find the embedding tensor
    var t_opt = find_tensor(ctx, "token_embd.weight")
    if t_opt is None:
        print("ERROR: embedding tensor not found")
        return
    var t = t_opt.value()
    print("Embedding tensor dims:", t.dims[0], t.dims[1])
    print("Embedding ggml_type:", t.ggml_type)
    
    # Check if there's a special kernel for this type
    print("\nChecking supported quant types:")
    print("  Q4_K (type 12): supported")
    print("  Q5_K (type 13): supported")
    print("  Q6_K (type 14): ", "supported" if t.ggml_type != 14 else "NOT SUPPORTED!")
    print("  Q4_0 (type 2): supported")
    print("  Q8_0 (type 8): supported")
    print("  IQ4_XS (type 23): supported")


def main() raises:
    check_embedding()

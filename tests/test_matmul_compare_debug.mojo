# Compare matmul implementations
# SPDX-License-Identifier: Apache-2.0

from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config, TransformerModel
from src.core.tensor import tensor_zeros
from src.core.ops.quantized.qweight import qweight_from_fp16
from std.utils.static_tuple import StaticTuple


def check_matmul_impl() raises:
    """Compare matmul implementations with same input."""
    var model_path = "Hy-MT2-7B-Q4_K_M.gguf"

    # Load dequantized model
    print("Loading dequantized model...")
    var ctx = load_gguf(model_path)
    var config = load_config(ctx)
    var m = TransformerModel(config, ctx^, 512, quant_resident=False)

    # Get layer 0 Q weight in FP16
    var lw = m.params.layers[0]
    var q_fp16 = lw.q_w

    print("Q weight shape:", q_fp16.shape()[0], q_fp16.shape()[1])

    # Create a test input
    var hidden = config.hidden
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, hidden))
    for i in range(hidden):
        x.set(i, Scalar[DType.float16](Float16(0.01)))

    # Method 1: matmul_weight_cpu_threaded (dequantized path)
    from src.core.ops.cpu.matmul_cpu import matmul_weight_cpu_threaded
    var out1 = matmul_weight_cpu_threaded[DType.float16](x, q_fp16)

    # Method 2: Wrap as QWeight and use proj (quantized path with FP16)
    var q_w = qweight_from_fp16(q_fp16)
    var dummy_scale = m._dummy_scale
    var out2 = q_w.proj(x, dummy_scale)

    print("Output shapes:", out1.shape()[0], out1.shape()[1], "and", out2.shape()[0], out2.shape()[1])

    # Compare
    var n_heads = config.n_heads
    var head_dim = hidden // n_heads
    var max_diff = Float32(0.0)

    for i in range(n_heads * head_dim):
        var v1 = Float32(out1.get(i))
        var v2 = Float32(out2.get(i))
        var diff = abs(v1 - v2)
        if diff > max_diff:
            max_diff = diff

    print("Max diff between matmul implementations:", max_diff)


def main() raises:
    check_matmul_impl()

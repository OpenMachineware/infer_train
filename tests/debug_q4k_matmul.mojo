# Test Q4_K × Q8_K matmul correctness

from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.quantized.quant_types import QuantType
from src.core.gguf_loader import load_gguf, find_tensor
from src.core.transformer import TransformerModel, load_config
from src.core.ops.quantized.qweight import QWeight
from std.utils.static_tuple import StaticTuple
from std.memory import Pointer
from std.origin import MutUntrackedOrigin


def main() raises:
    # Load a real Q4_K weight from the model
    var ctx = load_gguf("Hy-MT2-7B-Q4_K_M.gguf")
    var config = load_config(ctx)
    var m = TransformerModel(config, ctx^, 512)

    # Get the first layer's q_proj weight
    var lw = m.qparams.layers[0]
    var w_q4 = lw.q_w  # QWeight with Q4_K_M

    print("Q weight:")
    print("  ggml_type:", w_q4.ggml_type)
    print("  quantized:", w_q4.quantized)
    print("  n_out:", w_q4.n_out, "n_in:", w_q4.n_in)
    print("  data shape[0]:", w_q4.data.shape()[0], "shape[1]:", w_q4.data.shape()[1])

    # Create a test input (simple pattern for easier debugging)
    var hidden = config.hidden
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, hidden))
    for i in range(hidden):
        x.set(i, Scalar[DType.float16](Float16(i % 10)))

    # Compute using the QWeight projection (quantized path)
    var dummy_scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))
    var out_quant = w_q4.proj(x, dummy_scale)
    print("Quantized projection output shape[0]:", out_quant.shape()[0], "shape[1]:", out_quant.shape()[1])
    print("First 5 values:", out_quant.get(0), out_quant.get(1), out_quant.get(2), out_quant.get(3), out_quant.get(4))

    # Compute using dequantized weights (run in non-quant mode)
    # Create a second model with quant_resident=False
    var ctx2 = load_gguf("Hy-MT2-7B-Q4_K_M.gguf")
    var config2 = load_config(ctx2)
    var m2 = TransformerModel(config2, ctx2^, 512, quant_resident=False)
    var lw2 = m2.qparams.layers[0]

    # Simple matmul: out = x @ w^T
    # x: [1, hidden], w: [n_out, hidden] -> out: [1, n_out]
    var w_fp16 = lw2.q_w.data
    print("FP16 weight shape[0]:", w_fp16.shape()[0], "shape[1]:", w_fp16.shape()[1])

    var out_fp16 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, w_fp16.shape()[0]))
    for i in range(w_fp16.shape()[0]):
        var acc = Float32(0)
        for j in range(hidden):
            acc += Float32(x.get(j)) * Float32(w_fp16.get(i * hidden + j))
        out_fp16.set(i, Scalar[DType.float16](acc))

    print("FP16 projection output shape[0]:", out_fp16.shape()[0], "shape[1]:", out_fp16.shape()[1])
    print("First 5 values:", out_fp16.get(0), out_fp16.get(1), out_fp16.get(2), out_fp16.get(3), out_fp16.get(4))

    # Compare
    var max_diff = Float32(0)
    for i in range(min(out_quant.shape()[1], 100)):
        var d = abs(Float32(out_quant.get(i)) - Float32(out_fp16.get(i)))
        if d > max_diff:
            max_diff = d
    print("Max diff (first 100):", max_diff)

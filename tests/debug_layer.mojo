# Debug forward pass layer by layer
# Compare quantized vs dequantized at each layer

from src.core.gguf_loader import load_gguf, GGUFTensor
from src.core.transformer import TransformerModel, load_config
from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.quantized.qweight import QWeight, qweight_from_fp16
from src.core.ops.quantized.quant_types import QuantType
from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc
from std.math import abs
from std.utils.static_tuple import StaticTuple


comptime QK_K = 256
comptime BB_Q4K = 144


def dequantize_q4_k_block(
    block: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
):
    """Dequantize a single Q4_K block to float32."""
    var d_half = block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(d_half.unsafe_load[width=1](offset=0))
    var dmin = Float32(d_half.unsafe_load[width=1](offset=1))
    var scales = block.unsafe_offset(4)
    var qs = block.unsafe_offset(16)

    for j in range(8):
        var sc: Int
        var m: Int
        if j < 4:
            sc = Int(scales.unsafe_load[width=1](offset=j)) & 63
            m = Int(scales.unsafe_load[width=1](offset=j + 4)) & 63
        else:
            sc = (Int(scales.unsafe_load[width=1](offset=j + 4)) & 0xF) | (
                (Int(scales.unsafe_load[width=1](offset=j - 4)) >> 6) << 4
            )
            m = (Int(scales.unsafe_load[width=1](offset=j + 4)) >> 4) | (
                (Int(scales.unsafe_load[width=1](offset=j)) >> 6) << 4
            )

        var q_start = (j // 2) * 32
        var is_high = (j % 2) == 1

        for i in range(32):
            var byte_idx = q_start + i
            var qv_raw = Int(qs.unsafe_load[width=1](offset=byte_idx))
            var qv: Int
            if is_high:
                qv = qv_raw >> 4
            else:
                qv = qv_raw & 0x0F

            var val = d * Float32(sc) * Float32(qv) - dmin * Float32(m)
            dst.unsafe_offset(j * 32 + i).unsafe_store(val=Scalar[DType.float32](val))


def compare_matmul(
    x: Tensor[DType.float16, 2],
    w_q: QWeight,
    name: String,
) -> Float32:
    """Compare Q8_K matmul with dequantized matmul, return max diff."""
    if not w_q.quantized:
        return Float32(0)  # Skip if not quantized

    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w_q.n_out

    # Dequantize weights
    var w_fp16 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))
    var nb = K // QK_K
    var bb = BB_Q4K

    for j in range(N):
        for b in range(nb):
            var w_block = w_q.data.data().unsafe_offset(j * nb * bb + b * bb)
            var w_dequant = unsafe_alloc[Scalar[DType.float32]](QK_K)
            dequantize_q4_k_block(w_block, w_dequant)

            for i in range(QK_K):
                w_fp16.set(j * K + b * QK_K + i, Scalar[DType.float16](Float16(w_dequant.unsafe_offset(i).unsafe_load())))
            w_dequant.unsafe_free()

    # Compute reference
    var out_ref = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, N))
    for j in range(min(N, 10)):  # Only first 10 columns for speed
        var sum = Float32(0)
        for k in range(K):
            sum += Float32(x.get(k)) * Float32(w_fp16.get(j * K + k))
        out_ref.set(j, Scalar[DType.float16](Float16(sum)))

    # Compute Q8_K
    var dummy_scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))
    var out_q8k = matmul_quantized_q8k[QuantType.Q4_K_M](x, w_q.data, dummy_scale)

    # Compare
    var max_diff = Float32(0)
    for j in range(min(N, 10)):
        var d = abs(Float32(out_ref.get(j)) - Float32(out_q8k.get(j)))
        if d > max_diff:
            max_diff = d

    print(name, "max diff:", max_diff)
    return max_diff


def main() raises:
    print("Debugging forward pass layer by layer...")

    # Load model
    var ctx = load_gguf("Qwen3-0.6B-UD-Q4_K_XL.gguf")
    var config = load_config(ctx)
    var model = TransformerModel(config, ctx^, 512)

    # Get first layer weights
    var layer0 = model.qparams.layers[0]

    # Create a simple input
    var hidden = config.hidden
    print("Hidden size:", hidden)

    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, hidden))
    for i in range(hidden):
        x.set(i, Scalar[DType.float16](Float16(0.1)))

    # Compare Q projection
    print("\n1. Layer 0 attention Q projection...")
    var q_diff = compare_matmul(x, layer0.q_w, "attn_q")

    # Compare K projection
    print("\n2. Layer 0 attention K projection...")
    var k_diff = compare_matmul(x, layer0.k_w, "attn_k")

    # Compare V projection
    print("\n3. Layer 0 attention V projection...")
    var v_diff = compare_matmul(x, layer0.v_w, "attn_v")

    # Compare O projection
    print("\n4. Layer 0 attention O projection...")
    var o_diff = compare_matmul(x, layer0.o_w, "attn_o")

    # Compare FFN gate
    print("\n5. Layer 0 FFN gate...")
    var gate_diff = compare_matmul(x, layer0.gate_w, "ffn_gate")

    # Compare FFN up
    print("\n6. Layer 0 FFN up...")
    var up_diff = compare_matmul(x, layer0.up_w, "ffn_up")

    print("\nSummary:")
    print("  Q projection diff:", q_diff)
    print("  K projection diff:", k_diff)
    print("  V projection diff:", v_diff)
    print("  O projection diff:", o_diff)
    print("  FFN gate diff:", gate_diff)
    print("  FFN up diff:", up_diff)

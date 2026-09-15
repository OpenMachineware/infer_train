# Test full matmul with real weights
# Compare Q8_K path with dequantized BLAS path

from src.core.gguf_loader import load_gguf
from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.quantized.quant_types import QuantType
from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc
from std.math import abs
from std.utils.static_tuple import StaticTuple


comptime QK_K = 256
comptime BB_Q4K = 144


def dequantize_q4_k_weight(
    w_data: Tensor[DType.uint8, 2],
    n_out: Int,
    n_in: Int,
) -> Tensor[DType.float16, 2]:
    """Dequantize a full Q4_K weight matrix."""
    var nb = n_in // QK_K
    var bb = BB_Q4K

    var out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](n_out, n_in))

    for j in range(n_out):
        for b in range(nb):
            var w_block = w_data.data().unsafe_offset(j * nb * bb + b * bb)

            # Dequantize this block
            var d_half = w_block.unsafe_bitcast[Scalar[DType.float16]]()
            var d = Float32(d_half.unsafe_load[width=1](offset=0))
            var dmin = Float32(d_half.unsafe_load[width=1](offset=1))
            var scales = w_block.unsafe_offset(4)
            var qs = w_block.unsafe_offset(16)

            for sb in range(8):
                # Unpack scale and min
                var sc: Int
                var m: Int
                if sb < 4:
                    sc = Int(scales.unsafe_load[width=1](offset=sb)) & 63
                    m = Int(scales.unsafe_load[width=1](offset=sb + 4)) & 63
                else:
                    sc = (Int(scales.unsafe_load[width=1](offset=sb + 4)) & 0xF) | (
                        (Int(scales.unsafe_load[width=1](offset=sb - 4)) >> 6) << 4
                    )
                    m = (Int(scales.unsafe_load[width=1](offset=sb + 4)) >> 4) | (
                        (Int(scales.unsafe_load[width=1](offset=sb)) >> 6) << 4
                    )

                var q_start = (sb // 2) * 32
                var is_high = (sb % 2) == 1

                for i in range(32):
                    var byte_idx = q_start + i
                    var qv_raw = Int(qs.unsafe_load[width=1](offset=byte_idx))
                    var qv: Int
                    if is_high:
                        qv = qv_raw >> 4
                    else:
                        qv = qv_raw & 0x0F

                    var val = d * Float32(sc) * Float32(qv) - dmin * Float32(m)
                    var idx = j * n_in + b * QK_K + sb * 32 + i
                    out.set(idx, Scalar[DType.float16](Float16(val)))

    return out


def main() raises:
    print("Testing full Q4_K matmul...")

    # Load model weight
    var ctx = load_gguf("Qwen3-0.6B-UD-Q4_K_XL.gguf")

    var w_data: Tensor[DType.uint8, 2] = Tensor[DType.uint8, 2](StaticTuple[Int, 2](0, 0))
    var n_out = 0
    var n_in = 0

    for tensor in ctx.tensors:
        if tensor.ggml_type == 12 and "attn_q.weight" in tensor.name:
            w_data = ctx.load_tensor(tensor)
            n_out = tensor.dims[1]
            n_in = tensor.dims[0]
            print("Found tensor:", tensor.name)
            break

    if w_data.numel() == 0:
        print("Error: No Q4_K weight found")
        return

    print("Weight shape:", n_out, "x", n_in)

    # Create a simple input
    var M = 1
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, n_in))
    for i in range(n_in):
        x.set(i, Scalar[DType.float16](Float16(0.1)))

    # Dequantize weights to fp16
    print("Dequantizing weights...")
    var w_fp16 = dequantize_q4_k_weight(w_data, n_out, n_in)

    # Compute reference using simple matmul
    print("Computing reference matmul...")
    var out_ref = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, n_out))
    for j in range(n_out):
        var sum = Float32(0)
        for k in range(n_in):
            sum += Float32(x.get(k)) * Float32(w_fp16.get(j * n_in + k))
        out_ref.set(j, Scalar[DType.float16](Float16(sum)))

    # Compute using Q8_K path
    print("Computing Q8_K matmul...")
    var dummy_scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))
    var out_q8k = matmul_quantized_q8k[QuantType.Q4_K_M](x, w_data, dummy_scale)

    # Compare
    print("\nComparing outputs:")
    var max_diff = Float32(0)
    var sum_diff = Float32(0)
    var count = 0
    for j in range(n_out):
        var ref_val = Float32(out_ref.get(j))
        var q8k_val = Float32(out_q8k.get(j))
        var d = abs(ref_val - q8k_val)
        sum_diff += d
        if d > max_diff:
            max_diff = d
        count += 1

    print("Max diff:", max_diff)
    print("Mean diff:", sum_diff / Float32(count))

    # Print first few values
    print("\nFirst 10 values:")
    for j in range(10):
        print("  j=", j, " ref=", Float32(out_ref.get(j)), " q8k=", Float32(out_q8k.get(j)))

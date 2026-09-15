# Test real Q4_K block from model
# Compare vec_dot_q4_k_q8_k with reference dequantize + dot

from src.core.gguf_loader import load_gguf
from src.core.tensor import Tensor
from src.core.ops.cpu.simd.simd_neon import vec_dot_q4_k_q8_k
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
    """Dequantize a single Q4_K block to float32.

    Q4_K value: x = d * sc * q - dmin * m
    """
    var d_half = block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(d_half.unsafe_load[width=1](offset=0))
    var dmin = Float32(d_half.unsafe_load[width=1](offset=1))
    var scales = block.unsafe_offset(4)
    var qs = block.unsafe_offset(16)

    for j in range(8):
        # Unpack scale and min
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

        # Each scale j covers 32 elements
        # j=0,1: from bytes 0-31 (low/high nibble)
        # j=2,3: from bytes 32-63 (low/high nibble)
        # etc.
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


def quantize_to_q8_k_ref(
    x: Tensor[DType.float16, 1],
    dst: Pointer[UInt8, MutUntrackedOrigin],
):
    """Quantize to Q8_K format."""
    var n = x.shape()[0]

    # Find max
    var amax = Float32(0)
    for i in range(n):
        var v = abs(Float32(x.get(i)))
        if v > amax:
            amax = v

    if amax == 0:
        dst.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(val=Scalar[DType.float32](0))
        return

    var iscale = 127.0 / amax
    var d = amax / 127.0

    # Store scale
    dst.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(val=Scalar[DType.float32](d))

    # Quantize
    var qs_ptr = dst.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
    for i in range(n):
        var v = Int(round(iscale * Float32(x.get(i))))
        if v > 127:
            v = 127
        if v < -127:
            v = -127
        qs_ptr.unsafe_offset(i).unsafe_store(val=Scalar[DType.int8](v))

    # Compute bsums
    var bsums_ptr = dst.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
    for j in range(16):
        var sum = Int16(0)
        for ii in range(16):
            sum += Int16(qs_ptr.unsafe_offset(j * 16 + ii).unsafe_load())
        bsums_ptr.unsafe_offset(j).unsafe_store(val=Scalar[DType.int16](sum))


def main() raises:
    print("Testing real Q4_K block from model...")

    # Load model and get the quantized weight tensor
    var ctx = load_gguf("Qwen3-0.6B-UD-Q4_K_XL.gguf")

    # Find the first Q4_K weight tensor
    var w_data: Tensor[DType.uint8, 2] = Tensor[DType.uint8, 2](StaticTuple[Int, 2](0, 0))
    var n_out = 0
    var n_in = 0
    var ggml_type = 0

    for tensor in ctx.tensors:
        if tensor.ggml_type == 12 and "attn_q.weight" in tensor.name:  # Q4_K
            w_data = ctx.load_tensor(tensor)
            n_out = tensor.dims[1]
            n_in = tensor.dims[0]
            ggml_type = tensor.ggml_type
            print("Found tensor:", tensor.name)
            break

    if w_data.numel() == 0:
        print("Error: No Q4_K weight found")
        return

    print("Weight shape:", n_out, "x", n_in)
    print("ggml_type:", ggml_type)

    # Get the first block of the first column
    var nb = n_in // QK_K
    var bb = BB_Q4K

    print("Number of blocks per column:", nb)

    # Load the first Q4_K block
    var w_block = w_data.data().unsafe_offset(0 * nb * bb + 0 * bb)

    # Create a test input vector
    var x = Tensor[DType.float16, 1](StaticTuple[Int, 1](QK_K))
    for i in range(QK_K):
        x.set(i, Scalar[DType.float16](Float16(0.5)))

    # Quantize input to Q8_K
    var q8_data = unsafe_alloc[UInt8](292)
    quantize_to_q8_k_ref(x, q8_data)

    # Compute vec_dot using our kernel
    var result_kernel = vec_dot_q4_k_q8_k(w_block, q8_data)

    # Compute reference: dequantize Q4_K block, then dot product
    var w_dequant = unsafe_alloc[Scalar[DType.float32]](QK_K)
    dequantize_q4_k_block(w_block, w_dequant)

    # Read Q8_K scale
    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())
    var q8_qs = q8_data.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()

    var result_ref = Float32(0)
    for i in range(QK_K):
        var w_val = w_dequant.unsafe_offset(i).unsafe_load()
        var q8_val = Float32(q8_qs.unsafe_offset(i).unsafe_load())
        result_ref += w_val * q8_d * q8_val

    print("Kernel result:", result_kernel)
    print("Reference result:", result_ref)
    print("Difference:", abs(result_kernel - result_ref))

    # Print some debug info about the Q4_K block
    var d_half = w_block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(d_half.unsafe_load[width=1](offset=0))
    var dmin = Float32(d_half.unsafe_load[width=1](offset=1))
    print("d=", d, " dmin=", dmin)

    # Print scales
    var scales = w_block.unsafe_offset(4)
    print("Scales bytes:", end=" ")
    for i in range(12):
        print(Int(scales.unsafe_load[width=1](offset=i)), end=" ")
    print()

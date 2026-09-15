# Test Q6_K kernel with real data from model
from src.core.gguf_loader import load_gguf
from src.core.tensor import Tensor, tensor_zeros
from src.core.ops.cpu.simd.simd_neon import vec_dot_q6_k_q8_k
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc
from std.math import abs
from std.utils.static_tuple import StaticTuple


comptime QK_K = 256
comptime BB_Q6K = 210


def quantize_to_q8_k_ref(
    x: Tensor[DType.float16, 1],
    dst: Pointer[UInt8, MutUntrackedOrigin],
):
    """Quantize to Q8_K format."""
    var n = x.shape()[0]

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

    dst.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(val=Scalar[DType.float32](d))

    var qs_ptr = dst.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
    for i in range(n):
        var v = Int(round(iscale * Float32(x.get(i))))
        if v > 127:
            v = 127
        if v < -127:
            v = -127
        qs_ptr.unsafe_offset(i).unsafe_store(val=Scalar[DType.int8](v))

    var bsums_ptr = dst.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
    for j in range(16):
        var sum = Int16(0)
        for ii in range(16):
            sum += Int16(qs_ptr.unsafe_offset(j * 16 + ii).unsafe_load())
        bsums_ptr.unsafe_offset(j).unsafe_store(val=Scalar[DType.int16](sum))


def main() raises:
    print("Testing Q6_K kernel with real model data...")

    # Load model
    var ctx = load_gguf("Qwen3-0.6B-UD-Q4_K_XL.gguf")

    # Find the output weight (which is Q6_K)
    var w_data: Tensor[DType.uint8, 2] = Tensor[DType.uint8, 2](StaticTuple[Int, 2](0, 0))
    var n_out = 0
    var n_in = 0

    for tensor in ctx.tensors:
        if tensor.ggml_type == 14:  # Q6_K
            w_data = ctx.load_tensor(tensor)
            n_out = tensor.dims[1]
            n_in = tensor.dims[0]
            print("Found Q6_K tensor:", tensor.name)
            break

    if w_data.numel() == 0:
        print("Error: No Q6_K tensor found")
        return

    print("Tensor shape:", n_out, "x", n_in)

    # Get the first Q6_K block
    var nb = n_in // QK_K
    var bb = BB_Q6K

    var w_block = w_data.data().unsafe_offset(0 * nb * bb + 0 * bb)

    # Create a simple input vector
    var x = Tensor[DType.float16, 1](StaticTuple[Int, 1](QK_K))
    for i in range(QK_K):
        x.set(i, Scalar[DType.float16](Float16(0.1)))

    # Quantize input to Q8_K
    var q8_data = unsafe_alloc[UInt8](292)
    quantize_to_q8_k_ref(x, q8_data)

    # Run the kernel
    var result = vec_dot_q6_k_q8_k(w_block, q8_data)

    print("Kernel result:", result)

    # Dequantize the Q6_K block to verify
    # Q6_K layout: ql(128) qh(64) scales(16) d(2)
    var d = Float32(w_block.unsafe_offset(208).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())
    print("d =", d)

    # Print scales
    var scales = w_block.unsafe_offset(192).unsafe_bitcast[Scalar[DType.int8]]()
    print("Scales:", end=" ")
    for i in range(16):
        print(Int(scales.unsafe_load[width=1](offset=i)), end=" ")
    print()

    # Print some ql and qh values
    var ql = w_block.unsafe_offset(0)
    var qh = w_block.unsafe_offset(128)
    print("ql[0:4]:", end=" ")
    for i in range(4):
        print(Int(ql.unsafe_load[width=1](offset=i)), end=" ")
    print()
    print("qh[0:4]:", end=" ")
    for i in range(4):
        print(Int(qh.unsafe_load[width=1](offset=i)), end=" ")
    print()

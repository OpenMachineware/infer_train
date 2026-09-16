# Test Q4_K matmul correctness
# Compare quantized Q8_K path with dequantized BLAS path

from src.core.tensor import Tensor, tensor_zeros, tensor_from_data
from src.core.ops.quantized.qweight import QWeight, qweight_from_fp16
from src.core.ops.quantized.quant_types import QuantType, block_bytes, block_elems
from src.core.ops.cpu.matmul_q8k import matmul_quantized_q8k
from src.core.ops.cpu.blas_cpu import matmul_quantized_blas_tiled
from std.memory import unsafe_alloc
from std.origin import MutUntrackedOrigin
from std.math import abs
from random import random_float32


comptime QK_K = 256
comptime BB_Q4K = 144  # block_bytes for Q4_K


def quantize_row_to_q4_k_ref(
    x: Tensor[DType.float32, 1],
    dst: Pointer[UInt8, MutUntrackedOrigin],
):
    """Reference Q4_K quantization following llama.cpp.

    Q4_K block layout (144 bytes):
    - d: fp16 scale at offset 0
    - dmin: fp16 min scale at offset 2
    - scales: 12 bytes at offset 4
    - qs: 128 bytes at offset 16
    """
    var n = x.shape()[0]
    var nb = n // QK_K

    for b in range(nb):
        var block_start = b * QK_K
        var block_dst = dst.unsafe_offset(b * BB_Q4K)

        # Quantize each 32-element sub-block
        var scales: Array[Float32, 8].zero
        var mins: Array[Float32, 8].zero

        for sb in range(8):
            var amax = Float32(0)
            var max_val = Float32(0)
            for i in range(32):
                var v = x.get(block_start + sb * 32 + i)
                var av = abs(v)
                if av > amax:
                    amax = av
                    max_val = v

            if amax == 0:
                continue

            # Quantize to 4-bit: range [0, 15]
            # Formula: v = d * sc * q - dmin * m
            # We need to find d, sc, m such that v ≈ d * sc * q - dmin * m

            # Simplified approach: use symmetric quantization
            var scale = amax / 15.0
            scales[sb] = scale
            mins[sb] = 0.0

        # Pack scales and mins into 6-bit values (simplified)
        # TODO: implement proper 6-bit packing

        # For now, just store the super-block scale
        var d_half = block_dst.unsafe_bitcast[Scalar[DType.float16]]()
        d_half.unsafe_store(val=Scalar[DType.float16](Float16(1.0)))
        d_half.unsafe_offset(1).unsafe_store(val=Scalar[DType.float16](Float16(0.0)))

        # Store quantized values
        var qs_ptr = block_dst.unsafe_offset(16)
        for sb in range(8):
            for i in range(32):
                var v = x.get(block_start + sb * 32 + i)
                var q = Int(round(v / scales[sb])) if scales[sb] > 0 else 0
                q = max(0, min(15, q))
                # Pack two 4-bit values per byte
                if i % 2 == 0:
                    var existing = qs_ptr.unsafe_load[width=1](offset=sb * 16 + i // 2)
                    qs_ptr.unsafe_offset(sb * 16 + i // 2).unsafe_store(val=Scalar[DType.uint8]((existing & 0xF0) | UInt8(q)))
                else:
                    var existing = qs_ptr.unsafe_load[width=1](offset=sb * 16 + i // 2)
                    qs_ptr.unsafe_offset(sb * 16 + i // 2).unsafe_store(val=Scalar[DType.uint8]((existing & 0x0F) | (UInt8(q) << 4)))


def test_simple():
    """Test with a simple known case."""
    print("Testing Q4_K matmul correctness...")

    # Create a simple test case: 1x256 input, 2x256 weight
    var M = 1
    var K = 256
    var N = 2

    # Create input tensor
    var x_data = unsafe_alloc[Scalar[DType.float16]](M * K)
    for i in range(M * K):
        x_data.unsafe_offset(i).unsafe_store(val=Scalar[DType.float16](Float16(1.0)))
    var x = Tensor[DType.float16, 2](StaticTuple[Int, 2](M, K), x_data)

    # Create weight tensor with known values
    # We'll use identity-like weights for simplicity
    var w_fp16 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))
    for j in range(N):
        for k in range(K):
            w_fp16.set(j * K + k, Scalar[DType.float16](Float16(1.0)))

    # Compute expected result using fp16 matmul
    # y = W @ x^T where W is [N, K] and x is [M, K]
    # Expected: y[j] = sum_k W[j,k] * x[0,k] = sum_k 1.0 * 1.0 = K = 256
    print(f"Expected: all outputs = {K}")

    # Now test the quantized path
    # We need to create a Q4_K weight tensor
    # For this test, we'll use the dequantized path as reference

    var qweight = qweight_from_fp16(w_fp16)
    var dummy_scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))

    # This would use the fp16 path, not the quantized path
    # We need to test with actual Q4_K weights

    print("Test setup complete. Need to test with actual quantized weights.")


def main():
    test_simple()


if __name__ == "__main__":
    main()

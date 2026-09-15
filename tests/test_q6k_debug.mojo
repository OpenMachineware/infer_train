# Debug test for Q6_K kernel
# Compare element-by-element with dequantized reference

from tensor import Tensor
from core.dtypes import DType
from core.quantization import quantize_q6_k, dequantize_q6_k
from core.tensor_ops import randn
from std.memory import Pointer
from std.origin import MutUntrackedOrigin

comptime QK_K = 256


def _dequantize_q6_k_block(
    src: Pointer[UInt8, MutUntrackedOrigin],
    dst: Tensor[DType.float32, 2],
    offset: Int,
):
    """Dequantize one Q6_K block (210 bytes) to 256 f32 values."""
    var ql = src.unsafe_offset(0)
    var qh = src.unsafe_offset(128)
    var scales = src.unsafe_offset(192).unsafe_bitcast[Scalar[DType.int8]]()
    var d = Float32(src.unsafe_offset(208).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())

    for n in range(2):
        for l in range(32):
            var is_idx = 8 * n + l // 16
            var scale = Int32(scales.unsafe_load[width=1](offset=is_idx).value)

            # Extract q6 values for this position
            var ql_byte = ql.unsafe_load[width=1](offset=32 * n + l).value
            var qh_byte = qh.unsafe_load[width=1](offset=32 * n + l).value

            # 4 values per byte: l, l+32, l+64, l+96
            for m in range(4):
                var ql_low = Int((ql_byte >> (2 * m)) & 0x0F)
                var qh_bit = Int((qh_byte >> m) & 0x03)
                var q6 = ql_low | (qh_bit << 4)

                # Apply scale and bias
                var val = d * Float32(scale) * Float32(q6 - 32)

                var idx = offset + 128 * n + 32 * m + l
                dst.data().unsafe_offset(idx).unsafe_store(Scalar[DType.float32](val))


def _dequantize_q8_k_block(
    src: Pointer[UInt8, MutUntrackedOrigin],
    dst: Tensor[DType.float32, 2],
    offset: Int,
):
    """Dequantize one Q8_K block (292 bytes) to 256 f32 values."""
    var d = Float32(src.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())
    var qs = src.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()

    for i in range(256):
        var q = Int32(qs.unsafe_load[width=1](offset=i).value)
        var val = d * Float32(q)
        dst.data().unsafe_offset(offset + i).unsafe_store(Scalar[DType.float32](val))


fn test_q6k_element_debug() raises:
    """Debug Q6_K kernel by comparing with element-by-element dequantization"""
    print("=== Q6_K Element Debug ===\n")

    # Create a small test vector
    var original = Tensor[DType.float32](256)
    for i in range(256):
        original.data().unsafe_offset(i).unsafe_store(Scalar[DType.float32]((i.float32() - 128.0) * 0.1))

    # Quantize to Q6_K
    var q6k = quantize_q6_k(original)
    print("Q6_K block size: ", q6k.num_bytes(), " bytes")

    # Dequantize back using our function
    var dequant = Tensor[DType.float32](256)
    _dequantize_q6_k_block(q6k.data(), dequant, 0)

    # Compare element by element
    var max_diff = 0.0
    var sum_diff = 0.0
    var max_idx = 0
    for i in range(256):
        var orig_val = original.data().unsafe_offset(i).unsafe_load().value
        var deq_val = dequant.data().unsafe_offset(i).unsafe_load().value
        var diff = abs(orig_val - deq_val)
        sum_diff += diff
        if diff > max_diff:
            max_diff = diff
            max_idx = i

    print("Max diff: ", max_diff, " at index ", max_idx)
    print("Mean diff: ", sum_diff / 256.0)

    # Now test with Q8_K activation
    print("\n=== Testing Q6_K × Q8_K ===")

    # Create activation vector
    var activation = Tensor[DType.float32](256)
    for i in range(256):
        activation.data().unsafe_offset(i).unsafe_store(Scalar[DType.float32]((i.float32() - 128.0) * 0.05))

    # Quantize to Q8_K manually
    var q8k_buf = Pointer[DType.uint8, MutUntrackedOrigin].unsafe_alloc(292)
    var amax = Float32(0)
    for i in range(256):
        var v = activation.data().unsafe_offset(i).unsafe_load().value
        if abs(v) > amax:
            amax = abs(v)

    var d = amax / 127.0
    var iscale = 127.0 / amax
    q8k_buf.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(Scalar[DType.float32](d))

    var qs_ptr = q8k_buf.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
    for i in range(256):
        var v = Int(round(iscale * activation.data().unsafe_offset(i).unsafe_load().value))
        if v > 127:
            v = 127
        if v < -127:
            v = -127
        qs_ptr.unsafe_offset(i).unsafe_store(Scalar[DType.int8](v))

    # Compute bsums
    var bsums_ptr = q8k_buf.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
    for j in range(16):
        var sum = Int16(0)
        for ii in range(16):
            sum += Int16(qs_ptr.unsafe_offset(j * 16 + ii).unsafe_load().value)
        bsums_ptr.unsafe_offset(j).unsafe_store(Scalar[DType.int16](sum))

    print("Q8_K scale: ", d)
    print("Q8_K bsums[0]: ", bsums_ptr.unsafe_load[width=1](offset=0).value)

    # Compute dot product using kernel
    from core.ops.cpu.simd.simd_neon import vec_dot_q6_k_q8_k
    var kernel_result = vec_dot_q6_k_q8_k(q6k.data(), q8k_buf)
    print("\nKernel result: ", kernel_result)

    # Compute reference by dequantizing first
    var deq_weight = dequant
    var deq_act = Tensor[DType.float32](256)
    _dequantize_q8_k_block(q8k_buf, deq_act, 0)

    var reference_sum = 0.0
    for i in range(256):
        var w = deq_weight.data().unsafe_offset(i).unsafe_load().value
        var a = deq_act.data().unsafe_offset(i).unsafe_load().value
        reference_sum += w * a
    print("Reference sum: ", reference_sum)
    print("Diff: ", abs(kernel_result - reference_sum))

    q8k_buf.unsafe_free()


fn main() raises:
    test_q6k_element_debug()

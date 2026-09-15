# Test Q6_K kernel independently
from src.core.ops.cpu.simd.simd_neon import vec_dot_q6_k_q8_k
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc

def main():
    # Create test Q6_K block (210 bytes)
    # Layout: ql(128) qh(64) scales(16) d(2)
    var q6_block = unsafe_alloc[UInt8](210)

    # Create test Q8_K activation (292 bytes)
    # Layout: d(4) qs(256) bsums(16)
    var q8_data = unsafe_alloc[UInt8](292)

    # Initialize Q6_K
    # d is at offset 192+16 = 208 (after ql, qh, scales)
    # Wait, let me check the actual layout

    # Q6_K block layout:
    # ql[QK_K/2] = 128 bytes (lower 4 bits)
    # qh[QK_K/4] = 64 bytes (upper 2 bits)
    # scales[QK_K/16] = 16 bytes (8-bit scales)
    # d = 2 bytes (fp16)
    # Total = 128 + 64 + 16 + 2 = 210 bytes

    # Set d = 1.0 (fp16) at offset 208
    var d_half = q6_block.unsafe_offset(208).unsafe_bitcast[Scalar[DType.float16]]()
    d_half.unsafe_store(0, 1.0)

    # Set scales to 32 (scales are quantized with 8 bits, centered at 32)
    # Q6_K formula: x = d * (scale[i] - 32) * q6
    for i in range(16):
        q6_block.unsafe_offset(192 + i).unsafe_store(0, 32)  # scale = 32, so d * 0 * q6 = 0

    # For a non-zero result, use scale = 33 (so scale - 32 = 1)
    for i in range(16):
        q6_block.unsafe_offset(192 + i).unsafe_store(0, 33)

    # Set ql to simple pattern (lower 4 bits = 5, upper 4 bits = 5)
    for i in range(128):
        q6_block.unsafe_offset(i).unsafe_store(0, 0x55)  # 0101_0101

    # Set qh to 0 (upper 2 bits = 0)
    for i in range(64):
        q6_block.unsafe_offset(128 + i).unsafe_store(0, 0)

    # Set Q8_K d = 1.0 (float32)
    var q8_d = q8_data.unsafe_bitcast[Scalar[DType.float32]]()
    q8_d.unsafe_store(0, 1.0)

    # Set Q8_K qs to 1
    for i in range(256):
        q8_data.unsafe_offset(4 + i).unsafe_store(0, 1)

    # Run the kernel
    var result = vec_dot_q6_k_q8_k(q6_block, q8_data)

    # Q6_K: 6-bit values in range 0-63, centered at 32
    # With ql = 0x55 and qh = 0:
    # - ql lower nibble = 5
    # - ql upper nibble = 5
    # - qh bits = 0
    # So the 6-bit value is: lower_4 + upper_2 << 4 = 5 + 0 = 5
    # But Q6_K has the formula: x = d * (scale - 32) * (q6 - 32)
    # Wait, let me check the actual Q6_K formula

    print("Result: ", result)
    print("This is a basic test to verify the kernel runs without crashing")

# Test Q4_K kernel independently
from src.core.ops.cpu.simd.simd_neon import vec_dot_q4_k_q8_k
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc

def main():
    # Create test Q4_K block (144 bytes)
    # Layout: d(2) dmin(2) scales(12) qs(128)
    var q4_block = unsafe_alloc[UInt8](144)

    # Create test Q8_K activation (292 bytes)
    # Layout: d(4) qs(256) bsums(16)
    var q8_data = unsafe_alloc[UInt8](292)

    # Initialize with simple test data
    # Set d = 1.0 (fp16)
    var d_half = q4_block.unsafe_bitcast[Scalar[DType.float16]]()
    d_half.unsafe_store(0, 1.0)

    # Set dmin = 0 (fp16)
    d_half.unsafe_store(1, 0.0)

    # Set scales to 1 (using _get_scale_min_k4 format)
    # For j < 4: sc = scales[j] & 63, min = scales[j+4] & 63
    # For j >= 4: complex extraction
    # Simple approach: scales[0-3] = 1, scales[4-7] = 0 (for min)
    for i in range(12):
        q4_block.unsafe_offset(4 + i).unsafe_store(0, 1)

    # Set qs to simple pattern (low nibbles = 5, high nibbles = 10)
    for i in range(128):
        q4_block.unsafe_offset(16 + i).unsafe_store(0, 0x5A)  # 0101_1010 pattern

    # Set Q8_K d = 1.0 (float32)
    var q8_d = q8_data.unsafe_bitcast[Scalar[DType.float32]]()
    q8_d.unsafe_store(0, 1.0)

    # Set Q8_K qs to 1
    for i in range(256):
        q8_data.unsafe_offset(4 + i).unsafe_store(0, 1)

    # Set Q8_K bsums (for bias)
    for i in range(16):
        var bsum = q8_data.unsafe_offset(260 + i * 2).unsafe_bitcast[Scalar[DType.int16]]()
        bsum.unsafe_store(0, 16)

    # Run the kernel
    var result = vec_dot_q4_k_q8_k(q4_block, q8_data)

    # With q4 = 5/10 and q8 = 1
    # Q4_K uses 8 scales for 256 elements (32 elements per scale)
    # Each scale covers low nibble for 16 elements and high nibble for 16 elements
    # With qs = 0x5A, low nibble = 5, high nibble = 10
    # Expected: 128 * 5 + 128 * 10 = 1920
    print("Result: ", result)
    print("Expected: ~1920.0 (no bias since min=0)")

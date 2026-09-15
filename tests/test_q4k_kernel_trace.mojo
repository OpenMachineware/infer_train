# Trace Q4_K kernel execution
# SPDX-License-Identifier: Apache-2.0

from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc


def _get_scale_min_k4(
    j: Int, scales: Pointer[UInt8, MutUntrackedOrigin]
) -> Tuple[Int, Int]:
    """Unpack the 6-bit scale and min for Q4_K/Q5_K sub-block j."""
    if j < 4:
        return (
            Int(scales.unsafe_load[width=1](offset=j)) & 63,
            Int(scales.unsafe_load[width=1](offset=j + 4)) & 63,
        )
    var d = (Int(scales.unsafe_load[width=1](offset=j + 4)) & 0xF) | (
        (Int(scales.unsafe_load[width=1](offset=j - 4)) >> 6) << 4
    )
    var m = (Int(scales.unsafe_load[width=1](offset=j + 4)) >> 4) | (
        (Int(scales.unsafe_load[width=1](offset=j)) >> 6) << 4
    )
    return (d, m)


def trace_kernel() raises:
    """Manually trace through the kernel logic."""
    # Create a synthetic Q4_K block for testing
    # Block layout: d(2) dmin(2) scales(12) qs(128) = 144 bytes
    var data = unsafe_alloc[UInt8](144)
    
    # Set d = 1.0 (fp16 = 0x3C00)
    data.unsafe_store(0, UInt8(0x00))
    data.unsafe_store(1, UInt8(0x3C))
    
    # Set dmin = 0.0 (fp16 = 0x0000)
    data.unsafe_store(2, UInt8(0x00))
    data.unsafe_store(3, UInt8(0x00))
    
    # Set scales (12 bytes for 8 sub-blocks)
    # Each sub-block has scale and min (6 bits each)
    # For simplicity, set all to 1
    for i in range(12):
        data.unsafe_store(4 + i, UInt8(1))
    
    # Set qs (128 bytes)
    # Each byte holds 2 elements (4 bits each)
    # Set all to 8 (which gives value 0 after subtracting 8)
    for i in range(128):
        data.unsafe_store(16 + i, UInt8(0x88))  # Both nibbles = 8
    
    # Read the block header
    var half = data.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(half.unsafe_load[width=1](offset=0))
    var dmin = Float32(half.unsafe_load[width=1](offset=1))
    print("d:", d, "dmin:", dmin)
    
    # Read scales
    var scales = data.unsafe_offset(4)
    print("\nScales for each sub-block:")
    for j in range(8):
        var (scale_val, min_val) = _get_scale_min_k4(j, scales)
        print("  Sub-block", j, "scale:", scale_val, "min:", min_val)
    
    # Read qs
    var qs = data.unsafe_offset(16)
    print("\nFirst few qs bytes:")
    for i in range(5):
        var q = Int(qs.unsafe_load[width=1](offset=i))
        var lo = q & 0xF
        var hi = (q >> 4) & 0xF
        print("  qs[", i, "]=", q, "lo=", lo, "hi=", hi)
    
    # Expected output: d * sc * (q - 8) - dmin * m
    # With d=1, dmin=0, sc=1, m=1, q=8:
    # = 1 * 1 * (8 - 8) - 0 * 1 = 0
    
    data.unsafe_free()


def main() raises:
    trace_kernel()

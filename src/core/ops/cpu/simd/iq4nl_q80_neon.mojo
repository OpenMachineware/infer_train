# IQ4_NL NEON kernel for vector dot product
# Block size: 18 bytes for 32 weights (4.5 bpw)
# Non-linear quantization with 16-value lookup table

from src.core.ops.cpu.simd.simd_neon import neon_sdot, neon_addv, neon_tbl1
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.unsafe import bitcast
from std.builtin.globals import global_constant

# IQ4_NL lookup table (16 values)
comptime KVALUES_IQ4NL: Array[Int8, 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
]


def vec_dot_iq4nl_q80_neon(
    x: Pointer[UInt8, MutUntrackedOrigin],
    y: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """IQ4_NL × Q8_0 dot product - NEON SIMD optimized version.

    Block layout (18 bytes):
    - d: FP16 scale (2 bytes)
    - qs[16]: uint8 array (16 bytes, 4-bit packed)

    Q8_0 block layout (34 bytes):
    - d: FP16 scale (2 bytes)
    - qs[32]: int8 array (32 bytes)
    """
    ref values_ref = global_constant[KVALUES_IQ4NL]()
    var values = bitcast[DType.int8, 16](SIMD[DType.int8, 16](
        values_ref[0], values_ref[1], values_ref[2], values_ref[3],
        values_ref[4], values_ref[5], values_ref[6], values_ref[7],
        values_ref[8], values_ref[9], values_ref[10], values_ref[11],
        values_ref[12], values_ref[13], values_ref[14], values_ref[15],
    ))

    var sumf = Float32(0)

    for i in range(nb):
        var x_base = i * 18
        var y_base = i * 34

        # Q8_0 uses FP16 scale, not FP32!
        var d_x = Float32(x.unsafe_offset(x_base).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0))
        var d_y = Float32(y.unsafe_offset(y_base).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0))
        var d = d_x * d_y

        # Load 16 bytes of packed 4-bit values
        var q4bits = x.unsafe_load[width=16](offset=x_base + 2)

        # Load 32 int8 values (starts at offset 2, not 4!)
        var q8b0 = bitcast[DType.int8, 16](y.unsafe_load[width=16](offset=y_base + 2))
        var q8b1 = bitcast[DType.int8, 16](y.unsafe_load[width=16](offset=y_base + 18))

        # Unpack 4-bit values using TBL
        var mask4b = SIMD[DType.uint8, 16](0x0f)
        var values_u8 = bitcast[DType.uint8, 16](values)
        var q4b0 = bitcast[DType.int8, 16](neon_tbl1(values_u8, q4bits & mask4b))
        var q4b1 = bitcast[DType.int8, 16](neon_tbl1(values_u8, q4bits >> SIMD[DType.uint8, 16](4)))

        # SDOT
        var p = neon_sdot(SIMD[DType.int32, 4](0), q4b0, q8b0)
        p = neon_sdot(p, q4b1, q8b1)

        sumf += d * Float32(neon_addv(p))

    return sumf

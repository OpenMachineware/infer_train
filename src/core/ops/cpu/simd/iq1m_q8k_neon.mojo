# IQ1_M NEON kernel for vector dot product
# Block size: 56 bytes for 256 weights (1.75 bpw)
# Uses same IQ1S grid (2048 entries)

from src.core.ops.cpu.simd.simd_neon import neon_sdot, neon_addv
from src.core.ops.cpu.simd.iq1s_q8k_neon import IQ1S_GRID
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.unsafe import bitcast
from std.builtin.globals import global_constant

# IQ1M delta for bias correction
comptime IQ1M_DELTA: Float32 = 0.125

# Delta vectors for sign correction
comptime IQ1M_DELTAS: Array[UInt64, 4] = [
    0x0101010101010101,  # +1, +1
    0xff01ff01ff01ff01,  # -1, +1
    0x01ff01ff01ff01ff,  # +1, -1
    0xffffffffffffffff,  # -1, -1
]


@always_inline
def combine_s8_from_u64(lo: UInt64, hi: UInt64) -> SIMD[DType.int8, 16]:
    """Combine two uint64 (int8x8) into int8x16."""
    var combined = SIMD[DType.uint64, 2](lo, hi)
    return bitcast[DType.int8, 16](combined)


def vec_dot_iq1m_q8k_neon(
    x: Pointer[UInt8, MutUntrackedOrigin],
    y: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """IQ1_M × Q8_K dot product - NEON SIMD optimized version.

    Block layout (56 bytes):
    - qs[32]: uint8 array (32 bytes, grid index low 8 bits)
    - qh[16]: uint8 array (16 bytes, grid index high 3 bits + shift bit)
    - scales[8]: uint8 array (8 bytes, 3-bit scales)
    """
    ref grid_ref = global_constant[IQ1S_GRID]()
    var grid = grid_ref.unsafe_ptr()

    ref deltas_ref = global_constant[IQ1M_DELTAS]()
    var deltas = deltas_ref.unsafe_ptr()

    var sumf = Float32(0)

    for i in range(nb):
        var x_base = i * 56
        var y_base = i * 336

        var d_y = Float32(y.unsafe_offset(y_base).unsafe_bitcast[Scalar[DType.float32]]().unsafe_load[width=1](offset=0))

        # Extract merged scale from scales (packed as 4x 4-bit values in 16-bit)
        var scale_bytes = x.unsafe_load[width=2](offset=x_base + 48)
        var scale_u16 = UInt16(scale_bytes[0]) | (UInt16(scale_bytes[1]) << 8)
        # Treat uint16 as FP16 bits
        var d_x = Float32(bitcast[DType.float16, 1](SIMD[DType.uint16, 1](scale_u16))[0])
        var d = d_x * d_y

        var qs_offset = x_base
        var qh_offset = x_base + 32
        var scales_offset = x_base + 48
        var q8_offset = y_base + 4

        var sumi1 = Int32(0)
        var sumi2 = Int32(0)

        # Process 4 iterations (ib = 0, 2, 4, 6)
        for ib in range(0, 8, 2):
            var qs_base = qs_offset + (ib // 2) * 8
            var qh_base = qh_offset + (ib // 2) * 4

            # Load 8 qs values
            var qs_bytes = x.unsafe_load[width=8](offset=qs_base)

            # Load 4 qh values
            var qh_bytes = x.unsafe_load[width=4](offset=qh_base)

            # Construct grid indices
            var q1b0 = combine_s8_from_u64(
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[0]) | ((UInt32(qh_bytes[0]) << 8) & UInt32(0x700)))),
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[1]) | ((UInt32(qh_bytes[0]) << 4) & UInt32(0x700)))),
            )
            var q1b1 = combine_s8_from_u64(
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[2]) | ((UInt32(qh_bytes[1]) << 8) & UInt32(0x700)))),
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[3]) | ((UInt32(qh_bytes[1]) << 4) & UInt32(0x700)))),
            )
            var q1b2 = combine_s8_from_u64(
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[4]) | ((UInt32(qh_bytes[2]) << 8) & UInt32(0x700)))),
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[5]) | ((UInt32(qh_bytes[2]) << 4) & UInt32(0x700)))),
            )
            var q1b3 = combine_s8_from_u64(
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[6]) | ((UInt32(qh_bytes[3]) << 8) & UInt32(0x700)))),
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[7]) | ((UInt32(qh_bytes[3]) << 4) & UInt32(0x700)))),
            )

            # Load Q8 weights (64 bytes)
            var q8b0 = bitcast[DType.int8, 16](y.unsafe_load[width=16](offset=q8_offset + (ib // 2) * 64))
            var q8b1 = bitcast[DType.int8, 16](y.unsafe_load[width=16](offset=q8_offset + (ib // 2) * 64 + 16))
            var q8b2 = bitcast[DType.int8, 16](y.unsafe_load[width=16](offset=q8_offset + (ib // 2) * 64 + 32))
            var q8b3 = bitcast[DType.int8, 16](y.unsafe_load[width=16](offset=q8_offset + (ib // 2) * 64 + 48))

            # SDOT for main dot product
            var p1 = neon_sdot(SIMD[DType.int32, 4](0), q1b0, q8b0)
            p1 = neon_sdot(p1, q1b1, q8b1)
            var p2 = neon_sdot(SIMD[DType.int32, 4](0), q1b2, q8b2)
            p2 = neon_sdot(p2, q1b3, q8b3)

            # Delta correction using aux32 bits from qh
            var qh32_0 = UInt32(qh_bytes[0]) | (UInt32(qh_bytes[1]) << 8) | (UInt32(qh_bytes[2]) << 16) | (UInt32(qh_bytes[3]) << 24)
            var aux32 = ((qh32_0 >> 3) & UInt32(0x01010101)) | ((qh32_0 >> 6) & UInt32(0x02020202))

            var delta0 = deltas.unsafe_load[width=1](offset=Int(aux32 & UInt32(0xFF)))
            var delta1 = deltas.unsafe_load[width=1](offset=Int((aux32 >> 8) & UInt32(0xFF)))
            var delta2 = deltas.unsafe_load[width=1](offset=Int((aux32 >> 16) & UInt32(0xFF)))
            var delta3 = deltas.unsafe_load[width=1](offset=Int((aux32 >> 24) & UInt32(0xFF)))

            var d0 = combine_s8_from_u64(delta0, delta0)
            var d1 = combine_s8_from_u64(delta1, delta1)
            var d2 = combine_s8_from_u64(delta2, delta2)
            var d3 = combine_s8_from_u64(delta3, delta3)

            var p3 = neon_sdot(SIMD[DType.int32, 4](0), d0, q8b0)
            p3 = neon_sdot(p3, d1, q8b1)
            var p4 = neon_sdot(SIMD[DType.int32, 4](0), d2, q8b2)
            p4 = neon_sdot(p4, d3, q8b3)

            # Load scales
            var sc = UInt16(x.unsafe_load[width=2](offset=scales_offset + (ib // 2) * 2)[0]) | (UInt16(x.unsafe_load[width=2](offset=scales_offset + (ib // 2) * 2 + 1)[0]) << 8)

            # Extract 4 scales from packed 16-bit value
            var s0 = Int32((sc >> 0) & UInt16(0x7))
            var s1 = Int32((sc >> 3) & UInt16(0x7))
            var s2 = Int32((sc >> 6) & UInt16(0x7))
            var s3 = Int32((sc >> 9) & UInt16(0x7))

            # scales = 2 * (scale_bits & 7) + 1
            var ls0 = 2 * s0 + 1
            var ls1 = 2 * s1 + 1
            var ls2 = 2 * s2 + 1
            var ls3 = 2 * s3 + 1

            sumi1 += neon_addv(p1) * ls0 + neon_addv(p2) * ls1
            sumi2 += neon_addv(p3) * ls2 + neon_addv(p4) * ls3

        # Final result
        sumf += d * (Float32(sumi1) + IQ1M_DELTA * Float32(sumi2))

    return sumf

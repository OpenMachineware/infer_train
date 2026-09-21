# IQ1_M NEON kernel for vector dot product - Fully SIMD optimized
# Block size: 56 bytes for 256 weights (1.75 bpw)
# Matches llama.cpp NEON implementation exactly

from src.core.ops.cpu.simd.simd_neon import neon_sdot, neon_addv, neon_vpaddq_s32
from src.core.ops.cpu.simd.iq1s_q8k_neon import IQ1S_GRID
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.unsafe import bitcast
from std.builtin.globals import global_constant

# IQ1M delta for bias correction
comptime IQ1M_DELTA: Float32 = 0.125


@always_inline
def combine_s8_from_u64(lo: UInt64, hi: UInt64) -> SIMD[DType.int8, 16]:
    """Combine two uint64 (int8x8) into int8x16 - matches vcombine_s8."""
    var combined = SIMD[DType.uint64, 2](lo, hi)
    return bitcast[DType.int8, 16](combined)


@always_inline
def get_delta_vector(delta_bits: Int) -> SIMD[DType.int8, 16]:
    """Get delta vector based on 2-bit index (0-3).

    Index encoding:
    - bit 0: delta for first half (0 = +1, 1 = -1)
    - bit 1: delta for second half (0 = +1, 1 = -1)

    This matches llama.cpp's deltas.val[aux8[k]] lookup.
    """
    # Directly construct based on bits
    # This is safe because delta_bits is guaranteed to be 0-3
    if delta_bits == 0:  # (+1, +1)
        return SIMD[DType.int8, 16](1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    elif delta_bits == 1:  # (-1, +1)
        return SIMD[DType.int8, 16](-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1)
    elif delta_bits == 2:  # (+1, -1)
        return SIMD[DType.int8, 16](1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1)
    else:  # delta_bits == 3, (-1, -1)
        return SIMD[DType.int8, 16](-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)


def vec_dot_iq1m_q8k_neon(
    x: Pointer[UInt8, MutUntrackedOrigin],
    y: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """IQ1_M × Q8_K dot product - fully SIMD optimized matching llama.cpp exactly.

    Block layout (56 bytes):
    - qs[32]: grid index low 8 bits
    - qh[16]: grid index high 3 bits + delta bits
    - scales[8]: 4x uint16 packed scales
    """
    ref grid_ref = global_constant[IQ1S_GRID]()
    var grid = grid_ref.unsafe_ptr()

    var sumf = Float32(0)

    for i in range(nb):
        var x_base = i * 56
        var y_base = i * 292

        var d_y = Float32(y.unsafe_offset(y_base).unsafe_bitcast[Scalar[DType.float32]]().unsafe_load[width=1](offset=0))

        # Extract merged scale from 4 uint16 values
        var sc0 = UInt16(x.unsafe_load[width=1](offset=x_base + 48)) | (UInt16(x.unsafe_load[width=1](offset=x_base + 49)) << 8)
        var sc1 = UInt16(x.unsafe_load[width=1](offset=x_base + 50)) | (UInt16(x.unsafe_load[width=1](offset=x_base + 51)) << 8)
        var sc2 = UInt16(x.unsafe_load[width=1](offset=x_base + 52)) | (UInt16(x.unsafe_load[width=1](offset=x_base + 53)) << 8)
        var sc3 = UInt16(x.unsafe_load[width=1](offset=x_base + 54)) | (UInt16(x.unsafe_load[width=1](offset=x_base + 55)) << 8)
        var scale_u16 = (sc0 >> 12) | ((sc1 >> 8) & UInt16(0x00f0)) | ((sc2 >> 4) & UInt16(0x0f00)) | (sc3 & UInt16(0xf000))
        var d_x = Float32(bitcast[DType.float16, 1](SIMD[DType.uint16, 1](scale_u16))[0])
        var d = d_x * d_y

        var qs_ptr = x.unsafe_offset(x_base)
        var qh_ptr = x.unsafe_offset(x_base + 32)
        var q8_ptr = y.unsafe_offset(y_base + 4)

        # Access scales as uint16 array
        var sc_ptr = x.unsafe_offset(x_base + 48).unsafe_bitcast[Scalar[DType.uint16]]()

        var sumi1_vec = SIMD[DType.int32, 4](0)
        var sumi2_vec = SIMD[DType.int32, 4](0)

        # Process 8 sub-blocks, 2 at a time (matching llama.cpp ib += 2)
        for ib in range(0, 8, 2):
            # Load qh values (4 bytes for ib and ib+1)
            var qh0 = qh_ptr.unsafe_load[width=1](offset=0)  # qh[0]
            var qh1 = qh_ptr.unsafe_load[width=1](offset=1)  # qh[1]
            var qh2 = qh_ptr.unsafe_load[width=1](offset=2)  # qh[2]
            var qh3 = qh_ptr.unsafe_load[width=1](offset=3)  # qh[3]

            # Load 8 qs values
            var qs_bytes = qs_ptr.unsafe_load[width=8](offset=0)

            # Load grid vectors (matching llama.cpp lines 4149-4156)
            # Grid index = qs[k] | ((qh[k/2] << (8 - 4*(k%2))) & 0x700)
            var q1b0 = combine_s8_from_u64(
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[0]) | ((UInt32(qh0) << 8) & UInt32(0x700)))),
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[1]) | ((UInt32(qh0) << 4) & UInt32(0x700)))),
            )
            var q1b1 = combine_s8_from_u64(
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[2]) | ((UInt32(qh1) << 8) & UInt32(0x700)))),
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[3]) | ((UInt32(qh1) << 4) & UInt32(0x700)))),
            )
            var q1b2 = combine_s8_from_u64(
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[4]) | ((UInt32(qh2) << 8) & UInt32(0x700)))),
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[5]) | ((UInt32(qh2) << 4) & UInt32(0x700)))),
            )
            var q1b3 = combine_s8_from_u64(
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[6]) | ((UInt32(qh3) << 8) & UInt32(0x700)))),
                grid.unsafe_load[width=1](offset=Int(UInt32(qs_bytes[7]) | ((UInt32(qh3) << 4) & UInt32(0x700)))),
            )

            # Load Q8 weights (64 bytes = 4 vectors of 16 int8) - matches ggml_vld1q_s8_x4
            var q8b0 = bitcast[DType.int8, 16](q8_ptr.unsafe_load[width=16](offset=0))
            var q8b1 = bitcast[DType.int8, 16](q8_ptr.unsafe_load[width=16](offset=16))
            var q8b2 = bitcast[DType.int8, 16](q8_ptr.unsafe_load[width=16](offset=32))
            var q8b3 = bitcast[DType.int8, 16](q8_ptr.unsafe_load[width=16](offset=48))

            # Compute main dot products (matching llama.cpp lines 4160-4162)
            var p1_vec = neon_vpaddq_s32(
                neon_sdot(SIMD[DType.int32, 4](0), q1b0, q8b0),
                neon_sdot(SIMD[DType.int32, 4](0), q1b1, q8b1)
            )
            var p2_vec = neon_vpaddq_s32(
                neon_sdot(SIMD[DType.int32, 4](0), q1b2, q8b2),
                neon_sdot(SIMD[DType.int32, 4](0), q1b3, q8b3)
            )
            var p12_vec = neon_vpaddq_s32(p1_vec, p2_vec)

            # Delta correction: aux32 = ((qh32[0] >> 3) & 0x01010101) | ((qh32[0] >> 6) & 0x02020202)
            var qh32_0 = UInt32(qh0) | (UInt32(qh1) << 8) | (UInt32(qh2) << 16) | (UInt32(qh3) << 24)
            var aux32 = ((qh32_0 >> 3) & UInt32(0x01010101)) | ((qh32_0 >> 6) & UInt32(0x02020202))

            # Extract bytes as delta indices (aux8[k])
            var aux0 = Int(aux32 & UInt32(0xFF))
            var aux1 = Int((aux32 >> 8) & UInt32(0xFF))
            var aux2 = Int((aux32 >> 16) & UInt32(0xFF))
            var aux3 = Int((aux32 >> 24) & UInt32(0xFF))

            # Delta dot products (matching llama.cpp lines 4167-4169)
            var p3_vec = neon_vpaddq_s32(
                neon_sdot(SIMD[DType.int32, 4](0), get_delta_vector(aux0), q8b0),
                neon_sdot(SIMD[DType.int32, 4](0), get_delta_vector(aux1), q8b1)
            )
            var p4_vec = neon_vpaddq_s32(
                neon_sdot(SIMD[DType.int32, 4](0), get_delta_vector(aux2), q8b2),
                neon_sdot(SIMD[DType.int32, 4](0), get_delta_vector(aux3), q8b3)
            )
            var p34_vec = neon_vpaddq_s32(p3_vec, p4_vec)

            # Scales: scales_4 = sc[ib/2] >> [0, 3, 6, 9] (matching llama.cpp line 4171)
            var sc_val = sc_ptr.unsafe_load[width=1](offset=ib // 2)
            var scales_4 = SIMD[DType.int32, 4](
                Int32((Int(sc_val >> 0) & 7) << 1) + 1,
                Int32((Int(sc_val >> 3) & 7) << 1) + 1,
                Int32((Int(sc_val >> 6) & 7) << 1) + 1,
                Int32((Int(sc_val >> 9) & 7) << 1) + 1
            )

            # Accumulate with scales (matching llama.cpp lines 4175-4176)
            # sumi1 = vmlaq_s32(sumi1, scales_4, p12) => sumi1 += scales_4 * p12 (element-wise)
            sumi1_vec += scales_4 * p12_vec
            sumi2_vec += scales_4 * p34_vec

            # Advance pointers (matching llama.cpp line 4178)
            qs_ptr = qs_ptr.unsafe_offset(8)
            qh_ptr = qh_ptr.unsafe_offset(4)
            q8_ptr = q8_ptr.unsafe_offset(64)

        # Final result (matching llama.cpp line 4182)
        var sumi1_total = neon_addv(sumi1_vec)
        var sumi2_total = neon_addv(sumi2_vec)
        sumf += d * (Float32(sumi1_total) + IQ1M_DELTA * Float32(sumi2_total))

    return sumf

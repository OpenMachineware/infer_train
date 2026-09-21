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
    """IQ1_M × Q8_K dot product - matching llama.cpp logic exactly.

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
        # scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
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

        var sumi1 = Int32(0)
        var sumi2 = Int32(0)

        # Process 8 sub-blocks (ib = 0..7)
        for ib in range(8):
            # Extract deltas from qh[2*ib] and qh[2*ib+1]
            var qh0 = qh_ptr.unsafe_load[width=1](offset=2*ib)
            var qh1 = qh_ptr.unsafe_load[width=1](offset=2*ib + 1)

            var delta0 = Int32(-1 if (qh0 & 0x08) != 0 else 1)
            var delta1 = Int32(-1 if (qh0 & 0x80) != 0 else 1)
            var delta2 = Int32(-1 if (qh1 & 0x08) != 0 else 1)
            var delta3 = Int32(-1 if (qh1 & 0x80) != 0 else 1)

            var sum1_0 = Int32(0)
            var sum1_1 = Int32(0)
            var sum2_0 = Int32(0)
            var sum2_1 = Int32(0)

            # Process 4 grid lookups (l = 0..3)
            for l in range(4):
                var qs_val = qs_ptr.unsafe_load[width=1](offset=4*ib + l)
                var qh_val = qh_ptr.unsafe_load[width=1](offset=2*ib + l//2)

                # Grid index = qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)
                var shift = UInt32(8 - 4*(l % 2))
                var grid_idx = Int(UInt32(qs_val) | ((UInt32(qh_val) * shift) & UInt32(0x700)))

                # Load 8 int8 from grid (stored as UInt64)
                var grid_u64 = grid.unsafe_load[width=1](offset=grid_idx)
                var grid_vec = bitcast[DType.int8, 8](SIMD[DType.uint64, 1](grid_u64))

                # Load 8 int8 from q8
                var q8_vec = bitcast[DType.int8, 8](q8_ptr.unsafe_load[width=8](offset=32*ib + 8*l))

                # Compute dot product
                var lsum1 = Int32(0)
                var lsum2 = Int32(0)
                for j in range(8):
                    lsum1 += Int32(grid_vec[j]) * Int32(q8_vec[j])
                    lsum2 += Int32(q8_vec[j])

                var delta_val = delta0 if l == 0 else (delta1 if l == 1 else (delta2 if l == 2 else delta3))
                sum1_0 += lsum1 if l < 2 else 0
                sum1_1 += lsum1 if l >= 2 else 0
                sum2_0 += lsum2 * delta_val if l < 2 else 0
                sum2_1 += lsum2 * delta_val if l >= 2 else 0

            # Extract scales
            # ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1
            # ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1
            var sc_val = sc0 if (ib // 2) == 0 else (sc1 if (ib // 2) == 1 else (sc2 if (ib // 2) == 2 else sc3))
            var shift_base = UInt16(6 * (ib % 2))
            var ls1 = 2 * Int32((sc_val >> shift_base) & UInt16(0x7)) + 1
            var ls2 = 2 * Int32((sc_val >> (shift_base + UInt16(3))) & UInt16(0x7)) + 1

            sumi1 += sum1_0 * ls1 + sum1_1 * ls2
            sumi2 += sum2_0 * ls1 + sum2_1 * ls2

        sumf += d * (Float32(sumi1) + IQ1M_DELTA * Float32(sumi2))

    return sumf

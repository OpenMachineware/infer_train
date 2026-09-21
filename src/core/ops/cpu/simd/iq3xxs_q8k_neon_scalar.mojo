# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# IQ3_XXS × Q8_K kernel - Scalar reference implementation

from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.unsafe import bitcast
from std.builtin.globals import global_constant
from src.core.ops.cpu.simd.iq3xxs_q8k_neon import IQ3XXS_GRID, KEVEN_SIGNS_Q2XS

comptime QK_K = 256

def vec_dot_iq3xxs_q8k_scalar(
    x: Pointer[UInt8, MutUntrackedOrigin],
    y: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """IQ3_XXS × Q8_K dot product - Scalar reference implementation.

    Matches llama.cpp exactly for correctness verification.
    Block layout (98 bytes):
    - d: FP16 scale (2 bytes)
    - qs[0..95]: grid indices (96 bytes)
      - First 64 bytes: actual grid indices
      - Next 32 bytes: scales and signs (packed)
    """
    ref grid_ref = global_constant[IQ3XXS_GRID]()
    var grid = grid_ref.unsafe_ptr()

    ref signs_ref = global_constant[KEVEN_SIGNS_Q2XS]()
    var signs64 = signs_ref.unsafe_ptr()

    var sumf = Float32(0)

    for i in range(nb):
        var x_base = i * 98
        var y_base = i * 292

        var d_x = Float32(x.unsafe_offset(x_base).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0))
        var d_y = Float32(y.unsafe_offset(y_base).unsafe_bitcast[Scalar[DType.float32]]().unsafe_load[width=1](offset=0))
        var d = d_x * d_y

        var q3_ptr = x_base + 2  # Grid indices start at offset 2
        var gas_ptr = x_base + 66  # Scales and signs start at offset 66
        var q8_ptr = y_base + 4  # Q8 values start after d (4 bytes)

        var bsum = Int32(0)

        # Process 8 sub-blocks (ib32 = 0..7)
        for ib32 in range(8):
            # Load aux32 from gas (4 bytes)
            var aux32 = UInt32(x.unsafe_load[width=1](offset=gas_ptr)) |
                       (UInt32(x.unsafe_load[width=1](offset=gas_ptr + 1)) << 8) |
                       (UInt32(x.unsafe_load[width=1](offset=gas_ptr + 2)) << 16) |
                       (UInt32(x.unsafe_load[width=1](offset=gas_ptr + 3)) << 24)
            gas_ptr += 4

            # Scale: 2*(aux32 >> 28) + 1
            var ls = Int32(2 * (aux32 >> 28) + 1)

            var sumi = Int32(0)

            # Process 4 iterations (l = 0..3)
            for l in range(4):
                # Load 2 grid indices
                var q3_0 = Int(x.unsafe_load[width=1](offset=q3_ptr + 2*l))
                var q3_1 = Int(x.unsafe_load[width=1](offset=q3_ptr + 2*l + 1))

                # Load grid values (each entry = 4 bytes = 4 int8)
                var grid1_u32 = grid.unsafe_load[width=1](offset=q3_0)
                var grid2_u32 = grid.unsafe_load[width=1](offset=q3_1)

                # Extract 4 int8 values from each uint32
                var grid1 = [
                    Int32((grid1_u32 >> 0) & 0xFF),
                    Int32((grid1_u32 >> 8) & 0xFF),
                    Int32((grid1_u32 >> 16) & 0xFF),
                    Int32((grid1_u32 >> 24) & 0xFF),
                ]
                var grid2 = [
                    Int32((grid2_u32 >> 0) & 0xFF),
                    Int32((grid2_u32 >> 8) & 0xFF),
                    Int32((grid2_u32 >> 16) & 0xFF),
                    Int32((grid2_u32 >> 24) & 0xFF),
                ]

                # Load signs (one byte = 8 signs)
                var sign_idx = Int((aux32 >> UInt32(7*l)) & 127)
                var signs_u64 = signs64.unsafe_load[width=1](offset=sign_idx)

                # Bitcast uint64 to int8x8 to get the sign values
                var signs_vec = bitcast[DType.int8, 8](SIMD[DType.uint64, 1](signs_u64))
                var signs = [
                    Int32(signs_vec[0]),
                    Int32(signs_vec[1]),
                    Int32(signs_vec[2]),
                    Int32(signs_vec[3]),
                    Int32(signs_vec[4]),
                    Int32(signs_vec[5]),
                    Int32(signs_vec[6]),
                    Int32(signs_vec[7]),
                ]

                # Compute dot product for 8 Q8 values
                for j in range(4):
                    var q8_val0 = Int32(y.unsafe_load[width=1](offset=q8_ptr + j))
                    var q8_val1 = Int32(y.unsafe_load[width=1](offset=q8_ptr + j + 4))

                    sumi += grid1[j] * q8_val0 * signs[j]
                    sumi += grid2[j] * q8_val1 * signs[j + 4]

                q8_ptr += 8

            q3_ptr += 8
            bsum += sumi * ls

        sumf += d * Float32(bsum)

    return sumf * 0.25

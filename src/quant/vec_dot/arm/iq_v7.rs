// IQ3_XXS v7 - Match llama.cpp's scalar implementation exactly

use super::fp::fp16_to_fp32;
use crate::quant::types::{BlockIQ3XXS, BlockQ8K, QK_K};
use crate::quant::vec_dot::tables::IQ3XXS_GRID;

/// Sign bits table (matches ksigns_iq2xs)
static KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17,
    18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34,
    163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175, 48, 177, 178,
    51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63, 192, 65, 66, 195,
    68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83,
    212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99,
    228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240, 113, 114,
    243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
];

/// Mask for sign bits (matches kmask_iq2xs)
static KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// IQ3_XXS × Q8_K (scalar - exact llama.cpp match)
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_neon_v7(
    n: usize,
    x: &[BlockIQ3XXS],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;

    let grid_u32 = IQ3XXS_GRID.as_ptr();

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let mut q3 = x[i].qs.as_ptr();
        let mut gas = q3.add(QK_K / 4);
        let mut q8 = y[i].qs.as_ptr();

        let mut bsum: i32 = 0;

        for _ in 0..QK_K / 32 {
            let aux32 = std::ptr::read_unaligned(gas as *const u32);
            gas = gas.add(4);
            let ls = 2 * (aux32 >> 28) + 1;

            let mut sumi: i32 = 0;

            for l in 0..4 {
                let grid1 = grid_u32.add(*q3.add(2 * l) as usize) as *const u8;
                let grid2 =
                    grid_u32.add(*q3.add(2 * l + 1) as usize) as *const u8;
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];

                for j in 0..4 {
                    let sign1 =
                        if signs & KMASK_IQ2XS[j] != 0 { -1 } else { 1 };
                    let sign2 =
                        if signs & KMASK_IQ2XS[j + 4] != 0 { -1 } else { 1 };

                    sumi += *grid1.add(j) as i32 * *q8.add(j) as i32 * sign1;
                    sumi +=
                        *grid2.add(j) as i32 * *q8.add(j + 4) as i32 * sign2;
                }
                q8 = q8.add(8);
            }
            q3 = q3.add(8);

            bsum += sumi * ls as i32;
        }

        sumf += d * bsum as f32;
    }

    0.25 * sumf
}

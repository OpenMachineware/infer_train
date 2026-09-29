// IQ3_XXS v7 - Match llama.cpp's scalar implementation exactly
// No manual vectorization - let the compiler optimize

use super::fp::fp16_to_fp32;
use crate::quant::types::{BlockIQ3XXS, BlockQ8K, QK_K};
use crate::quant::vec_dot::tables::{IQ3XXS_GRID, KEVEN_SIGNS_Q2XS};

/// Mask table for IQ2XS/IQ3XXS sign extraction
static KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// IQ3_XXS × Q8_K (scalar implementation matching llama.cpp)
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_neon_v7(
    n: usize,
    x: &[BlockIQ3XXS],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;

    let grid = IQ3XXS_GRID.as_ptr() as *const u8;
    let signs_table = KEVEN_SIGNS_Q2XS.as_ptr();
    let mask_table = KMASK_IQ2XS.as_ptr();

    let mut sumf = 0.0f32;

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let q3 = x[i].qs.as_ptr();
        let gas = q3.add(QK_K / 4);
        let q8 = y[i].qs.as_ptr();

        let mut bsum: i32 = 0;

        for ib32 in 0..QK_K / 32 {
            // Load aux32
            let aux32 =
                std::ptr::read_unaligned(gas.add(ib32 * 4) as *const u32);
            let ls = 2 * (aux32 >> 28) + 1;

            let mut sumi: i32 = 0;

            for l in 0..4 {
                let grid1 =
                    grid.add(q3.add(ib32 * 8 + 2 * l).read() as usize * 4);
                let grid2 =
                    grid.add(q3.add(ib32 * 8 + 2 * l + 1).read() as usize * 4);
                let signs =
                    signs_table.add(((aux32 >> (7 * l)) & 127) as usize * 8);

                for j in 0..4 {
                    let q8_base = q8.add(ib32 * 32 + l * 8);

                    let sign1 = *signs.add(j + 0) as i32;
                    let sign2 = *signs.add(j + 4) as i32;

                    sumi += *grid1.add(j) as i32
                        * *q8_base.add(j + 0) as i32
                        * sign1;
                    sumi += *grid2.add(j) as i32
                        * *q8_base.add(j + 4) as i32
                        * sign2;
                }
            }

            bsum += sumi * ls as i32;
        }

        sumf += d * bsum as f32;
    }

    0.25 * sumf
}

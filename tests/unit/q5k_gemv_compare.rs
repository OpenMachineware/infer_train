//! Compare interleaved GEMV vs vec_dot for Q5_K
use infer_train::backend::arm::{gemv, repack, vec_dot};
use infer_train::types::{BlockQ5_K, BlockQ5_Kx8, BlockQ8_K, QK_K};

#[test]
fn test_gemv_vs_vec_dot() {
    let nb = 4; // blocks per row
    let n_rows = 8;

    // Create weights in STANDARD row-major layout:
    // Layout: [row0_blk0, row0_blk1, row0_blk2, row0_blk3, row1_blk0, ...]
    // stride = nb * size_of::<BlockQ5_K>() = distance between consecutive rows
    let mut weights: Vec<BlockQ5_K> =
        vec![unsafe { std::mem::zeroed() }; n_rows * nb];

    for row in 0..n_rows {
        for blk in 0..nb {
            let i = row * nb + blk; // linear index
            let block = &mut weights[row * nb + blk]; // standard row-major layout

            block.d = half::f16::to_bits(half::f16::from_f32(
                0.1 + (i as f32) * 0.01,
            ));
            block.dmin = half::f16::to_bits(half::f16::from_f32(
                0.01 + (i as f32) * 0.001,
            ));
            for j in 0..128 {
                block.qs[j] = ((j * 17 + i) % 256) as u8;
            }
            for j in 0..16 {
                block.qh[j] = ((j * 13 + i) % 256) as u8;
            }
            let scales: [u8; 12] = [
                0x15, 0x25, 0x35, 0x45, 0x55, 0x65, 0x75, 0x85, 0x95, 0xA5,
                0xB5, 0xC5,
            ];
            block.scales = scales;
        }
    }

    let mut input: Vec<BlockQ8_K> = Vec::new();
    for b in 0..nb {
        let mut block: BlockQ8_K = unsafe { std::mem::zeroed() };
        block.d = 0.5 + b as f32 * 0.1;
        for i in 0..256 {
            block.qs[i] = ((i as i8).wrapping_add((50 + b * 10) as i8));
        }
        for s in 0..16 {
            let mut sum: i16 = 0;
            for k in 0..16 {
                sum += block.qs[s * 16 + k] as i16;
            }
            block.bsums[s] = sum;
        }
        input.push(block);
    }

    // vec_dot: standard row-major access
    let mut output_vec_dot = vec![0.0f32; n_rows];
    for row in 0..n_rows {
        let row_weights = &weights[row * nb..(row + 1) * nb];
        unsafe {
            output_vec_dot[row] =
                vec_dot::vec_dot_q5_k_q8_k(row_weights, &input);
        }
    }

    // Repack: convert from row-major to BlockQ5_Kx8 interleaved format
    let stride = std::mem::size_of::<BlockQ5_K>() * nb; // bytes per row
    let weight_bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            n_rows * nb * std::mem::size_of::<BlockQ5_K>(),
        )
        .to_vec()
    };
    let size_x8 = std::mem::size_of::<BlockQ5_Kx8>();
    let mut repacked: Vec<u8> = vec![0u8; nb * size_x8];
    unsafe {
        repack::repack_q5_k_to_x8(
            &weight_bytes,
            stride,
            &mut repacked,
            n_rows,
            nb,
        );
    }
    let weights_x8: &[BlockQ5_Kx8] = unsafe {
        std::slice::from_raw_parts(repacked.as_ptr() as *const BlockQ5_Kx8, nb)
    };

    // GEMV
    let mut output_gemv = vec![0.0f32; n_rows];
    unsafe {
        gemv::gemv_q5_k_x8_neon(
            weights_x8,
            &input,
            &mut output_gemv,
            n_rows,
            nb * QK_K,
        );
    }

    println!("{:>6} {:>15} {:>15} {:>12}", "row", "vec_dot", "gemv_x8", "diff");
    let mut max_diff = 0.0f32;
    for row in 0..n_rows {
        let diff = (output_vec_dot[row] - output_gemv[row]).abs();
        max_diff = max_diff.max(diff);
        println!(
            "{:>6} {:>15.6} {:>15.6} {:>12.6}",
            row, output_vec_dot[row], output_gemv[row], diff
        );
    }
    println!("Max diff: {:.6}", max_diff);

    // Allow larger tolerance due to floating point precision differences
    assert!(
        max_diff < 1.0,
        "GEMV output differs from vec_dot: max_diff={}",
        max_diff
    );
}

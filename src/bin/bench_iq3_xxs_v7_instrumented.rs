// Instrumented IQ3_XXS v7 scalar - find which step is slow

use std::hint::black_box;
use std::time::Instant;

use infer_train::quant::types::{BlockIQ3XXS, BlockQ8K, QK_K};
use infer_train::quant::vec_dot::arm::fp::fp16_to_fp32;
use infer_train::quant::vec_dot::tables::IQ3XXS_GRID;

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

static KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

const N_BLOCKS: usize = 16;
const ITERATIONS: usize = 10000;

fn main() {
    // Generate test data
    let x = generate_iq3_xxs_blocks(N_BLOCKS);
    let y = generate_q8_k_blocks(N_BLOCKS);
    let n = N_BLOCKS * QK_K;

    println!(
        "=== IQ3_XXS v7 Instrumented ({} blocks, {} iterations) ===\n",
        N_BLOCKS, ITERATIONS
    );

    // Count operations
    let mut count_aux32_load = 0u64;
    let mut count_q3_read = 0u64;
    let mut count_grid_lookup = 0u64;
    let mut count_sign_extract = 0u64;
    let mut count_q8_read = 0u64;
    let mut count_multiply = 0u64;
    let mut count_accumulate = 0u64;

    // Time each step
    let mut t_aux32_load = 0u64;
    let mut t_q3_read = 0u64;
    let mut t_grid_lookup = 0u64;
    let mut t_sign_extract = 0u64;
    let mut t_q8_read = 0u64;
    let mut t_multiply = 0u64;
    let mut t_accumulate = 0u64;

    let grid_u32 = IQ3XXS_GRID.as_ptr();

    unsafe {
        for _ in 0..ITERATIONS {
            for i in 0..N_BLOCKS {
                let d = fp16_to_fp32(x[i].d) * y[i].d;
                black_box(d);

                let mut q3 = x[i].qs.as_ptr();
                let mut gas = q3.add(QK_K / 4);
                let mut q8 = y[i].qs.as_ptr();

                for _ in 0..QK_K / 32 {
                    // Step 1: aux32 load
                    let t1 = Instant::now();
                    let aux32 = std::ptr::read_unaligned(gas as *const u32);
                    gas = gas.add(4);
                    t_aux32_load += t1.elapsed().as_nanos() as u64;
                    count_aux32_load += 1;

                    let ls = 2 * (aux32 >> 28) + 1;

                    for l in 0..4 {
                        // Step 2: q3 read (grid indices)
                        let t2 = Instant::now();
                        let idx1 = *q3.add(2 * l);
                        let idx2 = *q3.add(2 * l + 1);
                        t_q3_read += t2.elapsed().as_nanos() as u64;
                        count_q3_read += 2;

                        // Step 3: grid lookup
                        let t3 = Instant::now();
                        let grid1 = grid_u32.add(idx1 as usize) as *const u8;
                        let grid2 = grid_u32.add(idx2 as usize) as *const u8;
                        t_grid_lookup += t3.elapsed().as_nanos() as u64;
                        count_grid_lookup += 2;

                        // Step 4: sign extract
                        let t4 = Instant::now();
                        let signs =
                            KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                        t_sign_extract += t4.elapsed().as_nanos() as u64;
                        count_sign_extract += 1;

                        for j in 0..4 {
                            // Step 5: q8 read
                            let t5 = Instant::now();
                            let q8_0 = *q8.add(j);
                            let q8_4 = *q8.add(j + 4);
                            t_q8_read += t5.elapsed().as_nanos() as u64;
                            count_q8_read += 2;

                            // Step 6: multiply
                            let t6 = Instant::now();
                            let sign1 = if signs & KMASK_IQ2XS[j] != 0 {
                                -1i32
                            } else {
                                1i32
                            };
                            let sign2 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                                -1i32
                            } else {
                                1i32
                            };
                            let g1 = *grid1.add(j) as i32;
                            let g2 = *grid2.add(j) as i32;
                            let prod1 = g1 * q8_0 as i32 * sign1;
                            let prod2 = g2 * q8_4 as i32 * sign2;
                            t_multiply += t6.elapsed().as_nanos() as u64;
                            count_multiply += 2;

                            // Step 7: accumulate
                            let t7 = Instant::now();
                            black_box(prod1 + prod2);
                            t_accumulate += t7.elapsed().as_nanos() as u64;
                            count_accumulate += 1;
                        }
                        q8 = q8.add(8);
                    }
                    q3 = q3.add(8);
                }
            }
        }
    }

    println!("Step breakdown:");
    println!(
        "  aux32 load:    {:7} ops, {:7} ns total, {:5.1} ns/op",
        count_aux32_load,
        t_aux32_load,
        t_aux32_load as f64 / count_aux32_load as f64
    );
    println!(
        "  q3 read:       {:7} ops, {:7} ns total, {:5.1} ns/op",
        count_q3_read,
        t_q3_read,
        t_q3_read as f64 / count_q3_read as f64
    );
    println!(
        "  grid lookup:   {:7} ops, {:7} ns total, {:5.1} ns/op",
        count_grid_lookup,
        t_grid_lookup,
        t_grid_lookup as f64 / count_grid_lookup as f64
    );
    println!(
        "  sign extract:  {:7} ops, {:7} ns total, {:5.1} ns/op",
        count_sign_extract,
        t_sign_extract,
        t_sign_extract as f64 / count_sign_extract as f64
    );
    println!(
        "  q8 read:       {:7} ops, {:7} ns total, {:5.1} ns/op",
        count_q8_read,
        t_q8_read,
        t_q8_read as f64 / count_q8_read as f64
    );
    println!(
        "  multiply:      {:7} ops, {:7} ns total, {:5.1} ns/op",
        count_multiply,
        t_multiply,
        t_multiply as f64 / count_multiply as f64
    );
    println!(
        "  accumulate:    {:7} ops, {:7} ns total, {:5.1} ns/op",
        count_accumulate,
        t_accumulate,
        t_accumulate as f64 / count_accumulate as f64
    );

    let total_ops = count_aux32_load
        + count_q3_read
        + count_grid_lookup
        + count_sign_extract
        + count_q8_read
        + count_multiply
        + count_accumulate;
    let total_ns = t_aux32_load
        + t_q3_read
        + t_grid_lookup
        + t_sign_extract
        + t_q8_read
        + t_multiply
        + t_accumulate;

    println!(
        "\nTotal: {} ops, {} ns, {:.1} ns/op",
        total_ops,
        total_ns,
        total_ns as f64 / total_ops as f64
    );
}

fn generate_iq3_xxs_blocks(count: usize) -> Vec<BlockIQ3XXS> {
    use rand::{Rng, SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(42);
    let mut blocks = Vec::with_capacity(count);
    for _ in 0..count {
        let mut qs = [0u8; 96];
        for j in 0..96 {
            qs[j] = rng.r#gen();
        }
        let d = half::f16::from_f32(rng.gen_range(0.1..2.0)).to_bits();
        blocks.push(BlockIQ3XXS { qs, d });
    }
    blocks
}

fn generate_q8_k_blocks(count: usize) -> Vec<BlockQ8K> {
    use rand::{Rng, SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(42);
    let mut blocks = Vec::with_capacity(count);
    for _ in 0..count {
        let mut qs = [0i8; QK_K];
        for j in 0..QK_K {
            qs[j] = rng.gen_range(-128..127);
        }
        let bsums = [0i16; 16];
        let d = rng.gen_range(0.1..2.0);
        blocks.push(BlockQ8K { d, qs, bsums });
    }
    blocks
}

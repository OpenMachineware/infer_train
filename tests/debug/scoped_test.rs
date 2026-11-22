use infer_train::backend::arm::vec_dot::*;
use infer_train::types::*;
use std::time::Instant;

fn main() {
    let in_features = 1024;
    let out_features = 2048;
    let n_blocks = in_features / 256;

    let block_size = std::mem::size_of::<BlockQ5_K>();
    let total_weight_bytes = out_features * n_blocks * block_size;
    let weights_bytes: Vec<u8> = vec![0x88u8; total_weight_bytes];

    let input: Vec<BlockQ8_K> =
        vec![BlockQ8_K { d: 1.0, qs: [1; 256], bsums: [0; 16] }; n_blocks];
    let mut output = vec![0.0f32; out_features];

    let col_stride = n_blocks * block_size;

    // Single-threaded baseline
    let iters = 50;
    let start = Instant::now();
    for _ in 0..iters {
        for row in 0..out_features {
            let row_offset = row * col_stride;
            let weight_row: &[BlockQ5_K] = unsafe {
                std::slice::from_raw_parts(
                    weights_bytes.as_ptr().add(row_offset) as *const BlockQ5_K,
                    n_blocks,
                )
            };
            output[row] = vec_dot_q5_k_q8_k(weight_row, &input);
        }
    }
    let single_time = start.elapsed() / iters;

    // Multi-threaded using rayon
    use rayon::prelude::*;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();

    let start = Instant::now();
    for _ in 0..iters {
        pool.install(|| {
            output.par_iter_mut().enumerate().for_each(|(row, out)| {
                let row_offset = row * col_stride;
                unsafe {
                    let weight_row = std::slice::from_raw_parts(
                        weights_bytes.as_ptr().add(row_offset)
                            as *const BlockQ5_K,
                        n_blocks,
                    );
                    let input_row =
                        std::slice::from_raw_parts(input.as_ptr(), n_blocks);
                    *out = vec_dot_q5_k_q8_k(weight_row, input_row);
                }
            });
        });
    }
    let multi_time = start.elapsed() / iters;

    let flops = 2 * out_features * in_features;
    println!(
        "Single-threaded: {:?} ({:.1} GFLOPS)",
        single_time,
        flops as f64 / single_time.as_secs_f64() / 1e9
    );
    println!(
        "Multi-threaded (8 threads via rayon): {:?} ({:.1} GFLOPS)",
        multi_time,
        flops as f64 / multi_time.as_secs_f64() / 1e9
    );
    println!(
        "Speedup: {:.2}x",
        single_time.as_secs_f64() / multi_time.as_secs_f64()
    );
}

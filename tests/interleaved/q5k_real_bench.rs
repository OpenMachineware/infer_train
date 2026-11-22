use infer_train::backend::arm::{gemv, quant, repack};
use infer_train::model::GgufFile;
use infer_train::types::*;
use std::time::Instant;

fn main() {
    let model_path = "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf";
    println!("Loading model from: {}", model_path);
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    // Find a Q5_K tensor with proper dimensions
    let tensor = gguf
        .tensors
        .iter()
        .find(|t| {
            matches!(t.quant_type, QuantType::Q5_K)
                && (t.dims[1] as usize) % QK_K == 0
        })
        .expect("No suitable Q5_K tensor found");

    println!("Tensor: {}", tensor.name);
    println!("  Shape: {:?}", tensor.dims);

    let weight_data =
        gguf.get_tensor_data(&tensor.name).expect("Failed to read tensor");

    let out_features = tensor.dims[0] as usize;
    let in_features = tensor.dims[1] as usize;
    let n_blocks = in_features / QK_K;
    let stride = tensor.strides[0] as usize;

    println!("  out_features: {}, in_features: {}", out_features, in_features);
    println!("  n_blocks: {}, stride: {} bytes\n", n_blocks, stride);

    // Create input
    let mut input_f32: Vec<f32> =
        (0..in_features).map(|i| (i as f32 % 100.0 - 50.0) / 100.0).collect();
    let mut input_q8k: Vec<BlockQ8_K> =
        vec![unsafe { std::mem::zeroed() }; n_blocks];
    quant::quantize_q8_k(&input_f32, &mut input_q8k);

    let mut output_std = vec![0.0f32; out_features];
    let mut output_x8 = vec![0.0f32; out_features];

    // Repack weights
    println!("Repacking weights...");
    let start = Instant::now();
    let n_row_groups = out_features / 8;
    let mut weights_packed: Vec<u8> =
        vec![0u8; n_row_groups * n_blocks * std::mem::size_of::<BlockQ5_Kx8>()];
    repack::repack_q5_k_to_x8(
        &weight_data,
        stride,
        &mut weights_packed,
        out_features,
        n_blocks,
    );
    let repack_time = start.elapsed();
    println!("Repack time: {:.2?}", repack_time);

    let weights_x8: &[BlockQ5_Kx8] = unsafe {
        std::slice::from_raw_parts(
            weights_packed.as_ptr() as *const BlockQ5_Kx8,
            n_row_groups * n_blocks,
        )
    };

    // Warmup
    for _ in 0..5 {
        unsafe {
            gemv::gemv_q5_k_transposed(
                &weight_data,
                stride,
                &input_q8k,
                &mut output_std,
                out_features,
                in_features,
                1,
            );
            gemv::gemv_q5_k_x8_neon(
                weights_x8,
                &input_q8k,
                &mut output_x8,
                out_features,
                in_features,
            );
        }
    }

    // Benchmark
    let iters = 50;
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            gemv::gemv_q5_k_transposed(
                &weight_data,
                stride,
                &input_q8k,
                &mut output_std,
                out_features,
                in_features,
                1,
            );
        }
    }
    let std_time = start.elapsed();
    let std_avg = std_time.as_micros() as f64 / iters as f64;

    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            gemv::gemv_q5_k_x8_neon(
                weights_x8,
                &input_q8k,
                &mut output_x8,
                out_features,
                in_features,
            );
        }
    }
    let x8_time = start.elapsed();
    let x8_avg = x8_time.as_micros() as f64 / iters as f64;

    // Verify
    let max_diff = (0..out_features)
        .map(|i| (output_std[i] - output_x8[i]).abs())
        .fold(0.0f32, f32::max);

    let flops = 2 * out_features * in_features;
    let total_bytes = weight_data.len()
        + input_q8k.len() * std::mem::size_of::<BlockQ8_K>()
        + out_features * 4;

    println!("\n=== Results ===");
    println!(
        "Standard GEMV:   {:.2} us ({:.1} GFLOPS, {:.1} GB/s)",
        std_avg,
        flops as f64 / std_avg / 1000.0,
        total_bytes as f64 / std_avg / 1000.0
    );
    println!(
        "Interleaved GEMV: {:.2} us ({:.1} GFLOPS, {:.1} GB/s)",
        x8_avg,
        flops as f64 / x8_avg / 1000.0,
        total_bytes as f64 / x8_avg / 1000.0
    );
    println!("Speedup: {:.2}x", std_avg / x8_avg);
    println!("Max diff: {:.6}", max_diff);
}

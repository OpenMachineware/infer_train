use infer_train::backend::arm::{gemv, quant, vec_dot};
use infer_train::model::GgufFile;
use infer_train::types::*;
use std::time::Instant;

fn main() {
    let path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(path).unwrap();

    // Get a Q5_K weight tensor for testing
    let tensor =
        gguf.tensors.iter().find(|t| t.name == "blk.0.attn_q.weight").unwrap();

    let weight_data = gguf.get_tensor_data("blk.0.attn_q.weight").unwrap();
    let in_features = tensor.dims[0] as usize; // 1024
    let out_features = tensor.dims[1] as usize; // 2048
    let col_stride = tensor.strides[1] as usize; // 704 bytes

    println!("=== GEMV Micro-benchmark ===");
    println!("Weight: {} ({}x{})", tensor.name, in_features, out_features);
    println!("Quant type: {:?}", tensor.quant_type);
    println!("Weight bytes: {} MB", weight_data.len() as f64 / 1024.0 / 1024.0);
    println!("col_stride: {} bytes", col_stride);

    // Create test input
    let n_blocks = in_features / 256;
    let mut input_q8k: Vec<BlockQ8_K> =
        vec![unsafe { std::mem::zeroed() }; n_blocks];

    // Create random f32 input
    let input_f32: Vec<f32> =
        (0..in_features).map(|i| (i as f32 * 0.01).sin()).collect();
    quant::quantize_q8_k(&input_f32, &mut input_q8k);

    let mut output = vec![0.0f32; out_features];

    // Warmup
    for _ in 0..3 {
        gemv::gemv_q5_k_transposed(
            weight_data,
            col_stride,
            &input_q8k,
            &mut output,
            out_features,
            in_features,
            1, // single thread
        );
    }

    // Measure single-thread
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        gemv::gemv_q5_k_transposed(
            weight_data,
            col_stride,
            &input_q8k,
            &mut output,
            out_features,
            in_features,
            1,
        );
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() as f64 / iterations as f64;

    println!("\n=== Results (single-thread) ===");
    println!("Average GEMV time: {:.2} μs", avg_us);
    println!(
        "Weight bytes per GEMV: {} MB",
        weight_data.len() as f64 / 1024.0 / 1024.0
    );

    // Calculate effective memory bandwidth
    let bytes_per_gv = weight_data.len() as f64;
    let bandwidth_gbps = bytes_per_gv / (avg_us * 1e-6) / 1e9;
    println!("Effective bandwidth: {:.2} GB/s", bandwidth_gbps);
    println!(
        "Memory bandwidth utilization: {:.2}%",
        bandwidth_gbps / 400.0 * 100.0
    );

    // Calculate theoretical time (if memory bandwidth were the only bottleneck)
    let theoretical_time_us = bytes_per_gv / 400e9 * 1e6;
    println!(
        "\nTheoretical min time (at 400 GB/s): {:.2} μs",
        theoretical_time_us
    );
    println!("Actual/Theoretical ratio: {:.1}x", avg_us / theoretical_time_us);

    // Now measure vec_dot alone
    println!("\n=== vec_dot Micro-benchmark ===");

    // Get a single column of weights
    let weight_col: &[BlockQ5_K] = unsafe {
        std::slice::from_raw_parts(
            weight_data.as_ptr() as *const BlockQ5_K,
            n_blocks,
        )
    };

    // Warmup
    for _ in 0..100 {
        let _ = vec_dot::vec_dot_q5_k_q8_k(weight_col, &input_q8k);
    }

    // Measure
    let iterations = 10000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = vec_dot::vec_dot_q5_k_q8_k(weight_col, &input_q8k);
    }
    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() as f64 / iterations as f64;

    println!("Average vec_dot time: {:.2} ns", avg_ns);
    println!(
        "Bytes per vec_dot: {} bytes",
        n_blocks * std::mem::size_of::<BlockQ5_K>()
    );

    // Check if this matches GEMV time
    let expected_gv_time_us = avg_ns * out_features as f64 / 1000.0;
    println!(
        "Expected GEMV time (vec_dot × {}): {:.2} μs",
        out_features, expected_gv_time_us
    );
}

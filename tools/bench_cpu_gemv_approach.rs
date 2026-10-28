use infer_train::inference::{Device, InferenceContext};
use infer_train::quant::types::{BlockQ8K, QK_K};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    // Get layer 0 w1 weight (gate projection)
    let layer = &ctx.model.layers[0];
    let w1 = layer.w1.as_ref().unwrap();
    let intermediate_size = w1.shape[1]; // 3072
    let hidden_size = w1.shape[0]; // 1024

    println!("=== CPU GEMV Performance Test ===");
    println!("Weight shape: {:?}", w1.shape);
    println!("Weight dtype: {:?}", w1.dtype);
    println!("Hidden size: {}", hidden_size);
    println!("Intermediate size: {}\n", intermediate_size);

    // Create a test input vector
    let hidden: Vec<f32> =
        (0..hidden_size).map(|i| (i as f32 * 0.001).sin()).collect();

    // Test current GEMV implementation
    let iterations = 100;
    let mut total_time = std::time::Duration::ZERO;

    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let result =
            ctx.model.gemv_impl(&hidden, w1, intermediate_size).unwrap();
        total_time += start.elapsed();

        // Prevent optimization
        std::hint::black_box(result);
    }

    let avg_time = total_time / iterations;
    let flops = 2 * hidden_size * intermediate_size; // multiply-add = 2 flops
    let gflops = flops as f64 / avg_time.as_secs_f64() / 1e9;

    println!("Current implementation (dequant all then parallel):");
    println!("  Avg time: {:?}", avg_time);
    println!("  Performance: {:.2} GFLOPS\n", gflops);

    // Calculate expected time based on llama.cpp
    // llama.cpp on 0.6B model: 98.6 t/s, each token has ~500ms forward pass
    // FFN has 3 GEMV with total ~9M multiply-adds
    // Expected GEMV time: ~500ms / 28 layers / 3 GEMV ≈ 6ms per GEMV
    // But we're seeing 11.13ms for 3 GEMV, so ~3.7ms per GEMV

    // Estimate improvement needed
    let current_gemm_time = avg_time.as_millis();
    let target_gemm_time = 1.0; // 1ms per GEMV to achieve 100 t/s
    let speedup_needed = current_gemm_time as f64 / target_gemm_time;

    println!("Target GEMV time: {:.1}ms", target_gemm_time);
    println!("Speedup needed: {:.1}x\n", speedup_needed);

    // Memory bandwidth analysis
    let weight_bytes = w1.bytes().len();
    let input_bytes = hidden_size * 4;
    let output_bytes = intermediate_size * 4;
    let total_bytes = weight_bytes + input_bytes + output_bytes;

    let bandwidth_gbps = total_bytes as f64 / avg_time.as_secs_f64() / 1e9;

    println!("Memory bandwidth:");
    println!("  Weight: {} MB", weight_bytes as f64 / 1e6);
    println!("  Input: {} KB", input_bytes as f64 / 1e3);
    println!("  Output: {} KB", output_bytes as f64 / 1e3);
    println!("  Total bandwidth: {:.2} GB/s", bandwidth_gbps);
}

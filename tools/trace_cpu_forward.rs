use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
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

    println!("=== Quantization Overhead Analysis ===\n");

    // Test input
    let hidden: Vec<f32> =
        (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();

    // Benchmark quantization
    let iterations = 100;
    let mut total_time = std::time::Duration::ZERO;

    for _ in 0..iterations {
        let start = std::time::Instant::now();

        // Quantize to Q8_K
        let nb = hidden.len() / QK_K;
        let mut x_q8k =
            vec![BlockQ8K { d: 0.0, qs: [0; 256], bsums: [0; 16] }; nb];

        #[cfg(target_arch = "aarch64")]
        infer_train::quant::kv_cache::quantize_row_q8_k_neon(
            &hidden, &mut x_q8k,
        );

        total_time += start.elapsed();
        std::hint::black_box(x_q8k);
    }

    let avg_time = total_time / iterations;
    println!("Quantization time (1024 elements -> Q8_K): {:?}", avg_time);

    // Calculate how many times we quantize per forward pass
    println!("\n=== Quantization Count per Layer ===");
    println!("QKV projections: 3 GEMV → 3 quantizations");
    println!("Output projection: 1 GEMV → 1 quantization");
    println!("FFN: 3 GEMV → 3 quantizations");
    println!("Total per layer: 7 quantizations");
    println!("Total for 28 layers: {} quantizations", 28 * 7);

    let total_quant_time_per_pass = avg_time * (28 * 7) as u32;
    println!(
        "\nTotal quantization time per forward pass: {:?}",
        total_quant_time_per_pass
    );

    // Compare with actual forward pass time
    println!("\n=== Impact Analysis ===");
    let forward_pass_time = std::time::Duration::from_millis(250); // ~4 t/s = 250ms per token
    println!("Actual forward pass time: {:?}", forward_pass_time);
    println!(
        "Quantization overhead: {:.1}%",
        total_quant_time_per_pass.as_secs_f64()
            / forward_pass_time.as_secs_f64()
            * 100.0
    );

    println!("\n=== llama.cpp Approach ===");
    println!("llama.cpp quantizes input ONCE per forward pass");
    println!("Then reuses the quantized input for all GEMV calls");
    println!("This saves {} quantizations!", 28 * 7 - 1);
    println!("Time saved: {:?}", total_quant_time_per_pass - avg_time);
}

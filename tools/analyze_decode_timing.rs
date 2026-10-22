// Detailed decode timing analysis
// Measures each step of the decode process precisely

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::{Device, InferenceContext};
use std::time::Instant;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    // Load model
    let t0 = Instant::now();
    let mut gpu_ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load GPU model");
    let load_time = t0.elapsed();

    println!("Model load: {:.2} ms", load_time.as_secs_f64() * 1000.0);
    println!("Config: hidden={}, heads={}",
        gpu_ctx.config.hidden_size, gpu_ctx.config.n_heads);

    // Warmup
    let _ = gpu_ctx.forward_gpu_batched(&[1]).unwrap();
    for _ in 0..3 {
        let _ = gpu_ctx.forward_gpu_batched(&[0]).unwrap();
    }

    // Detailed timing for 5 decode tokens
    println!("\n=== Detailed Decode Timing ===");

    for i in 0..5 {
        println!("\n--- Token {} ---", i + 1);
        let token_start = Instant::now();

        let logits = gpu_ctx.forward_gpu_batched(&[i as u32]).unwrap();

        let total_time = token_start.elapsed();

        // Sampling timing
        let sample_start = Instant::now();
        let _next_token = logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32);
        let sample_time = sample_start.elapsed();

        println!("  Total forward: {:.2} ms", total_time.as_secs_f64() * 1000.0);
        println!("  Sampling: {:.2} µs", sample_time.as_secs_f64() * 1_000_000.0);
        println!("  Throughput: {:.2} t/s", 1000.0 / total_time.as_secs_f64() / 1000.0);
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This tool requires Metal support");
}

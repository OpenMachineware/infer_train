// Profile decode operations in detail
// Measures dispatch count and bandwidth utilization

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::{Device, InferenceContext};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Profiling decode: {}", model_path);

    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load GPU model");

    // Prefill
    let prompt_tokens: Vec<u32> = vec![1];
    let _logits = ctx.forward_gpu_batched(&prompt_tokens).expect("Prefill failed");

    // Decode with profiling
    println!("\nDecode profiling (10 tokens):");
    for i in 0..10 {
        let start = std::time::Instant::now();
        let logits = ctx.forward_gpu_batched(&[1]).expect("Decode failed");
        let elapsed = start.elapsed();
        
        println!("Token {}: {:?} ({:.1} t/s)", i, elapsed, 1000.0 / elapsed.as_millis() as f64);
    }

    // Print operator statistics
    println!("\nOperator dispatch statistics:");
    if let Some(ref stats) = ctx.gpu_stats() {
        for (op, count) in stats.operator_counts() {
            println!("  {}: {} dispatches", op, count);
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    eprintln!("This tool requires Metal support");
}

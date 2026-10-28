use infer_train::inference::{Device, InferenceContext};
use infer_train::ops::sampling::SamplingStrategy;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== CPU Inference Benchmark (0.6B Model) ===");
    println!("Model: {}", model_path);

    // Load model and create context
    let start = std::time::Instant::now();
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    println!("Load time: {:?}", start.elapsed());

    println!("\n=== Architecture Comparison ===");
    println!("Our implementation:");
    println!("  - Uses Rayon work-stealing thread pool");
    println!("  - Each GEMV call: quantize input, parallel vec_dot");
    println!("  - No persistent buffers between forward passes");
    println!("  - Block-tiling: not implemented");
    println!();
    println!("llama.cpp:");
    println!("  - Custom thread pool with barrier sync");
    println!("  - Pre-quantizes input once per forward pass");
    println!("  - Work buffer pre-allocated in graph plan");
    println!("  - Block-tiling: 16x16 for cache efficiency");
    println!("  - Optional llamafile SGEMM optimization");
    println!();

    // Check Rayon thread count
    let num_threads = rayon::current_num_threads();
    println!("Rayon threads: {}", num_threads);
    println!();

    // Use simple token IDs for testing (BOS + some common tokens)
    let prompt_tokens: Vec<u32> = vec![
        ctx.config.bos_token_id,
        1,
        2,
        3,
        4,
        5, // Simple test tokens
    ];
    println!("Prompt tokens: {}", prompt_tokens.len());

    // Warmup
    let _ = ctx.generate(&prompt_tokens, 5, &SamplingStrategy::Greedy);

    // Benchmark decode
    println!("\n=== Decode Benchmark (20 tokens) ===");
    let start = std::time::Instant::now();
    let output_tokens =
        ctx.generate(&prompt_tokens, 20, &SamplingStrategy::Greedy);
    let elapsed = start.elapsed();

    let tps = 20.0 / elapsed.as_secs_f64();
    println!("Tokens: {}", output_tokens.len());
    println!("Time: {:?}", elapsed);
    println!("Speed: {:.2} t/s", tps);

    println!("\n=== Performance Analysis ===");
    let model_size_mb = 387.0;
    let tokens_per_sec = tps;
    let bandwidth_gbps = model_size_mb * tokens_per_sec / 1000.0;
    println!("Effective memory bandwidth: {:.2} GB/s", bandwidth_gbps);
    println!("M1 Pro theoretical bandwidth: ~400 GB/s");
    println!("Bandwidth utilization: {:.2}%", bandwidth_gbps / 400.0 * 100.0);
    println!();
    println!("llama.cpp achieves: ~97 t/s");
    println!("llama.cpp bandwidth: ~37.5 GB/s (9.4% utilization)");
    println!("Our gap: {:.1}x", 97.0 / tps);

    println!(
        "\nOutput token IDs: {:?}",
        &output_tokens[..std::cmp::min(10, output_tokens.len())]
    );
}

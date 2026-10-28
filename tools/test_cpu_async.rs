use infer_train::inference::{CpuAsyncExecutor, Device, InferenceContext};
use infer_train::ops::sampling::SamplingStrategy;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== CPU Async Pipeline Benchmark ===");
    println!("Model: {}", model_path);

    // Load model
    let start = std::time::Instant::now();
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    println!("Load time: {:?}", start.elapsed());

    // Create async executor
    let mut executor = CpuAsyncExecutor::new(ctx, 100);

    // Test prompt
    let prompt_tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    println!("\nPrompt tokens: {}", prompt_tokens.len());

    // Warmup
    let _ = executor.generate(&prompt_tokens, 5, &SamplingStrategy::Greedy);

    // Benchmark
    println!("\n=== Decode Benchmark (20 tokens) ===");
    let start = std::time::Instant::now();
    let output_tokens =
        executor.generate(&prompt_tokens, 20, &SamplingStrategy::Greedy);
    let elapsed = start.elapsed();

    let tps = 20.0 / elapsed.as_secs_f64();
    println!("Tokens: {}", output_tokens.len());
    println!("Time: {:?}", elapsed);
    println!("Speed: {:.2} t/s", tps);

    println!("\nComparison:");
    println!("  Before (single thread): 0.86 t/s");
    println!("  llama.cpp (multi thread): 98.6 t/s");
    println!("  Our result: {:.2} t/s", tps);
}

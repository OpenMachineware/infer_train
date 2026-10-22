// Comprehensive end-to-end timing analysis
// Measures every step from model load to decode

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::{Device, InferenceContext};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    println!("=== Full Pipeline Timing Analysis ===\n");

    // ===== PHASE 1: Model Loading =====
    println!("--- Phase 1: Model Loading ---");
    let load_start = Instant::now();

    let mut gpu_ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load GPU model");

    let load_time = load_start.elapsed();
    println!("Total load time: {:.2} ms", load_time.as_secs_f64() * 1000.0);

    let config = &gpu_ctx.config;
    println!("\nModel config:");
    println!("  hidden_size: {}", config.hidden_size);
    println!("  n_heads: {}", config.n_heads);
    println!("  n_kv_heads: {}", config.n_kv_heads);
    println!("  head_dim: {}", config.head_dim);
    println!("  n_layers: {}", config.n_layers);
    println!("  vocab_size: {}", config.vocab_size);

    // ===== PHASE 2: Prefill (first token) =====
    println!("\n--- Phase 2: Prefill ---");
    let prefill_tokens: Vec<u32> = vec![1]; // Single token prefill

    let prefill_start = Instant::now();
    let logits = gpu_ctx.forward_gpu_batched(&prefill_tokens)
        .expect("Prefill failed");
    let prefill_time = prefill_start.elapsed();

    println!("Prefill time: {:.2} ms", prefill_time.as_secs_f64() * 1000.0);
    println!("Logits count: {}", logits.len());

    // Check logits quality
    let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
    println!("Logits: min={:.4}, max={:.4}, nan={}", min, max, nan_count);

    // Sample first token
    let sample_start = Instant::now();
    let mut next_token = logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    let sample_time = sample_start.elapsed();
    println!("Sampling time: {:.2} µs", sample_time.as_secs_f64() * 1_000_000.0);
    println!("First token: {}", next_token);

    // ===== PHASE 3: Decode (subsequent tokens) =====
    println!("\n--- Phase 3: Decode (10 tokens) ---");

    let mut decode_times: Vec<f64> = Vec::new();
    let mut sample_times: Vec<f64> = Vec::new();

    for i in 0..10 {
        let decode_start = Instant::now();
        let logits = gpu_ctx.forward_gpu_batched(&[next_token])
            .expect("Decode failed");
        let decode_time = decode_start.elapsed();

        let sample_start = Instant::now();
        next_token = logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        let sample_time = sample_start.elapsed();

        decode_times.push(decode_time.as_secs_f64() * 1000.0);
        sample_times.push(sample_time.as_secs_f64() * 1_000_000.0);

        if i == 0 {
            println!("Token 1: {:.2} ms (first decode)", decode_times[0]);
        }
    }

    // Calculate statistics
    let avg_decode: f64 = decode_times.iter().sum::<f64>() / decode_times.len() as f64;
    let min_decode = decode_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_decode = decode_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let avg_sample: f64 = sample_times.iter().sum::<f64>() / sample_times.len() as f64;

    println!("\nDecode statistics:");
    println!("  Average: {:.2} ms/token", avg_decode);
    println!("  Range: {:.2} - {:.2} ms", min_decode, max_decode);
    println!("  Throughput: {:.2} t/s", 1000.0 / avg_decode);
    println!("  Sampling: {:.2} µs/token", avg_sample);

    // ===== Summary =====
    println!("\n=== Performance Summary ===");
    println!("Model load: {:.2} ms", load_time.as_secs_f64() * 1000.0);
    println!("Prefill: {:.2} ms", prefill_time.as_secs_f64() * 1000.0);
    println!("Decode: {:.2} ms/token ({:.2} t/s)", avg_decode, 1000.0 / avg_decode);
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This tool requires Metal support");
}

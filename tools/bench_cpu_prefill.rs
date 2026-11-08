// CPU prefill benchmark
// Tests batch prefill performance

use infer_train::cpu::CpuWorkBuffers;
use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use infer_train::model::LlamaModel;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <batch_size>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let batch_size: usize = args[2].parse().unwrap_or(64);

    println!("=== CPU Prefill Benchmark ===\n");
    println!("Model: {}", model_path);
    println!("Batch size: {}", batch_size);

    // Load model
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    println!("Model loaded successfully");

    let hidden_size = ctx.config.hidden_size;
    let intermediate_size = ctx.config.intermediate_size;
    let n_heads = ctx.config.n_heads;
    let n_kv_heads = ctx.config.n_kv_heads;
    let head_dim = ctx.config.head_dim;
    let vocab_size = ctx.config.vocab_size;
    let n_layers = ctx.model.layers.len();

    println!(
        "Config: hidden={}, intermediate={}, layers={}",
        hidden_size, intermediate_size, n_layers
    );

    // Test scheduler thread allocation
    println!("\n=== Scheduler Thread Allocation ===");
    let scheduler = infer_train::cpu::get_scheduler();

    let m_ffn = intermediate_size;
    let k = hidden_size;

    let threads_ffn = scheduler.optimal_threads(m_ffn, k, batch_size);
    println!(
        "FFN gate (m={}, k={}, n={}): {} threads",
        m_ffn, k, batch_size, threads_ffn
    );

    let m_qkv = n_heads * head_dim * 3;
    let threads_qkv = scheduler.optimal_threads(m_qkv, k, batch_size);
    println!(
        "QKV (m={}, k={}, n={}): {} threads",
        m_qkv, k, batch_size, threads_qkv
    );

    let m_out = hidden_size;
    let threads_out = scheduler.optimal_threads(m_out, k, batch_size);
    println!(
        "Output (m={}, k={}, n={}): {} threads",
        m_out, k, batch_size, threads_out
    );

    // Create tokens
    let tokens: Vec<u32> = (1..=batch_size as u32).collect();

    // Initialize KV cache
    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Initialize buffers
    let mut buffers = CpuWorkBuffers::new(
        hidden_size,
        intermediate_size,
        n_heads,
        n_kv_heads,
        head_dim,
        ctx.config.key_length,
        ctx.config.value_length,
        vocab_size,
        0,
        2048,
        n_layers,
    );
    buffers.init_norm_weights(&ctx.model);

    // Benchmark
    println!("\n=== Benchmark ===");
    let start = std::time::Instant::now();
    let _ = ctx.model.forward_pooled(&tokens, &mut kv_cache, &mut buffers);
    let elapsed = start.elapsed();

    let tokens_per_sec = batch_size as f64 / elapsed.as_secs_f64();
    println!(
        "Prefill {} tokens: {:?} ({:.2} t/s)",
        batch_size, elapsed, tokens_per_sec
    );
}

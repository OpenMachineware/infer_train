// Test GGUF parsing and model layer
// Run: cargo run --release --bin test_gguf -- models/qwen-7b-q4_0.gguf

use infer_train::model::{GgufFile, ModelConfig, WeightStore};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <gguf-file>", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    println!("Loading GGUF file: {}", path);

    // Test GGUF parsing
    let gguf = match GgufFile::open(path) {
        Ok(gguf) => gguf,
        Err(e) => {
            eprintln!("✗ Error: {}", e);
            std::process::exit(1);
        }
    };

    println!("✓ GGUF version: {}", gguf.version);
    println!("✓ Tensor count: {}", gguf.tensor_count);
    println!("✓ Metadata count: {}", gguf.metadata_kv_count);
    println!();

    // Test ModelConfig
    let config = ModelConfig::from_gguf(&gguf);
    println!("=== Model Config ===");
    println!("  Architecture: {}", config.architecture);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Intermediate size: {}", config.intermediate_size);
    println!("  Layers: {}", config.n_layers);
    println!("  Heads: {} (KV: {})", config.n_heads, config.n_kv_heads);
    println!("  Vocab size: {}", config.vocab_size);
    println!(
        "  RoPE dim: {} (base: {})",
        config.rope_dim, config.rope_freq_base
    );
    println!("  RMS norm eps: {}", config.rms_norm_eps);
    println!("  Max position: {}", config.max_position_embeddings);
    println!("  Head dim: {}", config.head_dim());
    println!();

    // Test WeightStore
    println!("=== Loading Weights ===");
    match WeightStore::from_gguf(path) {
        Ok(store) => {
            println!("✓ Loaded {} tensors", store.tensor_count());
            println!(
                "✓ Total size: {:.2} MB",
                store.total_bytes() as f64 / 1024.0 / 1024.0
            );

            // Test tensor access
            println!("\n=== Sample Tensors ===");
            for name in store.tensor_names().iter().take(5) {
                if let Some(tensor) = store.get(name) {
                    println!(
                        "  {} {:?} {:?} ({} bytes)",
                        tensor.name,
                        tensor.shape,
                        tensor.dtype,
                        tensor.bytes().len()
                    );
                }
            }
        }
        Err(e) => {
            eprintln!("✗ WeightStore error: {}", e);
        }
    }
}

// Compare layer-by-layer output with llama.cpp
// This requires building llama.cpp with debug output first

use infer_train::inference::{ForwardExecutor, SimpleModel};
use infer_train::model::GgufFile;
use infer_train::tokenizer::create_tokenizer;

fn main() {
    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    // Create tokenizer
    let meta = gguf.tokenizer_metadata().expect("Failed to get tokenizer metadata");
    let tokenizer = create_tokenizer(meta).expect("Failed to create tokenizer");

    // Create model
    let model = SimpleModel::load(model_path, tokenizer).expect("Failed to create model");

    // Create executor
    let mut executor = ForwardExecutor::new(model);

    // Process the first token
    let token_id = 27u32;  // <|im_start|>

    println!("\n=== Processing token {} ===", token_id);
    let _logits = executor.forward(token_id).expect("Forward failed");

    println!("\n=== Summary ===");
    println!("First token processed, seq_pos = {}", executor.seq_pos);

    // Check KV cache
    println!("\n=== KV Cache ===");
    for layer_idx in 0..3 {
        let k_cache = &executor.kv_cache_k[layer_idx];
        let v_cache = &executor.kv_cache_v[layer_idx];

        let k_min = k_cache.iter().cloned().fold(f32::INFINITY, f32::min);
        let k_max = k_cache.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let v_min = v_cache.iter().cloned().fold(f32::INFINITY, f32::min);
        let v_max = v_cache.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("L{} K cache: min={:.4} max={:.4}", layer_idx, k_min, k_max);
        println!("L{} V cache: min={:.4} max={:.4}", layer_idx, v_min, v_max);
    }
}

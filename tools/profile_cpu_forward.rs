use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== CPU Forward Profiling ===");

    // Load model
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    println!("Model loaded: {} layers", ctx.layer_count());
    println!("Hidden size: {}", ctx.config.hidden_size);
    println!("Vocab size: {}", ctx.config.vocab_size);
    println!("BOS token: {}", ctx.config.bos_token_id);

    // Test tokens
    let prompt_tokens: Vec<u32> = vec![ctx.config.bos_token_id, 1, 2, 3, 4, 5];
    println!("Test tokens: {:?}", prompt_tokens);

    // Create KV cache
    let mut kv_cache = KVCache::new(&ctx.config, 100, KVCacheFormat::F32);

    // Profile prefill
    println!("\n=== Prefill (6 tokens) ===");
    let start = std::time::Instant::now();
    let logits = ctx.forward(&prompt_tokens, &mut kv_cache);
    let prefill_time = start.elapsed();
    println!("Prefill time: {:?}", prefill_time);
    match &logits {
        Some(l) => {
            println!("Logits: {} values", l.len());
            if l.len() > 0 {
                println!("First 5 logits: {:?}", &l[..5.min(l.len())]);
                println!(
                    "Max logit: {}",
                    l.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                );
            }
        }
        None => println!("Logits: None (forward returned None)"),
    }

    // Profile decode (single token)
    println!("\n=== Decode (1 token) ===");
    let start = std::time::Instant::now();
    let logits2 = ctx.forward(&[1], &mut kv_cache);
    let decode_time = start.elapsed();
    println!("Decode time: {:?}", decode_time);
    match &logits2 {
        Some(l) => {
            println!("Logits: {} values", l.len());
        }
        None => println!("Logits: None (forward returned None)"),
    }

    // Show KV cache state
    println!("\n=== KV Cache State ===");
    println!("Sequence length: {}", kv_cache.seq_len());
}

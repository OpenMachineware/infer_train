use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== CPU Operator Profiling ===\n");

    let mut ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    let mut kv_cache = KVCache::new(&ctx.config, 100, KVCacheFormat::F32);

    let prompt_tokens: Vec<u32> = vec![1];

    // Time embedding
    let start = std::time::Instant::now();
    let hidden = ctx.model.embedding_lookup_impl(1).expect("Embedding failed");
    println!("Embedding: {:?}", start.elapsed());

    // Time each layer
    for (i, layer) in ctx.model.layers.iter().enumerate() {
        let start = std::time::Instant::now();
        let _ = ctx.model.forward_layer(
            layer,
            i,
            &hidden,
            &mut kv_cache,
            ctx.config.n_heads,
            ctx.config.head_dim,
            ctx.config.n_kv_heads,
        );
        let elapsed = start.elapsed();
        if i < 3 || i == ctx.model.layers.len() - 1 {
            println!("Layer {}: {:?}", i, elapsed);
        }
    }
}

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};
use infer_train::model::GgufFile;
use std::time::Instant;

fn main() {
    let model_path = "./Qwen3-0.6B-UD-Q4_K_XL.gguf";
    println!("Loading model...");
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Cpu).unwrap();
    println!("Model loaded");

    // Test with different batch sizes
    for n_tokens in [2, 4, 8, 16, 32].iter() {
        let tokens: Vec<u32> = (0..*n_tokens as u32).collect();
        let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

        let start = Instant::now();
        let _ = ctx.forward_pooled(&tokens, &mut kv_cache);
        let elapsed = start.elapsed();

        let tps = *n_tokens as f64 / elapsed.as_secs_f64();
        println!("Prefill {} tokens: {:?} ({:.1} t/s)", n_tokens, elapsed, tps);
    }
}

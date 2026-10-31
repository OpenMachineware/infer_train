use infer_train::cpu::CpuWorkBuffers;
use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = &args[1];

    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu).unwrap();
    let hidden_size = ctx.config.hidden_size;
    let n_layers = ctx.model.layers.len();

    let mut buffers = CpuWorkBuffers::new(
        hidden_size,
        ctx.config.intermediate_size,
        ctx.config.n_heads,
        ctx.config.n_kv_heads,
        ctx.config.head_dim,
        ctx.config.key_length,
        ctx.config.value_length,
        ctx.config.vocab_size,
        0,
        2048,
        n_layers,
    );

    // Time embedding lookup
    let n = 1000;
    let mut total = std::time::Duration::ZERO;
    for i in 0..n {
        let token = (i % 100) as u32;
        let start = Instant::now();
        ctx.model.embedding_lookup_into(token, &mut buffers.hidden);
        total += start.elapsed();
    }

    println!("Embedding lookup ({} calls):", n);
    println!("Total: {:?}", total);
    println!("Per call: {:?}", total / n as u32);
}

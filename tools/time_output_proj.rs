use infer_train::cpu::CpuWorkBuffers;
use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = &args[1];

    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu).unwrap();
    let hidden_size = ctx.config.hidden_size;
    let vocab_size = ctx.config.vocab_size;
    let n_layers = ctx.model.layers.len();

    let mut buffers = CpuWorkBuffers::new(
        hidden_size,
        ctx.config.intermediate_size,
        ctx.config.n_heads,
        ctx.config.n_kv_heads,
        ctx.config.head_dim,
        ctx.config.key_length,
        ctx.config.value_length,
        vocab_size,
        0,
        2048,
        n_layers,
    );

    // Fill hidden_alt with test values
    for i in 0..hidden_size {
        buffers.hidden_alt[i] = i as f32 / hidden_size as f32;
    }

    // Time output projection
    let n = 100;
    let mut total = std::time::Duration::ZERO;
    for _ in 0..n {
        let start = Instant::now();
        ctx.model.gemv_into_with_buffers_single_thread(
            &buffers.hidden_alt,
            ctx.model.output.as_ref().unwrap(),
            &mut buffers.logits,
            &mut buffers.q8k_input,
            &mut buffers.x_padded,
        );
        total += start.elapsed();
    }

    println!("Output projection (1024 x {}):", vocab_size);
    println!("Total: {:?} for {} iterations", total, n);
    println!("Per call: {:?}", total / n);
}

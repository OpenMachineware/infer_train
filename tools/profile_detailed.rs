// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Detailed timing breakdown for GPU decode
// Measures: embedding, upload, layers, output projection

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = args[1].clone();

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::time::Instant;

        println!("=== Detailed GPU Decode Timing ===\n");

        let result = std::thread::spawn(move || {
            let gpu_ctx = MetalContext::new().unwrap();
            let device = gpu_ctx.device();
            let ctx = InferenceContext::from_gguf_with_gpu_ctx(
                &model_path,
                gpu_ctx.clone(),
            )
            .unwrap();
            let config = ctx.model.config.clone();
            let n_layers = config.n_layers;
            let hidden_size = config.hidden_size;

            println!(
                "Model: {} layers, hidden_size={}\n",
                n_layers, hidden_size
            );

            let mut kv = KVCacheGPU::new(device, &config, 2048);

            // Prefill
            println!("=== Prefill (10 tokens) ===");
            let prefill_tokens: Vec<u32> = (1..=10).collect();
            let t = Instant::now();
            let _ = ctx.model.forward_gpu_native_batched(
                &prefill_tokens,
                &mut kv,
                &gpu_ctx,
            );
            println!("  Total: {:?}\n", t.elapsed());

            // Decode multiple tokens
            println!("=== Decode (10 tokens) ===");
            for i in 0..10 {
                let token = (11 + i) as u32;
                let t = Instant::now();
                let _ = ctx.model.forward_gpu_native_batched(
                    &[token],
                    &mut kv,
                    &gpu_ctx,
                );
                println!("  Token {}: {:?}", i + 1, t.elapsed());
            }
        })
        .join()
        .unwrap();
    }
}

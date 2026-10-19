// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Simple timing breakdown of forward pass

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

        println!("=== Timing Breakdown ===");

        let result = std::thread::spawn(move || {
            let t0 = std::time::Instant::now();
            let gpu_ctx = MetalContext::new().unwrap();
            println!("MetalContext::new: {:?}", t0.elapsed());

            let t1 = std::time::Instant::now();
            let device = gpu_ctx.device();
            println!("device(): {:?}", t1.elapsed());

            let t2 = std::time::Instant::now();
            let ctx = InferenceContext::from_gguf_with_gpu_ctx(
                &model_path,
                gpu_ctx.clone(),
            )
            .unwrap();
            println!("Load model: {:?}", t2.elapsed());

            let config = ctx.model.config.clone();
            let n_layers = config.n_layers;
            println!(
                "n_layers: {}, hidden_size: {}",
                n_layers, config.hidden_size
            );

            let t3 = std::time::Instant::now();
            let mut kv = KVCacheGPU::new(device, &config, 2048);
            println!("KVCacheGPU::new: {:?}", t3.elapsed());

            // Prefill
            let t4 = std::time::Instant::now();
            let prefill_tokens: Vec<u32> = (1..=10).collect();
            let _ = ctx.model.forward_gpu_native_batched(
                &prefill_tokens,
                &mut kv,
                &gpu_ctx,
            );
            println!("Prefill (10 tokens): {:?}", t4.elapsed());

            // Decode
            println!("\nDecode (5 tokens):");
            for i in 0..5 {
                let token = (11 + i) as u32;
                let t = std::time::Instant::now();
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

        println!("\nDone");
    }
}

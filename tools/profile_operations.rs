// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile individual operations in decode

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

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(&model_path, gpu_ctx.clone()).unwrap();
        let config = ctx.model.config.clone();
        let mut kv = KVCacheGPU::new(device.clone(), &config, 2048);

        // Get one layer to profile
        if ctx.model.layers.is_empty() {
            eprintln!("No layers!");
            return;
        }
        let layer = &ctx.model.layers[0];

        // Create a test input buffer
        let hidden_size = config.hidden_size;
        let input_data: Vec<f32> = (0..hidden_size).map(|i| (i as f32 * 0.01).sin()).collect();
        let input_buffer = device.new_buffer_with_data(
            input_data.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Profile output layer projection (151936 x 1024)
        println!("=== Output Layer Projection Profile ===");
        if let Some(output) = ctx.model.output.as_ref() {
            let vocab_size = output.shape[0];
            println!("Output layer shape: {:?}", output.shape);
            println!("Vocab size: {}, Hidden size: {}", vocab_size, hidden_size);

            let output_weights = gpu_ctx.get_or_create_buffer(output);
            let output_buffer = device.new_buffer(
                (vocab_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            // Time just the output projection
            let queue = gpu_ctx.queue();
            let mut times = Vec::new();

            for _ in 0..20 {
                let start = Instant::now();
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                gpu_ctx.mv_encode(
                    encoder,
                    &output_weights,
                    &input_buffer,
                    &output_buffer,
                    vocab_size,
                    hidden_size,
                    output.dtype,
                );

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                times.push(start.elapsed());
            }

            let avg: std::time::Duration = times.iter().sum::<std::time::Duration>() / times.len() as u32;
            println!("Average output projection time: {:?}", avg);
            println!("Throughput: {:.1} vocab/sec", 1.0 / avg.as_secs_f64());
        }

        // Profile one attention layer
        println!("\n=== Single Layer Profile ===");
        println!("Layer 0 weights:");
        if let Some(wq) = layer.wq.as_ref() {
            println!("  wq shape: {:?}, dtype: {:?}", wq.shape, wq.dtype);
        }
        if let Some(wk) = layer.wk.as_ref() {
            println!("  wk shape: {:?}, dtype: {:?}", wk.shape, wk.dtype);
        }
        if let Some(wv) = layer.wv.as_ref() {
            println!("  wv shape: {:?}, dtype: {:?}", wv.shape, wv.dtype);
        }
        if let Some(wo) = layer.wo.as_ref() {
            println!("  wo shape: {:?}, dtype: {:?}", wo.shape, wo.dtype);
        }
        if let Some(gate) = layer.gate_proj.as_ref() {
            println!("  gate_proj shape: {:?}, dtype: {:?}", gate.shape, gate.dtype);
        }
        if let Some(up) = layer.up_proj.as_ref() {
            println!("  up_proj shape: {:?}, dtype: {:?}", up.shape, up.dtype);
        }
        if let Some(down) = layer.down_proj.as_ref() {
            println!("  down_proj shape: {:?}, dtype: {:?}", down.shape, down.dtype);
        }
    }
}

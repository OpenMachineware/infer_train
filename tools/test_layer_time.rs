// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test single layer forward time

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = args[1].clone();

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::buffer::ForwardBufferPool;
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::cell::RefCell;
        use std::rc::Rc;
        use std::time::Instant;

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            &model_path,
            gpu_ctx.clone(),
        )
        .unwrap();
        let config = ctx.model.config.clone();
        let mut kv = KVCacheGPU::new(&device, &config, 2048);

        // Prefill to position 10
        let prefill_tokens: Vec<u32> = (1..=10).collect();
        let _ = ctx.model.forward_gpu_native_batched(
            &prefill_tokens,
            &mut kv,
            &gpu_ctx,
        );

        let hidden_size = config.hidden_size;
        let reuse_pool: Rc<RefCell<Vec<Rc<metal::Buffer>>>> =
            Rc::new(RefCell::new(Vec::new()));

        println!("=== Single Layer Full Forward Time ===");

        // Create test input
        let input_data: Vec<f32> = vec![0.1; hidden_size];
        let input_buffer = Rc::new(device.new_buffer_with_data(
            input_data.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let queue = gpu_ctx.queue();
        let mut times = Vec::new();

        // Measure
        for _ in 0..20 {
            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());

            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let layer_output = ctx.model.forward_layer_gpu_encode(
                &ctx.model.layers[0],
                0,
                &input_buffer,
                &mut kv,
                &gpu_ctx,
                &mut forward_pool,
                encoder,
            );

            encoder.end_encoding();

            let start = Instant::now();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            times.push(start.elapsed());
        }

        let avg: std::time::Duration =
            times.iter().sum::<std::time::Duration>() / times.len() as u32;
        println!("  Avg time for 1 layer: {:?}", avg);
        println!("  Expected for 28 layers: {:?}", avg * 28);
    }
}

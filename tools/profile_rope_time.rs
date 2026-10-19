// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile RoPE execution time

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = args[1].clone();

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::InferenceContext;
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::time::Instant;

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            &model_path,
            gpu_ctx.clone(),
        )
        .unwrap();
        let config = ctx.model.config.clone();

        let hidden_size = config.hidden_size;
        let head_dim = config.head_dim;
        let n_heads = config.n_heads;

        println!("=== RoPE Performance ===");
        println!(
            "Hidden size: {}, Head dim: {}, N heads: {}",
            hidden_size, head_dim, n_heads
        );

        // Create test Q buffer
        let q_size = n_heads * head_dim;
        let q_data: Vec<f32> = vec![0.1; q_size];
        let q_buffer = std::rc::Rc::new(device.new_buffer_with_data(
            q_data.as_ptr() as *const std::ffi::c_void,
            (q_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let output_buffer = std::rc::Rc::new(device.new_buffer(
            (q_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let queue = gpu_ctx.queue();
        let mut times = Vec::new();

        // Warm up
        for _ in 0..5 {
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            gpu_ctx.rope_neox_encode(
                encoder,
                &q_buffer,
                &[10], // position
                1,     // n_tokens
                n_heads,
                head_dim,
                10000.0,
                1.0,
                &output_buffer,
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // Measure
        for _ in 0..20 {
            let start = Instant::now();
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            gpu_ctx.rope_neox_encode(
                encoder,
                &q_buffer,
                &[10],
                1,
                n_heads,
                head_dim,
                10000.0,
                1.0,
                &output_buffer,
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            times.push(start.elapsed());
        }

        let avg: std::time::Duration =
            times.iter().sum::<std::time::Duration>() / times.len() as u32;
        println!("  Avg time: {:?}", avg);
        println!("  For 28 layers: {:?}", avg * 28);
    }
}

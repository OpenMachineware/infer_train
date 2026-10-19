// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile output layer projection time

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
        let hidden_size = ctx.model.config.hidden_size;

        // Create test input buffer
        let input_data: Vec<f32> = vec![0.1; hidden_size];
        let input_buffer = std::rc::Rc::new(device.new_buffer_with_data(
            input_data.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        if let Some(output) = ctx.model.output.as_ref() {
            let vocab_size = output.shape[0];
            println!("=== Output Layer Profile ===");
            println!("Shape: {} x {} (311M ops)", vocab_size, hidden_size);
            println!("Dtype: {:?}", output.dtype);

            let output_weights = gpu_ctx.get_or_create_buffer(output);
            let output_buffer = std::rc::Rc::new(device.new_buffer(
                (vocab_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let queue = gpu_ctx.queue();
            let mut times = Vec::new();

            // Warm up
            for _ in 0..5 {
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
            }

            // Measure
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

            let avg: std::time::Duration =
                times.iter().sum::<std::time::Duration>() / times.len() as u32;
            let min = times.iter().min().unwrap();
            let max = times.iter().max().unwrap();

            println!("\nOutput projection time:");
            println!("  Avg: {:?}", avg);
            println!("  Min: {:?}", min);
            println!("  Max: {:?}", max);

            let ops = vocab_size as f64 * hidden_size as f64 * 2.0;
            let gops = ops / avg.as_secs_f64() / 1e9;
            println!("  Performance: {:.2} GOPS", gops);

            // Calculate what percentage of total decode time this is
            println!(
                "\n  If total decode is ~16ms, output projection takes {:.1}%",
                avg.as_secs_f64() / 0.016 * 100.0
            );
        }
    }
}

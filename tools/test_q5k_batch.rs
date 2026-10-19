// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test Q5_K GEMV batch performance

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

        if let Some(layer) = ctx.model.layers.first() {
            let m = 1024;
            let k = 2048;

            println!("=== Q5_K Batch GEMV Performance ===");
            println!("Testing 7 GEMVs (one layer) in single command buffer");

            // Create test input
            let input_data: Vec<f32> = vec![0.1; k];
            let input_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                input_data.as_ptr() as *const std::ffi::c_void,
                (k * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let queue = gpu_ctx.queue();
            let mut times = Vec::new();

            // Warm up
            for _ in 0..5 {
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                // Encode all 7 GEMVs
                for w in [
                    &layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.w1,
                    &layer.w3, &layer.w2,
                ] {
                    if let Some(weight) = w.as_ref() {
                        let weights_buffer =
                            gpu_ctx.get_or_create_buffer(weight);
                        let out_size = weight.shape[0];
                        let output_buffer = device.new_buffer(
                            (out_size * 4) as u64,
                            metal::MTLResourceOptions::StorageModeShared,
                        );
                        gpu_ctx.mv_encode(
                            encoder,
                            &weights_buffer,
                            &input_buffer,
                            &output_buffer,
                            out_size,
                            k,
                            weight.dtype,
                        );
                    }
                }

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }

            // Measure
            for _ in 0..20 {
                let start = Instant::now();
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                // Encode all 7 GEMVs
                for w in [
                    &layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.w1,
                    &layer.w3, &layer.w2,
                ] {
                    if let Some(weight) = w.as_ref() {
                        let weights_buffer =
                            gpu_ctx.get_or_create_buffer(weight);
                        let out_size = weight.shape[0];
                        let output_buffer = device.new_buffer(
                            (out_size * 4) as u64,
                            metal::MTLResourceOptions::StorageModeShared,
                        );
                        gpu_ctx.mv_encode(
                            encoder,
                            &weights_buffer,
                            &input_buffer,
                            &output_buffer,
                            out_size,
                            k,
                            weight.dtype,
                        );
                    }
                }

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                times.push(start.elapsed());
            }

            let avg: std::time::Duration =
                times.iter().sum::<std::time::Duration>() / times.len() as u32;

            println!("  Avg time for 7 GEMVs: {:?}", avg);
            println!("  Per GEMV: {:?}", avg / 7);
            println!("  For 28 layers (196 GEMVs): {:?}", avg * 28);
        }
    }
}

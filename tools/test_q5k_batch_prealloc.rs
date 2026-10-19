// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test Q5_K GEMV batch performance with pre-allocated buffers

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
            println!("=== Q5_K Batch GEMV Performance (Pre-allocated) ===");
            println!("Testing 7 GEMVs (one layer) in single command buffer");

            // Pre-allocate all buffers
            let k = 2048usize;
            let input_data: Vec<f32> = vec![0.1; k];
            let input_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                input_data.as_ptr() as *const std::ffi::c_void,
                (k * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            // Pre-allocate output buffers
            let output_buffers: Vec<std::rc::Rc<metal::Buffer>> = vec![
                std::rc::Rc::new(device.new_buffer(
                    (1024 * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                )), // wq
                std::rc::Rc::new(device.new_buffer(
                    (1024 * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                )), // wk
                std::rc::Rc::new(device.new_buffer(
                    (1024 * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                )), // wv
                std::rc::Rc::new(device.new_buffer(
                    (2048 * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                )), // wo
                std::rc::Rc::new(device.new_buffer(
                    (3072 * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                )), // w1
                std::rc::Rc::new(device.new_buffer(
                    (3072 * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                )), // w3
                std::rc::Rc::new(device.new_buffer(
                    (1024 * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                )), // w2
            ];

            let queue = gpu_ctx.queue();
            let mut times = Vec::new();

            // Warm up
            for _ in 0..5 {
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                for (i, w) in [
                    layer.wq.as_ref(),
                    layer.wk.as_ref(),
                    layer.wv.as_ref(),
                    layer.wo.as_ref(),
                    layer.w1.as_ref(),
                    layer.w3.as_ref(),
                    layer.w2.as_ref(),
                ]
                .iter()
                .enumerate()
                {
                    if let Some(weight) = w {
                        let weights_buffer =
                            gpu_ctx.get_or_create_buffer(weight);
                        let out_size = weight.shape[0];
                        gpu_ctx.mv_encode(
                            encoder,
                            &weights_buffer,
                            &input_buffer,
                            &output_buffers[i],
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

                for (i, w) in [
                    layer.wq.as_ref(),
                    layer.wk.as_ref(),
                    layer.wv.as_ref(),
                    layer.wo.as_ref(),
                    layer.w1.as_ref(),
                    layer.w3.as_ref(),
                    layer.w2.as_ref(),
                ]
                .iter()
                .enumerate()
                {
                    if let Some(weight) = w {
                        let weights_buffer =
                            gpu_ctx.get_or_create_buffer(weight);
                        let out_size = weight.shape[0];
                        gpu_ctx.mv_encode(
                            encoder,
                            &weights_buffer,
                            &input_buffer,
                            &output_buffers[i],
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

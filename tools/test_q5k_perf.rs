// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test Q5_K GEMV performance

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
            if let Some(wq) = layer.wq.as_ref() {
                let m = wq.shape[0];
                let k = wq.shape[1];

                println!("=== Q5_K GEMV Performance ===");
                println!("wq: {} x {}", m, k);

                // Create test input
                let input_data: Vec<f32> = vec![0.1; k];
                let input_buffer =
                    std::rc::Rc::new(device.new_buffer_with_data(
                        input_data.as_ptr() as *const std::ffi::c_void,
                        (k * 4) as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    ));

                let weights_buffer = gpu_ctx.get_or_create_buffer(wq);
                let output_buffer = std::rc::Rc::new(device.new_buffer(
                    (m * 4) as u64,
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
                        &weights_buffer,
                        &input_buffer,
                        &output_buffer,
                        m,
                        k,
                        wq.dtype,
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
                        &weights_buffer,
                        &input_buffer,
                        &output_buffer,
                        m,
                        k,
                        wq.dtype,
                    );
                    encoder.end_encoding();
                    command_buffer.commit();
                    command_buffer.wait_until_completed();
                    times.push(start.elapsed());
                }

                let avg: std::time::Duration =
                    times.iter().sum::<std::time::Duration>()
                        / times.len() as u32;
                let ops = m as f64 * k as f64 * 2.0;
                let gops = ops / avg.as_secs_f64() / 1e9;

                println!("  Avg time: {:?}", avg);
                println!("  Performance: {:.2} GOPS", gops);
            }
        }
    }
}

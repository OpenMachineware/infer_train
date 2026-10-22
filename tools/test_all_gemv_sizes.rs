// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test Q5_K GEMV for all sizes used in DeepSeek-R1-Distill-Qwen-1.5B

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::InferenceContext;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::model::Tensor;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let model_path = &args[1];
        println!("=== Q5_K GEMV Size Comparison ===\n");

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            gpu_ctx.clone(),
        )
        .unwrap();

        println!("Model config:");
        println!("  hidden_size: {}", ctx.config.hidden_size);
        println!("  intermediate_size: {}", ctx.config.intermediate_size);
        println!("  n_layers: {}", ctx.config.n_layers);
        println!();

        if let Some(layer) = ctx.model.layers.first() {
            println!("Testing layer 0 kernels:\n");

            // Test Q/K/V projections (1536x1536)
            if let Some(wq) = layer.wq.as_ref() {
                test_gemv_size(&gpu_ctx, "wq (Q projection)", wq);
            }
            if let Some(wk) = layer.wk.as_ref() {
                test_gemv_size(&gpu_ctx, "wk (K projection)", wk);
            }
            if let Some(wv) = layer.wv.as_ref() {
                test_gemv_size(&gpu_ctx, "wv (V projection)", wv);
            }

            // Test FFN projections (gate/up: 1536x8192, down: 8192x1536)
            if let Some(w1) = layer.w1.as_ref() {
                test_gemv_size(&gpu_ctx, "w1 (gate projection)", w1);
            }
            if let Some(w3) = layer.w3.as_ref() {
                test_gemv_size(&gpu_ctx, "w3 (up projection)", w3);
            }
            if let Some(w2) = layer.w2.as_ref() {
                test_gemv_size(&gpu_ctx, "w2 (down projection)", w2);
            }

            // Test output projection (1536x151936)
            if let Some(output) = ctx.model.output.as_ref() {
                test_gemv_size(&gpu_ctx, "output (lm_head)", output);
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn test_gemv_size(gpu_ctx: &MetalContext, name: &str, weight: &Tensor) {
    use std::rc::Rc;
    use std::time::Instant;

    let device = gpu_ctx.device();
    let m = weight.shape[0];
    let k = weight.shape[1];

    println!("{}: {}x{}", name, m, k);

    // Create test input
    let input_data: Vec<f32> = vec![0.1; k];
    let input_buffer = Rc::new(device.new_buffer_with_data(
        input_data.as_ptr() as *const std::ffi::c_void,
        (k * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    let weights_buffer = gpu_ctx.get_or_create_buffer(weight);
    let output_buffer = Rc::new(device.new_buffer(
        (m * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    let queue = gpu_ctx.queue();

    // Warmup
    for _ in 0..5 {
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        gpu_ctx.mv_encode(
            encoder,
            &weights_buffer,
            &input_buffer,
            &output_buffer,
            m,
            k,
            weight.dtype,
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Measure
    let iterations = 20;
    let mut times = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        gpu_ctx.mv_encode(
            encoder,
            &weights_buffer,
            &input_buffer,
            &output_buffer,
            m,
            k,
            weight.dtype,
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }

    let avg: std::time::Duration =
        times.iter().sum::<std::time::Duration>() / iterations as u32;
    let ops = m as f64 * k as f64 * 2.0;
    let gops = ops / avg.as_secs_f64() / 1e9;
    let bandwidth_gbs =
        (m * k * 2 + m * 4 + k * 4) as f64 / avg.as_secs_f64() / 1e9;

    println!("  Avg time: {:?}", avg);
    println!("  Performance: {:.2} GOPS", gops);
    println!("  Bandwidth: {:.2} GB/s", bandwidth_gbs);
    println!();
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This benchmark requires Metal support");
}

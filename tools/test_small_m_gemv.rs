// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test small M GEMV performance to identify bottleneck

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
        println!("=== Small M GEMV Performance Analysis ===\n");

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();
        let ctx =
            infer_train::inference::InferenceContext::from_gguf_with_gpu_ctx(
                model_path,
                gpu_ctx.clone(),
            )
            .unwrap();

        if let Some(layer) = ctx.model.layers.first() {
            let wq = layer.wq.as_ref().expect("No wq");
            let wk = layer.wk.as_ref().expect("No wk");

            // Shape for weights: [K, M] (transposed storage for efficient GEMV)
            // GEMV: output = weights @ input, where input is [K] and output is [M]
            // So weights should be [M, K] for normal storage, or [K, M] for transposed
            let k = wq.shape[0]; // K dimension (1536)
            let m_q = wq.shape[1]; // M for Q (1536)
            let m_k = wk.shape[1]; // M for K (256) - this is the small M case

            println!(
                "DEBUG: wq shape: {:?}, wk shape: {:?}",
                wq.shape, wk.shape
            );
            println!("DEBUG: K={}, M_Q={}, M_K={}", k, m_q, m_k);

            println!("Configuration:");
            println!("  K dimension: {}", k);
            println!("  M for Q projection: {}", m_q);
            println!("  M for K projection: {}", m_k);
            println!();

            // Create test input
            let input_data: Vec<f32> = vec![0.1; k];
            let input_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                input_data.as_ptr() as *const std::ffi::c_void,
                (k * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            // Test Q projection (M=1536)
            println!("=== Q Projection (M={}) ===", m_q);
            let wq_buffer = gpu_ctx.get_or_create_buffer(wq);
            let q_output = std::rc::Rc::new(device.new_buffer(
                (m_q * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            // Warmup
            for _ in 0..5 {
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                gpu_ctx.mv_encode(
                    encoder,
                    &wq_buffer,
                    &input_buffer,
                    &q_output,
                    m_q,
                    k,
                    wq.dtype,
                );
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }

            // Measure
            let iterations = 100;
            let mut times = Vec::new();
            for _ in 0..iterations {
                let start = std::time::Instant::now();
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                gpu_ctx.mv_encode(
                    encoder,
                    &wq_buffer,
                    &input_buffer,
                    &q_output,
                    m_q,
                    k,
                    wq.dtype,
                );
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                times.push(start.elapsed());
            }

            let avg_q: std::time::Duration =
                times.iter().sum::<std::time::Duration>() / iterations as u32;
            let gops_q = (m_q * k) as f64 / 1e9 / avg_q.as_secs_f64();
            println!("  Avg time: {:?}", avg_q);
            println!("  GOPS: {:.2}", gops_q);
            println!();

            // Test K projection (M=256)
            println!("=== K Projection (M={}) ===", m_k);
            let wk_buffer = gpu_ctx.get_or_create_buffer(wk);
            let k_output = std::rc::Rc::new(device.new_buffer(
                (m_k * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            // Warmup
            for _ in 0..5 {
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                gpu_ctx.mv_encode(
                    encoder,
                    &wk_buffer,
                    &input_buffer,
                    &k_output,
                    m_k,
                    k,
                    wk.dtype,
                );
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }

            // Measure
            let mut times = Vec::new();
            for _ in 0..iterations {
                let start = std::time::Instant::now();
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                gpu_ctx.mv_encode(
                    encoder,
                    &wk_buffer,
                    &input_buffer,
                    &k_output,
                    m_k,
                    k,
                    wk.dtype,
                );
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                times.push(start.elapsed());
            }

            let avg_k: std::time::Duration =
                times.iter().sum::<std::time::Duration>() / iterations as u32;
            let gops_k = (m_k * k) as f64 / 1e9 / avg_k.as_secs_f64();
            println!("  Avg time: {:?}", avg_k);
            println!("  GOPS: {:.2}", gops_k);
            println!();

            // Analysis
            println!("=== Analysis ===");
            println!(
                "  Performance ratio: {:.2}x (Q projection faster)",
                gops_q / gops_k
            );
            println!("  Threadgroups: Q={} vs K={}", m_q, m_k);
            println!(
                "  Threads per operation: Q={} vs K={}",
                m_q * 32,
                m_k * 32
            );
            println!();

            // Calculate theoretical dispatch overhead
            println!("=== Dispatch Overhead Estimation ===");
            println!("  Assuming 10µs dispatch overhead per threadgroup:");
            println!(
                "    Q projection: {} threadgroups × 10µs = {}µs overhead",
                m_q,
                m_q * 10
            );
            println!(
                "    K projection: {} threadgroups × 10µs = {}µs overhead",
                m_k,
                m_k * 10
            );
            println!(
                "  K projection has {:.1}x higher overhead ratio",
                (m_q as f64 / m_k as f64)
                    * (avg_k.as_micros() as f64 / avg_q.as_micros() as f64)
            );
            println!();

            // Impact on decode
            println!("=== Impact on Decode ===");
            let kv_time = (avg_k.as_nanos() as f64 / 1000.0) * 2.0; // K + V
            let layer_gemv_time = avg_q.as_nanos() as f64 / 1000.0 + kv_time;
            println!("  Per layer GEMV time: {:.2}µs", layer_gemv_time);
            println!(
                "  28 layers total: {:.2}ms",
                layer_gemv_time * 28.0 / 1000.0
            );
            println!("  Current decode: 13.35ms");
            println!(
                "  GEMV contribution: {:.1}%",
                (layer_gemv_time * 28.0 / 1000.0 / 13.35) * 100.0
            );
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}

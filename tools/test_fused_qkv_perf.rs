// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test FusedQKV performance (three GEMV calls)
// The "fusion" is at DAG level to reduce node count, not kernel level

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::InferenceContext;
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
        println!("=== FusedQKV Performance (3x GEMV) ===\n");

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            gpu_ctx.clone(),
        )
        .unwrap();

        if let Some(layer) = ctx.model.layers.first() {
            // Get Q, K, V weights from first layer
            let wq = layer.wq.as_ref().expect("No wq");
            let wk = layer.wk.as_ref().expect("No wk");
            let wv = layer.wv.as_ref().expect("No wv");

            let hidden_size = wq.shape[1]; // 1536
            let q_out_size = wq.shape[0]; // 1536
            let k_out_size = wk.shape[0]; // 256
            let v_out_size = wv.shape[0]; // 256

            println!("Config:");
            println!("  hidden_size (K): {}", hidden_size);
            println!("  q_out_size (M): {}", q_out_size);
            println!("  k_out_size (M): {}", k_out_size);
            println!("  v_out_size (M): {}", v_out_size);
            println!(
                "  dtypes: Q={:?}, K={:?}, V={:?}",
                wq.dtype, wk.dtype, wv.dtype
            );
            println!();

            // Create test input
            let input_data: Vec<f32> = vec![0.1; hidden_size];
            let input_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                input_data.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            // Get weight buffers
            let wq_buffer = gpu_ctx.get_or_create_buffer(wq);
            let wk_buffer = gpu_ctx.get_or_create_buffer(wk);
            let wv_buffer = gpu_ctx.get_or_create_buffer(wv);

            // Create output buffers
            let q_output = std::rc::Rc::new(device.new_buffer(
                (q_out_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            let k_output = std::rc::Rc::new(device.new_buffer(
                (k_out_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            let v_output = std::rc::Rc::new(device.new_buffer(
                (v_out_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            // Warmup - measure three separate GEMV calls (what FusedQKV actually does)
            for _ in 0..5 {
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();

                // Three GEMV calls in sequence
                gpu_ctx.mv_encode(
                    encoder,
                    &wq_buffer,
                    &input_buffer,
                    &q_output,
                    q_out_size,
                    hidden_size,
                    wq.dtype,
                );
                gpu_ctx.mv_encode(
                    encoder,
                    &wk_buffer,
                    &input_buffer,
                    &k_output,
                    k_out_size,
                    hidden_size,
                    wk.dtype,
                );
                gpu_ctx.mv_encode(
                    encoder,
                    &wv_buffer,
                    &input_buffer,
                    &v_output,
                    v_out_size,
                    hidden_size,
                    wv.dtype,
                );

                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }

            // Measure
            let iterations = 50;
            let mut times = Vec::new();
            for _ in 0..iterations {
                let start = std::time::Instant::now();
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();

                // Three GEMV calls in sequence
                gpu_ctx.mv_encode(
                    encoder,
                    &wq_buffer,
                    &input_buffer,
                    &q_output,
                    q_out_size,
                    hidden_size,
                    wq.dtype,
                );
                gpu_ctx.mv_encode(
                    encoder,
                    &wk_buffer,
                    &input_buffer,
                    &k_output,
                    k_out_size,
                    hidden_size,
                    wk.dtype,
                );
                gpu_ctx.mv_encode(
                    encoder,
                    &wv_buffer,
                    &input_buffer,
                    &v_output,
                    v_out_size,
                    hidden_size,
                    wv.dtype,
                );

                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                times.push(start.elapsed());
            }

            let avg: std::time::Duration =
                times.iter().sum::<std::time::Duration>() / iterations as u32;
            let min_time = *times.iter().min().unwrap();

            println!("FusedQKV (3x GEMV in one command buffer):");
            println!("  Avg time: {:?}", avg);
            println!("  Min time: {:?}", min_time);
            println!();

            // Calculate GOPS
            // Q: M=1536, K=1536
            // K: M=256, K=1536
            // V: M=256, K=1536
            let q_gops = (q_out_size * hidden_size) as f64 / 1e9;
            let k_gops = (k_out_size * hidden_size) as f64 / 1e9;
            let v_gops = (v_out_size * hidden_size) as f64 / 1e9;
            let total_gops = q_gops + k_gops + v_gops;
            let gops = total_gops / avg.as_secs_f64();

            println!("GOPS breakdown:");
            println!(
                "  Q GEMV: {:.2} GOPS ({:.3} GOPS)",
                q_gops,
                q_gops / avg.as_secs_f64()
            );
            println!(
                "  K GEMV: {:.2} GOPS ({:.3} GOPS)",
                k_gops,
                k_gops / avg.as_secs_f64()
            );
            println!(
                "  V GEMV: {:.2} GOPS ({:.3} GOPS)",
                v_gops,
                v_gops / avg.as_secs_f64()
            );
            println!("  Total: {:.2} GOPS", total_gops);
            println!("  Throughput: {:.2} GOPS", gops);
            println!();

            // Impact on decode
            println!("Impact on decode:");
            println!("  28 layers × {:?} = {:?}", avg, avg * 28);
            println!("  Current decode: 13.5ms");
            println!(
                "  QKV contribution: {:.1}%",
                (avg.as_secs_f64() * 1000.0 * 28.0 / 13.5) * 100.0
            );
            println!();

            // Compare with separate dispatches
            println!("Note: FusedQKV runs 3 GEMV in ONE command buffer.");
            println!(
                "      Separate dispatches would add ~0.1ms overhead per layer."
            );
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}

// Measure breakdown of dispatch overhead components
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::quant::types::BlockQ4_0;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        let m = 1024;
        let k = 1024;

        // Create buffers
        let input = vec![0.1f32; k];
        let input_buffer = std::rc::Rc::new(ctx.device().new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (k * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Q4_0 weight buffer
        let nb = k / 32;
        let weight_buffer = std::rc::Rc::new(ctx.device().new_buffer(
            (nb * std::mem::size_of::<BlockQ4_0>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Warm up
        let _ =
            ctx.mv_q4_0_f32_gpu_weights(m, k, &weight_buffer, &input_buffer);

        println!("Measuring dispatch components for GEMV (100 iterations)");
        println!("M={}, K={}", m, k);

        let n_iter = 100;
        let mut create_output_times = Vec::with_capacity(n_iter);
        let mut create_cb_times = Vec::with_capacity(n_iter);
        let mut create_enc_times = Vec::with_capacity(n_iter);
        let mut encode_times = Vec::with_capacity(n_iter);
        let mut commit_times = Vec::with_capacity(n_iter);
        let mut wait_times = Vec::with_capacity(n_iter);

        let queue = ctx.queue();

        for _ in 0..n_iter {
            // 1. Create output buffer
            let t0 = std::time::Instant::now();
            let output_buffer = ctx.device().new_buffer(
                (m * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            create_output_times.push(t0.elapsed());

            // 2. Create command buffer
            let t1 = std::time::Instant::now();
            let cmd_buf = queue.new_command_buffer();
            create_cb_times.push(t1.elapsed());

            // 3. Create encoder
            let t2 = std::time::Instant::now();
            let encoder = cmd_buf.new_compute_command_encoder();
            create_enc_times.push(t2.elapsed());

            // 4. Encode GEMV kernel
            let nb = k / 32;
            let nb01 = (nb * std::mem::size_of::<BlockQ4_0>()) as u64;

            #[repr(C)]
            struct MvArgs {
                ne00: i32,
                ne01: i32,
                ne02: i32,
                nb00: u64,
                nb01: u64,
                nb02: u64,
                nb03: u64,
                ne10: i32,
                ne11: i32,
                ne12: i32,
                nb10: u64,
                nb11: u64,
                nb12: u64,
                nb13: u64,
                ne0: i32,
                ne1: i32,
                nr0: i32,
                r2: i16,
                r3: i16,
            }

            let args = MvArgs {
                ne00: k as i32,
                ne01: m as i32,
                ne02: 1,
                nb00: 1,
                nb01,
                nb02: 0,
                nb03: 0,
                ne10: k as i32,
                ne11: 1,
                ne12: 1,
                nb10: 4,
                nb11: k as u64 * 4,
                nb12: 0,
                nb13: 0,
                ne0: m as i32,
                ne1: 1,
                nr0: 4,
                r2: 1,
                r3: 1,
            };

            let t3 = std::time::Instant::now();
            // Use a simple add pipeline for encoding test
            encoder.set_compute_pipeline_state(&ctx.add_pipeline());
            encoder.set_buffer(0, Some(&weight_buffer), 0);
            encoder.set_buffer(1, Some(&input_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const std::ffi::c_void,
            );
            encoder.dispatch_thread_groups(
                metal::MTLSize {
                    width: (m as u64 + 7) / 8,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize { width: 32, height: 2, depth: 1 },
            );
            encode_times.push(t3.elapsed());

            // 5. End encoding + commit
            let t4 = std::time::Instant::now();
            encoder.end_encoding();
            cmd_buf.commit();
            commit_times.push(t4.elapsed());

            // 6. Wait
            let t5 = std::time::Instant::now();
            cmd_buf.wait_until_completed();
            wait_times.push(t5.elapsed());
        }

        let avg = |v: &[std::time::Duration]| {
            v.iter().map(|t| t.as_nanos() as f64).sum::<f64>()
                / v.len() as f64
                / 1000.0
        };

        println!(
            "\n  Create output buffer:   {:.2} us",
            avg(&create_output_times)
        );
        println!("  Create command buffer:  {:.2} us", avg(&create_cb_times));
        println!("  Create encoder:         {:.2} us", avg(&create_enc_times));
        println!("  Encode kernel:          {:.2} us", avg(&encode_times));
        println!("  Commit:                 {:.2} us", avg(&commit_times));
        println!("  Wait:                   {:.2} us", avg(&wait_times));

        let total: f64 = avg(&create_output_times)
            + avg(&create_cb_times)
            + avg(&create_enc_times)
            + avg(&encode_times)
            + avg(&commit_times)
            + avg(&wait_times);
        println!("\n  Total:                  {:.2} us", total);

        // Now measure the full mv_q4_0_f32_gpu_weights
        println!("\n\nMeasuring full mv_q4_0_f32_gpu_weights (100 iterations)");
        let mut full_times = Vec::with_capacity(100);
        for _ in 0..100 {
            let t = std::time::Instant::now();
            let _ = ctx.mv_q4_0_f32_gpu_weights(
                m,
                k,
                &weight_buffer,
                &input_buffer,
            );
            full_times.push(t.elapsed());
        }
        println!("  Average:                {:.2} us", avg(&full_times));
    }
}

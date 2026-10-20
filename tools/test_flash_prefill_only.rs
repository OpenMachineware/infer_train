// Test Flash Prefill kernel performance only
// SPDX-License-Identifier: Apache-2.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use metal::*;
use std::rc::Rc;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Load model context to get the kernel
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/Users/jia/Desktop/infer_train/DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf".to_string());

    let ctx = infer_train::inference::InferenceContext::from_gguf(
        &model_path,
        infer_train::inference::Device::Gpu,
    )
    .expect("Failed to load model");

    let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");

    let n_heads = ctx.model.config.n_heads;
    let head_dim = ctx.model.config.head_dim;

    println!("Testing Flash Prefill kernel alone");
    println!("n_heads={}, head_dim={}\n", n_heads, head_dim);

    // Test different prompt lengths
    let prompt_lengths = [16, 32, 64, 128, 256];

    println!("| Prompt | Time (ms) | Notes |");
    println!("|--------|-----------|-------|");

    for seq_q in prompt_lengths.iter() {
        // Create dummy Q, K, V buffers
        let q_size = *seq_q * n_heads * head_dim * 4; // F32
        let kv_size = n_heads * 512 * head_dim * 2; // F16, max_seq=512

        let q_buffer =
            Rc::new(device.new_buffer(
                q_size as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        let k_buffer =
            Rc::new(device.new_buffer(
                kv_size as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        let v_buffer =
            Rc::new(device.new_buffer(
                kv_size as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        let out_buffer =
            Rc::new(device.new_buffer(
                q_size as u64,
                MTLResourceOptions::StorageModeShared,
            ));

        // Fill with random data
        unsafe {
            std::ptr::write_bytes(q_buffer.contents() as *mut u8, 1, q_size);
            std::ptr::write_bytes(k_buffer.contents() as *mut u8, 2, kv_size);
            std::ptr::write_bytes(v_buffer.contents() as *mut u8, 3, kv_size);
        }

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Test the kernel
        let start = std::time::Instant::now();

        gpu_ctx.flash_prefill_batch_encode(
            &encoder,
            1, // batch
            n_heads,
            *seq_q, // seq_q
            *seq_q, // seq_k
            head_dim,
            &q_buffer,
            &k_buffer,
            &v_buffer,
            &out_buffer,
            n_heads * head_dim * 4, // q_stride_b
            head_dim * 4,           // q_stride_h
            4,                      // q_stride_s
            0,                      // k_stride_b (shared)
            512 * head_dim * 2,     // k_stride_h
            head_dim * 2,           // k_stride_s
            0,                      // v_stride_b
            512 * head_dim * 2,     // v_stride_h
            head_dim * 2,           // v_stride_s
            n_heads * head_dim * 4, // o_stride_b
            head_dim * 4,           // o_stride_h
            4,                      // o_stride_s
            0,                      // start_pos
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let elapsed = start.elapsed();

        println!("| {:6} | {:9.1} | |", seq_q, elapsed.as_secs_f64() * 1000.0);
    }

    println!("\n=== Test Complete ===");
}

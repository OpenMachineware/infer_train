// Test Q4_K MV kernel correctness
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use infer_train::model::weights::WeightStore;
    use infer_train::quant::vec_dot::gpu::metal::MetalContext;
    use infer_train::model::tensor::DataType;

    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";

    // Load weights
    let store = WeightStore::from_gguf(model_path).expect("Failed to load weights");

    // Get first layer's wq weight
    let wq = store.get_tensor("blk.0.attn_q.weight").expect("Failed to get wq");
    println!("WQ: dtype={:?}, shape={:?}", wq.dtype, wq.shape);

    if wq.dtype != DataType::Q4_K {
        println!("Not Q4_K format, skipping");
        return;
    }

    // Create Metal context
    let gpu_ctx = MetalContext::new().expect("Failed to create MetalContext");

    // Create input vector (1024 elements)
    let input: Vec<f32> = (0..1024).map(|i| (i as f32) / 1024.0).collect();

    // GPU MV
    let input_buffer = gpu_ctx.device().new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        1024 * 4,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let weights_buffer = gpu_ctx.get_or_create_buffer(wq);
    let output_size = wq.shape[1]; // 2048 for Qwen 0.6B
    let output_buffer = gpu_ctx.device().new_buffer(
        output_size as u64 * 4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Zero output buffer
    unsafe {
        std::ptr::write_bytes(output_buffer.contents() as *mut u8, 0, output_buffer.length() as usize);
    }

    // Call GPU MV
    let command_buffer = gpu_ctx.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    let success = gpu_ctx.mv_encode(
        encoder,
        &weights_buffer,
        &input_buffer,
        &output_buffer,
        output_size,
        1024,
        DataType::Q4_K,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    println!("GPU MV success: {}", success);

    // Read output
    let output: Vec<f32> = unsafe {
        std::slice::from_raw_parts(output_buffer.contents() as *const f32, output_size).to_vec()
    };

    let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = output.iter().sum::<f32>() / output_size as f32;
    let zero_count = output.iter().filter(|&&x| x == 0.0).count();

    println!("GPU output: len={}, min={:.4}, max={:.4}, mean={:.4}, zeros={}",
        output.len(), min, max, mean, zero_count);
    println!("First 10: {:?}", &output[..10]);

    // CPU reference
    let cpu_output = infer_train::inference::dequant_to_f32(
        wq.bytes(),
        wq.dtype,
        output_size * 1024
    );
    println!("\nCPU dequant: len={}", cpu_output.len());
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This tool requires Metal support");
}

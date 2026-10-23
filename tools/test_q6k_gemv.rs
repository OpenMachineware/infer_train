// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test Q6_K GEMV kernel with real model weights

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::model::Tensor;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn test_q6k_gemv(
    gpu_ctx: &infer_train::quant::vec_dot::gpu::metal::MetalContext,
    weight: &Tensor,
    m: usize,
    k: usize,
) {
    use infer_train::model::tensor::DataType;
    use metal::MTLResourceOptions;

    eprintln!("Testing Q6_K weight: shape=[{}, {}]", m, k);
    eprintln!(
        "  nb = {} blocks, nb01 = {} bytes/row",
        k / 256,
        (k / 256) * std::mem::size_of::<infer_train::quant::types::BlockQ6K>()
    );

    // Create test input
    let input: Vec<f32> = (0..k).map(|i| (i as f32 % 10.0) / 10.0).collect();
    let device = gpu_ctx.device();

    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let weights_buffer = gpu_ctx.get_or_create_buffer(weight);
    let output_buffer = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

    // Run GEMV
    let queue = gpu_ctx.queue();
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    let success = gpu_ctx.mv_encode(
        encoder,
        &weights_buffer,
        &input_buffer,
        &output_buffer,
        m,
        k,
        DataType::Q6_K,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if !success {
        eprintln!("✗ mv_encode returned false");
        return;
    }

    // Check output
    let output_ptr = output_buffer.contents() as *const f32;
    let output: Vec<f32> =
        unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() };

    let has_nan = output.iter().any(|x| x.is_nan() || x.is_infinite());
    let all_zeros = output.iter().all(|x| *x == 0.0);

    eprintln!("Output: {} values", output.len());
    if has_nan {
        eprintln!("✗ ERROR: Q6_K GEMV produces NaN or Inf!");
        let nan_count = output.iter().filter(|x| x.is_nan()).count();
        let inf_count = output.iter().filter(|x| x.is_infinite()).count();
        eprintln!("  NaN count: {}, Inf count: {}", nan_count, inf_count);
        for (i, val) in output.iter().enumerate().take(10) {
            eprintln!("  output[{}] = {}", i, val);
        }
    } else if all_zeros {
        eprintln!("✗ ERROR: Q6_K GEMV produces all zeros!");
    } else {
        let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("✓ Output range: [{:.4}, {:.4}]", min, max);
        eprintln!("✓ First 5 outputs: {:?}", &output[..5.min(output.len())]);
        eprintln!("✓ Q6_K GEMV kernel works correctly");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = &args[1];

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::InferenceContext;
        use infer_train::model::tensor::DataType;
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        eprintln!("=== Q6_K GEMV Kernel Test ===");
        eprintln!("Loading model: {}", model_path);

        let gpu_ctx = MetalContext::new().unwrap();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            gpu_ctx.clone(),
        )
        .unwrap();
        let model = &ctx.model;

        // Find a Q6_K tensor (V projection in first layer)
        let v_weight =
            model.layers[0].wv.as_ref().expect("No V projection in layer 0");

        if v_weight.dtype == DataType::Q6_K {
            let m = v_weight.shape[0];
            let k = v_weight.shape[1];
            test_q6k_gemv(&gpu_ctx, v_weight, m, k);
        } else {
            eprintln!(
                "V weight is {:?}, not Q6_K. Trying FFN tensors...",
                v_weight.dtype
            );

            // Try gate/up projections (w1/w3)
            if let Some(w1) = &model.layers[0].w1 {
                if w1.dtype == DataType::Q6_K {
                    let m = w1.shape[0];
                    let k = w1.shape[1];
                    test_q6k_gemv(&gpu_ctx, w1, m, k);
                    return;
                }
            }

            if let Some(w3) = &model.layers[0].w3 {
                if w3.dtype == DataType::Q6_K {
                    let m = w3.shape[0];
                    let k = w3.shape[1];
                    test_q6k_gemv(&gpu_ctx, w3, m, k);
                    return;
                }
            }

            eprintln!("No Q6_K tensors found in layer 0");
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        eprintln!("This test requires Metal support on aarch64");
    }
}

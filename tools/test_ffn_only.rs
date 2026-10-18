// Test if FFN produces NaN

use infer_train::inference::{Device, InferenceContext};
use std::rc::Rc;

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let model = &ctx.model;
        let gpu_ctx = ctx.gpu_ctx.as_ref().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();
        let hidden_size = model.config.hidden_size;

        // Create a test input
        let input = vec![0.5f32; hidden_size];
        let input_buf = Rc::new(device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let layer = &model.layers[0];
        let w1 = layer.w1.as_ref().unwrap();
        let w2 = layer.w2.as_ref().unwrap();
        let w3 = layer.w3.as_ref().unwrap();

        println!("FFN weights:");
        println!("  w1: shape={:?}, dtype={:?}", w1.shape, w1.dtype);
        println!("  w2: shape={:?}, dtype={:?}", w2.shape, w2.dtype);
        println!("  w3: shape={:?}, dtype={:?}", w3.shape, w3.dtype);

        let intermediate_size = w1.shape[1];

        // Run FFN twice
        for run in 1..=2 {
            println!("\n=== Run {} ===", run);

            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();

            // Create output buffers
            let gate_buf = Rc::new(device.new_buffer(
                (intermediate_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            let up_buf = Rc::new(device.new_buffer(
                (intermediate_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            let silu_buf = Rc::new(device.new_buffer(
                (intermediate_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            let hidden_ffn_buf = Rc::new(device.new_buffer(
                (intermediate_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            let out_buf = Rc::new(device.new_buffer(
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            // Gate projection
            model.gemv_encode(
                enc,
                &input_buf,
                w1,
                intermediate_size,
                gpu_ctx,
                &gate_buf,
            );
            // Up projection
            model.gemv_encode(
                enc,
                &input_buf,
                w3,
                intermediate_size,
                gpu_ctx,
                &up_buf,
            );
            // SiLU
            gpu_ctx.silu_encode(enc, &gate_buf, &silu_buf, intermediate_size);
            // Elementwise multiply
            gpu_ctx.mul_encode(
                enc,
                &silu_buf,
                &up_buf,
                &hidden_ffn_buf,
                intermediate_size,
            );
            // Down projection
            model.gemv_encode(
                enc,
                &hidden_ffn_buf,
                w2,
                hidden_size,
                gpu_ctx,
                &out_buf,
            );

            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            // Check for NaN
            let mut output = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_buf.contents() as *const f32,
                    output.as_mut_ptr(),
                    hidden_size,
                );
            }

            let nan_count = output.iter().filter(|&&v| v.is_nan()).count();
            println!("FFN output: {} NaN", nan_count);
            if nan_count > 0 {
                println!("NaN positions:");
                for (i, &v) in output.iter().enumerate() {
                    if v.is_nan() {
                        println!("  Position {}: NaN", i);
                        if i > 10 {
                            println!("  ... and {} more", nan_count - i);
                            break;
                        }
                    }
                }
            }
            println!("First 5 output: {:?}", &output[..5]);
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}

// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Check if weight buffers are corrupted across passes

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::KVCacheGPU;
use std::cell::RefCell;
use std::rc::Rc;

fn check_weight_buffer(name: &str, buf: &metal::BufferRef, size: usize) {
    let data = unsafe {
        std::slice::from_raw_parts(buf.contents() as *const f32, size / 4)
    };
    let nan_count = data.iter().filter(|&&v| v.is_nan()).count();
    let inf_count = data.iter().filter(|&&v| v.is_infinite()).count();
    println!(
        "{}: {} NaN, {} inf, first 3: {:?}",
        name,
        nan_count,
        inf_count,
        &data[..3.min(data.len())]
    );
}

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
        let config = &model.config;
        let hidden_size = config.hidden_size;
        let head_dim = config.head_dim;

        // Get layer 0 weights
        let layer = &model.layers[0];
        let wq = layer.wq.as_ref().unwrap();
        let wk = layer.wk.as_ref().unwrap();
        let wv = layer.wv.as_ref().unwrap();

        println!("\n=== Before any forward pass ===");
        let wq_buf = gpu_ctx.get_or_create_buffer(wq);
        check_weight_buffer("WQ", &wq_buf, wq.bytes().len());
        let wk_buf = gpu_ctx.get_or_create_buffer(wk);
        check_weight_buffer("WK", &wk_buf, wk.bytes().len());
        let wv_buf = gpu_ctx.get_or_create_buffer(wv);
        check_weight_buffer("WV", &wv_buf, wv.bytes().len());

        // Prefill
        println!("\n=== After prefill (token 1) ===");
        let mut kv = KVCacheGPU::new(device, config, 2048);
        let result1 = model.forward_gpu_native_batched(&[1], &mut kv, gpu_ctx);

        // Check weights again
        check_weight_buffer("WQ", &wq_buf, wq.bytes().len());
        check_weight_buffer("WK", &wk_buf, wk.bytes().len());
        check_weight_buffer("WV", &wv_buf, wv.bytes().len());

        if let Some(ref r) = result1 {
            let nan_count = r.iter().filter(|&&v| v.is_nan()).count();
            println!("Result: {} NaN", nan_count);
        }

        // Decode
        println!("\n=== After decode (token 2) ===");
        let result2 = model.forward_gpu_native_batched(&[2], &mut kv, gpu_ctx);

        // Check weights again
        check_weight_buffer("WQ", &wq_buf, wq.bytes().len());
        check_weight_buffer("WK", &wk_buf, wk.bytes().len());
        check_weight_buffer("WV", &wv_buf, wv.bytes().len());

        if let Some(ref r) = result2 {
            let nan_count = r.iter().filter(|&&v| v.is_nan()).count();
            println!("Result: {} NaN", nan_count);
        }

        // Check if buffers are the same objects (by checking pointer)
        let wq_buf2 = gpu_ctx.get_or_create_buffer(wq);
        let ptr1 = wq_buf.contents() as *const u8;
        let ptr2 = wq_buf2.contents() as *const u8;
        println!(
            "\nWQ buffer same object: {} (ptr1={:?}, ptr2={:?})",
            ptr1 == ptr2,
            ptr1,
            ptr2
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}

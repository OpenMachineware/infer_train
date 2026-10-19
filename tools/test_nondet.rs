// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test where non-determinism starts

use infer_train::inference::{Device, InferenceContext};

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use std::rc::Rc;

        let ctx1 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let ctx2 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();

        // Test 1: Embedding lookup (CPU-only)
        println!("=== Test 1: Embedding Lookup ===");
        let emb1 = ctx1.model.embedding_lookup_impl(1).unwrap();
        let emb2 = ctx2.model.embedding_lookup_impl(1).unwrap();

        let max_diff: f32 = emb1
            .iter()
            .zip(emb2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        println!("Embedding max diff: {:.6}", max_diff);
        println!("Embedding first 5 (ctx1): {:?}", &emb1[..5]);
        println!("Embedding first 5 (ctx2): {:?}", &emb2[..5]);

        // Test 2: Create buffer and check
        println!("\n=== Test 2: GPU Buffer ===");
        let gpu_ctx = ctx1.gpu_ctx.as_ref().unwrap();
        let device = gpu_ctx.device();

        let buf1 = Rc::new(
            device
                .new_buffer(4096, metal::MTLResourceOptions::StorageModeShared),
        );
        let buf2 = Rc::new(
            device
                .new_buffer(4096, metal::MTLResourceOptions::StorageModeShared),
        );

        // Upload same embedding to both
        unsafe {
            std::ptr::copy_nonoverlapping(
                emb1.as_ptr() as *const std::ffi::c_void,
                buf1.contents(),
                4096,
            );
            std::ptr::copy_nonoverlapping(
                emb2.as_ptr() as *const std::ffi::c_void,
                buf2.contents(),
                4096,
            );
        }

        // Read back and compare
        let mut v1 = vec![0.0f32; 1024];
        let mut v2 = vec![0.0f32; 1024];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf1.contents() as *const f32,
                v1.as_mut_ptr(),
                1024,
            );
            std::ptr::copy_nonoverlapping(
                buf2.contents() as *const f32,
                v2.as_mut_ptr(),
                1024,
            );
        }

        let max_diff: f32 = v1
            .iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        println!("Buffer max diff: {:.6}", max_diff);

        // Test 3: Run first layer twice on SAME context
        println!("\n=== Test 3: Same context, two runs ===");
        let mut ctx =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();

        // Need to create fresh KV cache for each run
        use infer_train::inference::KVCacheGPU;
        let config = &ctx.model.config;
        let gpu_device = ctx.gpu_ctx.as_ref().unwrap().device();

        let mut kv1 = KVCacheGPU::new(gpu_device, config, 2048);
        let mut kv2 = KVCacheGPU::new(gpu_device, config, 2048);

        // Run layer 0 twice
        let r1 = ctx.model.forward_gpu_native_batched(
            &[1],
            &mut kv1,
            ctx.gpu_ctx.as_ref().unwrap(),
        );
        let r2 = ctx.model.forward_gpu_native_batched(
            &[1],
            &mut kv2,
            ctx.gpu_ctx.as_ref().unwrap(),
        );

        if let (Some(r1), Some(r2)) = (r1, r2) {
            let max_diff: f32 = r1
                .iter()
                .zip(r2.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            println!("Layer 0 max diff: {:.6}", max_diff);
            println!("r1 first 5: {:?}", &r1[..5]);
            println!("r2 first 5: {:?}", &r2[..5]);
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}

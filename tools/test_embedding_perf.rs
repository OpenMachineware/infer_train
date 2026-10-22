// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test embedding lookup performance

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::Device;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::model::llama::LlamaModel;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let args: Vec<String> = std::env::args().collect();
        if args.len() < 2 {
            eprintln!("Usage: {} <model.gguf>", args[0]);
            std::process::exit(1);
        }

        let model_path = &args[1];
        println!("=== Embedding Lookup Performance Test ===");
        println!("Model: {}\n", model_path);

        let ctx = infer_train::inference::InferenceContext::from_gguf(
            model_path,
            Device::Gpu,
        )
        .expect("Failed to load model");

        let model = &ctx.model;
        let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");
        let device = gpu_ctx.device();

        let hidden_size = model.config.hidden_size;

        // Test 1: CPU embedding lookup + GPU upload
        println!("Test 1: CPU Embedding Lookup + GPU Upload");
        let iterations = 100;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let token = 1u32;
            let hidden =
                model.embedding_lookup_impl(token).expect("Embedding failed");
            let _buffer = device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
        }
        let cpu_time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("  Time: {:.2} μs", cpu_time);

        // Test 2: Just GPU buffer creation (for comparison)
        println!("\nTest 2: GPU Buffer Creation Only");
        let dummy_data = vec![0.0f32; hidden_size];
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _buffer = device.new_buffer_with_data(
                dummy_data.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
        }
        let buffer_time =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("  Time: {:.2} μs", buffer_time);

        // Test 3: Just CPU lookup
        println!("\nTest 3: CPU Embedding Lookup Only");
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let token = 1u32;
            let _hidden =
                model.embedding_lookup_impl(token).expect("Embedding failed");
        }
        let lookup_time =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("  Time: {:.2} μs", lookup_time);

        println!("\n=== Summary ===");
        println!("CPU lookup: {:.2} μs", lookup_time);
        println!("GPU buffer: {:.2} μs", buffer_time);
        println!("Total (old): {:.2} μs", cpu_time);

        // Test 4: Cached buffer approach
        println!("\nTest 4: Cached Buffer Approach");
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let token = 1u32;
            let hidden =
                model.embedding_lookup_impl(token).expect("Embedding failed");
            let buffer = gpu_ctx.get_or_create_embed_buffer(hidden_size * 4);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    hidden.as_ptr() as *const std::ffi::c_void,
                    buffer.contents(),
                    hidden_size * 4,
                );
            }
        }
        let cached_time =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("  Time: {:.2} μs", cached_time);

        println!("\n=== Final Summary ===");
        println!("CPU lookup: {:.2} μs", lookup_time);
        println!("GPU buffer alloc: {:.2} μs", buffer_time);
        println!("Old approach: {:.2} μs", cpu_time);
        println!("Cached buffer: {:.2} μs", cached_time);
        let improvement = (cpu_time - cached_time) / cpu_time * 100.0;
        println!("Improvement: {:.1}%", improvement);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("This benchmark requires Metal support");
}

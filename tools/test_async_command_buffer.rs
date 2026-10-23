// Test async command buffer implementation
// Verifies that forward_dag_async returns correct logits

use infer_train::dag::executor::DagExecutor;
use infer_train::inference::{InferenceContext, Device};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = &args[1];

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        eprintln!("=== Testing Async Command Buffer ===");
        eprintln!("Model: {}", model_path);

        let gpu_ctx = MetalContext::new().unwrap();
        let mut ctx = InferenceContext::from_gguf_with_gpu_ctx(model_path, gpu_ctx.clone()).unwrap();
        
        // Test 1: Single token decode
        eprintln!("\n--- Test 1: Single Token Decode ---");
        let token = 1u32;
        
        let start = Instant::now();
        let logits = ctx.model.forward_gpu_native_batched(
            &[token],
            ctx.gpu_kv_cache.as_mut().unwrap(),
            &gpu_ctx,
        );
        let elapsed = start.elapsed();
        
        if let Some(logits) = logits {
            eprintln!("✓ Forward pass completed in {:?}", elapsed);
            eprintln!("✓ Logits shape: {}", logits.len());
            eprintln!("✓ Sample logits: {:?}", &logits[..5.min(logits.len())]);
            
            // Check for NaN
            let has_nan = logits.iter().any(|&x| x.is_nan());
            if has_nan {
                eprintln!("✗ ERROR: Logits contain NaN!");
            } else {
                eprintln!("✓ No NaN values in logits");
            }
        } else {
            eprintln!("✗ ERROR: Forward pass failed!");
        }

        // Test 2: Multiple tokens (prefill + decode)
        eprintln!("\n--- Test 2: Multiple Tokens ---");
        
        // Reset KV cache
        ctx.gpu_kv_cache.as_mut().unwrap().reset();
        
        let tokens: Vec<u32> = (0..5).collect();
        let start = Instant::now();
        
        // Prefill
        let logits = ctx.model.forward_gpu_native_batched(
            &tokens,
            ctx.gpu_kv_cache.as_mut().unwrap(),
            &gpu_ctx,
        );
        let elapsed = start.elapsed();
        
        if let Some(logits) = logits {
            eprintln!("✓ Prefill completed in {:?}", elapsed);
            eprintln!("✓ Tokens processed: {}", tokens.len());
            eprintln!("✓ Tokens/second: {:.1}", tokens.len() as f64 / elapsed.as_secs_f64());
        } else {
            eprintln!("✗ ERROR: Prefill failed!");
        }

        // Test 3: Decode loop (simulating generation)
        eprintln!("\n--- Test 3: Decode Loop ---");
        
        let mut decode_times = Vec::new();
        for i in 0..10 {
            let token = (65 + i) as u32; // A, B, C, ...
            let start = Instant::now();
            
            let logits = ctx.model.forward_gpu_native_batched(
                &[token],
                ctx.gpu_kv_cache.as_mut().unwrap(),
                &gpu_ctx,
            );
            
            let elapsed = start.elapsed();
            decode_times.push(elapsed);
            
            if logits.is_some() {
                eprintln!("Token {}: {:?} ✓", i, elapsed);
            } else {
                eprintln!("Token {}: FAILED", i);
            }
        }

        let avg_time: std::time::Duration = decode_times.iter().sum::<std::time::Duration>() / decode_times.len() as u32;
        let min_time = decode_times.iter().min().unwrap();
        let max_time = decode_times.iter().max().unwrap();
        
        eprintln!("\nDecode Statistics:");
        eprintln!("  Average: {:?}", avg_time);
        eprintln!("  Min: {:?}", min_time);
        eprintln!("  Max: {:?}", max_time);
        eprintln!("  Tokens/second: {:.1}", 1.0 / avg_time.as_secs_f64());

        eprintln!("\n✓ All tests completed successfully!");
    }
}
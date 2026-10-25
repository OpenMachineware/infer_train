// Test async pipeline CPU-GPU overlap
// Run: cargo run --release --features metal --bin test-async-overlap -- model.gguf

use infer_train::inference::{Device, InferenceContext};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let mut gpu_ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load GPU model");

    // Prefill
    let _ = gpu_ctx.forward_gpu_batched(&[1]);
    let _ = gpu_ctx.forward_gpu_batched(&[1]);

    // Measure async pipeline
    println!("\n=== Async Pipeline Test ===");

    // Submit token 1 (async)
    let start1 = std::time::Instant::now();
    let result1 = gpu_ctx.forward_gpu_batched(&[100]);
    let submit_time1 = start1.elapsed();
    println!("Submit 1: {:?} (result={:?})", submit_time1, result1.is_some());

    // Do CPU work (simulate sampling + logic)
    std::thread::sleep(std::time::Duration::from_millis(2));
    println!("CPU work: 2ms");

    // Submit token 2 (async) - should wait for previous CB
    let start2 = std::time::Instant::now();
    let result2 = gpu_ctx.forward_gpu_batched(&[101]);
    let total_time2 = start2.elapsed();
    println!("Submit 2: {:?} (result={:?})", total_time2, result2.is_some());

    // Check if overlap is working
    println!("\nAnalysis:");
    if total_time2.as_micros() < 10000 {
        println!("✓ Overlap working: Total time < 10ms");
        println!("  GPU was executing while CPU was sleeping");
    } else {
        println!("✗ No overlap: Total time > 10ms");
        println!("  CPU wait blocked GPU execution");
    }
}

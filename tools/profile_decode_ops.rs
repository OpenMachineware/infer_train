// Profile decode operations to find bottlenecks

use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use std::time::Instant;

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    let gpu_ctx = MetalContext::new().unwrap();
    let device = gpu_ctx.device();
    let mut ctx =
        InferenceContext::from_gguf_with_gpu_ctx(model_path, gpu_ctx.clone())
            .unwrap();
    let config = ctx.model.config.clone();
    let mut kv = KVCacheGPU::new(device, &config, 2048);

    // Warmup prefill
    let _ = ctx.model.forward_gpu_native_batched(&[1], &mut kv, &gpu_ctx);

    println!("=== Decode Operation Profiling ===\n");

    // Profile 10 decode steps
    let mut total_time = std::time::Duration::ZERO;
    for i in 0..10 {
        let token = (2 + i) as u32;
        let start = Instant::now();
        let result =
            ctx.model.forward_gpu_native_batched(&[token], &mut kv, &gpu_ctx);
        let elapsed = start.elapsed();

        if let Some(logits) = result {
            let has_nan = logits.iter().any(|&v| v.is_nan());
            println!("Token {}: {:?} (nan: {})", i, elapsed, has_nan);
        } else {
            println!("Token {}: FAILED", i);
        }

        total_time += elapsed;
    }

    println!("\nTotal: {:?}", total_time);
    println!("Average: {:?}", total_time / 10);
    println!("Throughput: {:.1} t/s", 10.0 / total_time.as_secs_f64());
}

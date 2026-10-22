#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::{Device, InferenceContext};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = &args[1];

    let mut gpu_ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();

    // Measure prefill with different token counts
    for &n_tokens in &[1, 10, 32, 64] {
        let tokens: Vec<u32> = (0..n_tokens).collect();

        // Reset KV cache
        gpu_ctx.reset_kv_cache();

        let start = std::time::Instant::now();
        let _logits = gpu_ctx.forward_gpu_batched(&tokens).unwrap();
        let elapsed = start.elapsed();

        let ms = elapsed.as_secs_f64() * 1000.0;
        let tps = n_tokens as f64 / elapsed.as_secs_f64();

        println!("Prefill {} tokens: {:.2} ms ({:.1} t/s)", n_tokens, ms, tps);
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() { println!("Requires Metal"); }

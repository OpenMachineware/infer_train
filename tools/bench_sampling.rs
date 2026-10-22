// Measure sampling overhead more accurately
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::{Device, InferenceContext};
use std::hint::black_box;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = &args[1];

    let mut gpu_ctx =
        InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();

    // Get logits from a forward pass
    let logits = gpu_ctx.forward_gpu_batched(&[1]).unwrap();
    println!("Logits len: {}", logits.len());

    // Warm up
    for _ in 0..10 {
        let _token = black_box(
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i as u32),
        );
    }

    // Measure sampling (greedy argmax)
    let iterations = 1000;
    let start = std::time::Instant::now();
    let mut result = 0u32;
    for _ in 0..iterations {
        result = black_box(
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i as u32)
                .unwrap_or(0),
        );
    }
    let elapsed = start.elapsed();

    let ns_per_sample = elapsed.as_nanos() as f64 / iterations as f64;
    println!("Result: {}", result);
    println!(
        "Sampling (argmax over {} logits): {:.2} µs/sample",
        logits.len(),
        ns_per_sample / 1000.0
    );

    // Estimate memory bandwidth
    let bytes = logits.len() * 4;
    let bandwidth_gbps = (bytes as f64 / (ns_per_sample / 1e9)) / 1e9;
    println!("Memory bandwidth: {:.2} GB/s", bandwidth_gbps);
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("Requires Metal");
}

use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use std::time::Instant;

fn main() {
    let ctx = MetalContext::new().unwrap();
    let size = 4096usize;
    let data = vec![1.0f32; size];
    let weight = vec![1.0f32; size];

    // Extended warmup
    println!("Warming up...");
    for _ in 0..50 {
        let _ = ctx.rms_norm_f32(1, size, &data, &weight, 1e-5);
        let _ = ctx.softmax_f32_fast(1, size, &data);
    }

    // Test with more rows
    println!("\nTesting single row (n_rows=1):");
    let mut times = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let _ = ctx.rms_norm_f32(1, size, &data, &weight, 1e-5);
        times.push(start.elapsed().as_micros());
    }
    let avg: f64 = times.iter().sum::<u128>() as f64 / times.len() as f64;
    println!("  RMS Norm: {:.2} µs", avg);

    let mut times = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let _ = ctx.softmax_f32_fast(1, size, &data);
        times.push(start.elapsed().as_micros());
    }
    let avg: f64 = times.iter().sum::<u128>() as f64 / times.len() as f64;
    println!("  Softmax:  {:.2} µs", avg);

    // Test with multiple rows (like llama.cpp actually uses)
    println!("\nTesting multiple rows (n_rows=32):");
    let data32: Vec<f32> = vec![1.0f32; size * 32];
    let mut times = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let _ = ctx.rms_norm_f32(32, size, &data32, &weight, 1e-5);
        times.push(start.elapsed().as_micros());
    }
    let avg: f64 = times.iter().sum::<u128>() as f64 / times.len() as f64;
    println!("  RMS Norm: {:.2} µs ({:.2} µs/row)", avg, avg / 32.0);

    let mut times = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let _ = ctx.softmax_f32_fast(32, size, &data32);
        times.push(start.elapsed().as_micros());
    }
    let avg: f64 = times.iter().sum::<u128>() as f64 / times.len() as f64;
    println!("  Softmax:  {:.2} µs ({:.2} µs/row)", avg, avg / 32.0);
}

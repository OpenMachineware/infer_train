// Simple FFN benchmark to isolate performance

use infer_train::inference::{Device, InferenceContext};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    let layer = &ctx.model.layers[0];
    let hidden_size = ctx.config.hidden_size;

    // Create input
    let hidden: Vec<f32> = vec![0.5; hidden_size];

    println!("=== FFN Benchmark ===");
    println!("Hidden size: {}", hidden_size);

    if let (Some(w1), Some(w2), Some(w3)) = (&layer.w1, &layer.w2, &layer.w3) {
        let intermediate_size = w1.shape[1];
        println!("Intermediate size: {}", intermediate_size);
        println!("w1 dtype: {:?}", w1.dtype);
        println!("w2 dtype: {:?}", w2.dtype);
        println!("w3 dtype: {:?}", w3.dtype);

        // Benchmark each GEMV separately
        println!("\n=== Individual GEMVs ===");

        // Gate projection
        for _ in 0..5 {
            let _ = ctx.model.gemv_impl(&hidden, w1, intermediate_size);
        }
        let t = std::time::Instant::now();
        for _ in 0..10 {
            let _ = ctx.model.gemv_impl(&hidden, w1, intermediate_size);
        }
        println!("Gate projection: {:?}", t.elapsed() / 10);

        // Up projection
        for _ in 0..5 {
            let _ = ctx.model.gemv_impl(&hidden, w3, intermediate_size);
        }
        let t = std::time::Instant::now();
        for _ in 0..10 {
            let _ = ctx.model.gemv_impl(&hidden, w3, intermediate_size);
        }
        println!("Up projection: {:?}", t.elapsed() / 10);

        // Down projection (needs intermediate input)
        let gate = ctx.model.gemv_impl(&hidden, w1, intermediate_size).unwrap();
        let up = ctx.model.gemv_impl(&hidden, w3, intermediate_size).unwrap();
        let hidden_ffn: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| g * (u / (1.0 + (-u).exp())))
            .collect();

        for _ in 0..5 {
            let _ = ctx.model.gemv_impl(&hidden_ffn, w2, hidden_size);
        }
        let t = std::time::Instant::now();
        for _ in 0..10 {
            let _ = ctx.model.gemv_impl(&hidden_ffn, w2, hidden_size);
        }
        println!("Down projection: {:?}", t.elapsed() / 10);

        // Benchmark full FFN
        println!("\n=== Full FFN (compute_ffn) ===");
        for _ in 0..5 {
            let _ = ctx.model.compute_ffn(layer, &hidden, hidden_size);
        }
        let t = std::time::Instant::now();
        for _ in 0..10 {
            let _ = ctx.model.compute_ffn(layer, &hidden, hidden_size);
        }
        println!("Full FFN: {:?}", t.elapsed() / 10);

        // Also benchmark ffn_swiglu
        println!("\n=== Full FFN (ffn_swiglu) ===");
        for _ in 0..5 {
            let _ = ctx.model.ffn_swiglu(&hidden, layer, hidden_size);
        }
        let t = std::time::Instant::now();
        for _ in 0..10 {
            let _ = ctx.model.ffn_swiglu(&hidden, layer, hidden_size);
        }
        println!("Full FFN (ffn_swiglu): {:?}", t.elapsed() / 10);
    }
}

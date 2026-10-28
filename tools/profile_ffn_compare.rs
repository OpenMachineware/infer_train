// Compare FFN timing between different methods

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
    let hidden: Vec<f32> = vec![0.5; hidden_size];

    println!("=== Method 1: Direct calls ===");
    // Warmup
    for _ in 0..5 {
        let _ = ctx.model.compute_ffn(layer, &hidden, hidden_size);
    }
    // Measure
    let t = std::time::Instant::now();
    for _ in 0..20 {
        let _ = ctx.model.compute_ffn(layer, &hidden, hidden_size);
    }
    println!("compute_ffn (avg of 20): {:?}", t.elapsed() / 20);

    println!("\n=== Method 2: Step-by-step measurement ===");
    let layer = &ctx.model.layers[0];
    let w1 = layer.w1.as_ref().unwrap();
    let w2 = layer.w2.as_ref().unwrap();
    let w3 = layer.w3.as_ref().unwrap();
    let intermediate_size = w1.shape[1];

    // Warmup
    let _ = ctx.model.gemv_impl(&hidden, w1, intermediate_size);

    let mut total_time = std::time::Duration::ZERO;

    for _ in 0..20 {
        let t0 = std::time::Instant::now();
        let gate = ctx.model.gemv_impl(&hidden, w1, intermediate_size).unwrap();
        total_time += t0.elapsed();

        let t1 = std::time::Instant::now();
        let up = ctx.model.gemv_impl(&hidden, w3, intermediate_size).unwrap();
        total_time += t1.elapsed();

        let t2 = std::time::Instant::now();
        let hidden_ffn: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| g * (u / (1.0 + (-u).exp())))
            .collect();
        total_time += t2.elapsed();

        let t3 = std::time::Instant::now();
        let _ = ctx.model.gemv_impl(&hidden_ffn, w2, hidden_size);
        total_time += t3.elapsed();
    }
    println!("Step-by-step (avg of 20): {:?}", total_time / 20);

    // Now profile what the profile-cpu-detailed tool does
    println!("\n=== Method 3: Same as profile-cpu-detailed ===");
    // The profile tool calls compute_ffn like this:
    // let ffn_out = ctx.model.compute_ffn(layer, &ffn_normed, hidden_size).unwrap();

    let ffn_normed: Vec<f32> = vec![0.5; hidden_size]; // Same as the profile tool's input

    // Warmup
    for _ in 0..5 {
        let _ = ctx.model.compute_ffn(layer, &ffn_normed, hidden_size);
    }

    // Single measurement like the profile tool does
    let t = std::time::Instant::now();
    let _ = ctx.model.compute_ffn(layer, &ffn_normed, hidden_size);
    println!("Single call (like profile tool): {:?}", t.elapsed());

    // Multiple measurements
    let t = std::time::Instant::now();
    for _ in 0..20 {
        let _ = ctx.model.compute_ffn(layer, &ffn_normed, hidden_size);
    }
    println!("Multiple calls (avg of 20): {:?}", t.elapsed() / 20);
}

// Detailed FFN profiling with timing breakdown

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

    let w1 = layer.w1.as_ref().unwrap();
    let w2 = layer.w2.as_ref().unwrap();
    let w3 = layer.w3.as_ref().unwrap();
    let intermediate_size = w1.shape[1];

    println!(
        "Hidden size: {}, Intermediate: {}",
        hidden_size, intermediate_size
    );

    // Profile a single call in detail
    println!("\n=== Single call breakdown ===");

    let mut gate_time = std::time::Duration::ZERO;
    let mut up_time = std::time::Duration::ZERO;
    let mut silu_time = std::time::Duration::ZERO;
    let mut down_time = std::time::Duration::ZERO;
    let mut total_time = std::time::Duration::ZERO;

    for _ in 0..20 {
        let t_total = std::time::Instant::now();

        let t0 = std::time::Instant::now();
        let gate = ctx.model.gemv_impl(&hidden, w1, intermediate_size).unwrap();
        gate_time += t0.elapsed();

        let t1 = std::time::Instant::now();
        let up = ctx.model.gemv_impl(&hidden, w3, intermediate_size).unwrap();
        up_time += t1.elapsed();

        let t2 = std::time::Instant::now();
        let hidden_ffn: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| g * (u / (1.0 + (-u).exp())))
            .collect();
        silu_time += t2.elapsed();

        let t3 = std::time::Instant::now();
        let _ = ctx.model.gemv_impl(&hidden_ffn, w2, hidden_size);
        down_time += t3.elapsed();

        total_time += t_total.elapsed();
    }

    println!("Gate GEMV: {:?}", gate_time / 20);
    println!("Up GEMV: {:?}", up_time / 20);
    println!("SiLU: {:?}", silu_time / 20);
    println!("Down GEMV: {:?}", down_time / 20);
    println!("Sum: {:?}", (gate_time + up_time + silu_time + down_time) / 20);
    println!("Total: {:?}", total_time / 20);

    // Now test what profile-cpu-detailed does
    println!("\n=== Simulating profile-cpu-detailed ===");

    // The profile tool does this before calling compute_ffn:
    let residual: Vec<f32> = vec![0.5; hidden_size];
    let ffn_norm = layer.ffn_norm.as_ref().unwrap();

    // Apply FFN norm
    let ffn_normed = ctx
        .model
        .apply_rms_norm_impl(&residual, ffn_norm, hidden_size)
        .unwrap();

    // Now measure compute_ffn
    let t = std::time::Instant::now();
    for _ in 0..20 {
        let _ = ctx.model.compute_ffn(layer, &ffn_normed, hidden_size);
    }
    println!("compute_ffn with normed input: {:?}", t.elapsed() / 20);

    // Also try with the exact same input as the individual steps
    let t = std::time::Instant::now();
    for _ in 0..20 {
        let _ = ctx.model.compute_ffn(layer, &hidden, hidden_size);
    }
    println!("compute_ffn with hidden input: {:?}", t.elapsed() / 20);
}

// Benchmark SiLU activation performance

fn main() {
    let intermediate_size = 14336; // Typical FFN intermediate size for 7B model

    // Create dummy data
    let gate: Vec<f32> =
        (0..intermediate_size).map(|i| (i as f32 % 10.0) - 5.0).collect();
    let up: Vec<f32> = (0..intermediate_size)
        .map(|i| ((i + 100) as f32 % 10.0) - 5.0)
        .collect();

    // Warmup
    for _ in 0..10 {
        let _: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| g * (u / (1.0 + (-u).exp())))
            .collect();
    }

    // Benchmark scalar loop
    let n_runs = 100;
    let start = std::time::Instant::now();
    for _ in 0..n_runs {
        let _: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| g * (u / (1.0 + (-u).exp())))
            .collect();
    }
    let scalar_time = start.elapsed();
    let avg_ms = scalar_time.as_millis() as f64 / n_runs as f64;
    println!("Scalar SiLU (gate * silu(up)): {:.3} ms", avg_ms);

    // Benchmark alternative formula (silu(gate) * up)
    let start = std::time::Instant::now();
    for _ in 0..n_runs {
        let _: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
    }
    let alt_time = start.elapsed();
    let avg_ms = alt_time.as_millis() as f64 / n_runs as f64;
    println!("Scalar SiLU (silu(gate) * up): {:.3} ms", avg_ms);

    // Benchmark with fast_sigmoid approximation
    fn fast_sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + x.abs())
    }
    let start = std::time::Instant::now();
    for _ in 0..n_runs {
        let _: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| g * fast_sigmoid(-g) * u)
            .collect();
    }
    let approx_time = start.elapsed();
    let avg_ms = approx_time.as_millis() as f64 / n_runs as f64;
    println!("Approx SiLU (fast sigmoid): {:.3} ms", avg_ms);
}

// Comprehensive Rust K-quant vec_dot benchmark
use infer_train::quant::vec_dot::arm::*;
use infer_train::quant::types::*;
use half::f16;
use std::time::Instant;

fn generate_q8_k_input(k: usize) -> Vec<BlockQ8K> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for j in 0..256 {
                qs[j] = (((i * 17 + j * 13) % 256) as i32 - 128) as i8;
            }
            for j in 0..16 {
                bsums[j] = ((i * 23 + j * 7) % 256) as i16;
            }
            BlockQ8K {
                d: 0.01 + (i % 10) as f32 * 0.001,
                qs,
                bsums,
            }
        })
        .collect()
}

fn generate_q2_k_weights(k: usize) -> Vec<BlockQ2K> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 64];
            let mut scales = [0u8; 16];
            for j in 0..64 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..16 {
                scales[j] = ((i * 23 + j * 7) % 256) as u8;
            }
            BlockQ2K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn generate_q3_k_weights(k: usize) -> Vec<BlockQ3K> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 64];
            let mut hmask = [0u8; 32];
            let mut scales = [0u8; 12];
            for j in 0..64 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..32 {
                hmask[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i * 23 + j * 7) % 256) as u8;
            }
            BlockQ3K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                hmask,
                qs,
                scales,
            }
        })
        .collect()
}

fn generate_q4_k_weights(k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i * 23 + j * 7) % 256) as u8;
            }
            BlockQ4K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.1 + (i % 5) as f32 * 0.05).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn generate_q5_k_weights(k: usize) -> Vec<BlockQ5K> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut qh = [0u8; 32];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..32 {
                qh[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i * 23 + j * 7) % 256) as u8;
            }
            BlockQ5K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qh,
                qs,
            }
        })
        .collect()
}

fn generate_q6_k_weights(k: usize) -> Vec<BlockQ6K> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut scales = [0i8; 16];
            for j in 0..128 {
                ql[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..64 {
                qh[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            for j in 0..16 {
                scales[j] = ((i * 23 + j * 7) % 128) as i8 - 64;
            }
            BlockQ6K {
                ql,
                qh,
                scales,
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            }
        })
        .collect()
}

fn main() {
    unsafe {
        let k_values = [256, 1024, 4096];

        println!("┌──────────────────────────────────────────────────────────┐");
        println!("│       Rust K-quant vec_dot Performance                  │");
        println!("├──────────────────────────────────────────────────────────┤");
        println!("│  Format │ K=256  │ K=1024 │ K=4096 │ Avg   │ Stability  │");
        println!("├──────────────────────────────────────────────────────────┤");

        // Q2_K
        {
            let mut results = Vec::new();
            for &k in &k_values {
                let nb = k / 256;
                let weights = generate_q2_k_weights(k);
                let x = generate_q8_k_input(k);

                for _ in 0..100 {
                    let _ = vec_dot_q2_k_q8_k_neon(k, &weights[0..nb], &x);
                }

                let iterations = if k <= 256 { 1000000 } else { 100000 };
                let start = Instant::now();
                for _ in 0..iterations {
                    let _ = vec_dot_q2_k_q8_k_neon(k, &weights[0..nb], &x);
                }
                let elapsed = start.elapsed().as_secs_f64();
                let gflops = 2.0 * (k as f64) * (iterations as f64) / elapsed / 1e9;
                results.push(gflops);
            }

            let avg = (results[0] + results[1] + results[2]) / 3.0;
            let max_dev = results
                .iter()
                .map(|&r| (r - avg).abs() / avg * 100.0)
                .fold(0.0, f64::max);

            println!(
                "│  Q2_K   │ {:5.1} │ {:6.1} │ {:6.1} │ {:5.1} │ {:4.1}%     │",
                results[0], results[1], results[2], avg, max_dev
            );
        }

        // Q3_K
        {
            let mut results = Vec::new();
            for &k in &k_values {
                let nb = k / 256;
                let weights = generate_q3_k_weights(k);
                let x = generate_q8_k_input(k);

                for _ in 0..100 {
                    let _ = vec_dot_q3_k_q8_k_neon(k, &weights[0..nb], &x);
                }

                let iterations = if k <= 256 { 1000000 } else { 100000 };
                let start = Instant::now();
                for _ in 0..iterations {
                    let _ = vec_dot_q3_k_q8_k_neon(k, &weights[0..nb], &x);
                }
                let elapsed = start.elapsed().as_secs_f64();
                let gflops = 2.0 * (k as f64) * (iterations as f64) / elapsed / 1e9;
                results.push(gflops);
            }

            let avg = (results[0] + results[1] + results[2]) / 3.0;
            let max_dev = results
                .iter()
                .map(|&r| (r - avg).abs() / avg * 100.0)
                .fold(0.0, f64::max);

            println!(
                "│  Q3_K   │ {:5.1} │ {:6.1} │ {:6.1} │ {:5.1} │ {:4.1}%     │",
                results[0], results[1], results[2], avg, max_dev
            );
        }

        // Q4_K
        {
            let mut results = Vec::new();
            for &k in &k_values {
                let nb = k / 256;
                let weights = generate_q4_k_weights(k);
                let x = generate_q8_k_input(k);

                for _ in 0..100 {
                    let _ = vec_dot_q4_k_q8_k_neon(k, &weights[0..nb], &x);
                }

                let iterations = if k <= 256 { 1000000 } else { 100000 };
                let start = Instant::now();
                for _ in 0..iterations {
                    let _ = vec_dot_q4_k_q8_k_neon(k, &weights[0..nb], &x);
                }
                let elapsed = start.elapsed().as_secs_f64();
                let gflops = 2.0 * (k as f64) * (iterations as f64) / elapsed / 1e9;
                results.push(gflops);
            }

            let avg = (results[0] + results[1] + results[2]) / 3.0;
            let max_dev = results
                .iter()
                .map(|&r| (r - avg).abs() / avg * 100.0)
                .fold(0.0, f64::max);

            println!(
                "│  Q4_K   │ {:5.1} │ {:6.1} │ {:6.1} │ {:5.1} │ {:4.1}%     │",
                results[0], results[1], results[2], avg, max_dev
            );
        }

        // Q5_K
        {
            let mut results = Vec::new();
            for &k in &k_values {
                let nb = k / 256;
                let weights = generate_q5_k_weights(k);
                let x = generate_q8_k_input(k);

                for _ in 0..100 {
                    let _ = vec_dot_q5_k_q8_k_neon(k, &weights[0..nb], &x);
                }

                let iterations = if k <= 256 { 1000000 } else { 100000 };
                let start = Instant::now();
                for _ in 0..iterations {
                    let _ = vec_dot_q5_k_q8_k_neon(k, &weights[0..nb], &x);
                }
                let elapsed = start.elapsed().as_secs_f64();
                let gflops = 2.0 * (k as f64) * (iterations as f64) / elapsed / 1e9;
                results.push(gflops);
            }

            let avg = (results[0] + results[1] + results[2]) / 3.0;
            let max_dev = results
                .iter()
                .map(|&r| (r - avg).abs() / avg * 100.0)
                .fold(0.0, f64::max);

            println!(
                "│  Q5_K   │ {:5.1} │ {:6.1} │ {:6.1} │ {:5.1} │ {:4.1}%     │",
                results[0], results[1], results[2], avg, max_dev
            );
        }

        // Q6_K
        {
            let mut results = Vec::new();
            for &k in &k_values {
                let nb = k / 256;
                let weights = generate_q6_k_weights(k);
                let x = generate_q8_k_input(k);

                for _ in 0..100 {
                    let _ = vec_dot_q6_k_q8_k_neon(k, &weights[0..nb], &x);
                }

                let iterations = if k <= 256 { 1000000 } else { 100000 };
                let start = Instant::now();
                for _ in 0..iterations {
                    let _ = vec_dot_q6_k_q8_k_neon(k, &weights[0..nb], &x);
                }
                let elapsed = start.elapsed().as_secs_f64();
                let gflops = 2.0 * (k as f64) * (iterations as f64) / elapsed / 1e9;
                results.push(gflops);
            }

            let avg = (results[0] + results[1] + results[2]) / 3.0;
            let max_dev = results
                .iter()
                .map(|&r| (r - avg).abs() / avg * 100.0)
                .fold(0.0, f64::max);

            println!(
                "│  Q6_K   │ {:5.1} │ {:6.1} │ {:6.1} │ {:5.1} │ {:4.1}%     │",
                results[0], results[1], results[2], avg, max_dev
            );
        }

        println!("└──────────────────────────────────────────────────────────┘");
    }
}

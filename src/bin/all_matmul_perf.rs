// Compare Rust matmul performance with llama.cpp
// llama.cpp vec_dot baseline from previous tests:
// Q4_K: 82 GFLOPS, Q2_K: 67 GFLOPS, Q3_K: 50 GFLOPS, Q5_K: 70 GFLOPS
// Q6_K: 85 GFLOPS
use infer_train::quant::matmul::*;
use infer_train::quant::types::*;
use infer_train::quant::vec_dot::arm::*;
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

fn generate_q8_0_input(k: usize) -> Vec<BlockQ8_0> {
    let nb = k / 32;
    (0..nb)
        .map(|i| {
            let mut qs = [0i8; 32];
            for j in 0..32 {
                qs[j] = (((i * 17 + j * 13) % 256) as i32 - 128) as i8;
            }
            BlockQ8_0 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_q4_k_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..m * nb)
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

fn generate_q5_k_weights(m: usize, k: usize) -> Vec<BlockQ5K> {
    let nb = k / 256;
    (0..m * nb)
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

fn generate_q6_k_weights(m: usize, k: usize) -> Vec<BlockQ6K> {
    let nb = k / 256;
    (0..m * nb)
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

fn generate_q3_k_weights(m: usize, k: usize) -> Vec<BlockQ3K> {
    let nb = k / 256;
    (0..m * nb)
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

fn generate_q2_k_weights(m: usize, k: usize) -> Vec<BlockQ2K> {
    let nb = k / 256;
    (0..m * nb)
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

fn generate_q4_0_weights(m: usize, k: usize) -> Vec<BlockQ4_0> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockQ4_0 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_q5_0_weights(m: usize, k: usize) -> Vec<BlockQ5_0> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 16];
            let mut qh = [0u8; 4];
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..4 {
                qh[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            BlockQ5_0 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qh,
                qs,
            }
        })
        .collect()
}

// Baseline GFLOPS from llama.cpp vec_dot benchmarks
const LLAMACPP_BASELINE: &[(f64, f64)] = &[
    (82.0, 82.0),   // Q4_K
    (67.0, 67.0),   // Q2_K
    (50.0, 50.0),   // Q3_K
    (70.0, 70.0),   // Q5_K
    (85.0, 85.0),   // Q6_K
    (0.0, 0.0),     // Q4_0 (placeholder)
    (0.0, 0.0),     // Q5_0 (placeholder)
];

fn main() {
    unsafe {
        let k = 4096;
        let m_large = 4096;

        println!("┌────────────────────────────────────────────────────────────────┐");
        println!("│       Rust Matmul Performance vs llama.cpp (K=4096)           │");
        println!("├────────────────────────────────────────────────────────────────┤");
        println!("│  Format  │ M=1 GFLOPS │ M=4096 GFLOPS │ vs baseline │ Status  │");
        println!("├────────────────────────────────────────────────────────────────┤");

        // Test Q4_K (optimized)
        {
            // M=1 test
            let weights_m1 = generate_q4_k_weights(1, k);
            let x = generate_q8_k_input(k);
            let mut dst = [0.0f32; 1];

            let start = Instant::now();
            for _ in 0..10000 {
                matmul_q4_k_q8_k(&weights_m1, &x, &mut dst, k, 1);
            }
            let m1_gflops = 2.0 * 1.0 * (k as f64) * 10000.0 / start.elapsed().as_secs_f64() / 1e9;

            // M=4096 test
            let weights_m4096 = generate_q4_k_weights(m_large, k);
            let mut dst_m4096 = vec![0.0f32; m_large];

            let start = Instant::now();
            for _ in 0..100 {
                matmul_q4_k_q8_k(&weights_m4096, &x, &mut dst_m4096, k, m_large);
            }
            let m4096_gflops = 2.0 * (m_large as f64) * (k as f64) * 100.0
                / start.elapsed().as_secs_f64()
                / 1e9;

            let ratio = (m1_gflops + m4096_gflops) / 2.0 / 82.0;
            let status = if ratio >= 1.0 { "✓" } else if ratio >= 0.9 { "≈" } else { "✗" };
            println!(
                "│  Q4_K    │ {:10.1} │ {:13.1} │    {:5.0}%    │   {}     │",
                m1_gflops,
                m4096_gflops,
                ratio * 100.0,
                status
            );
        }

        // Test Q2_K
        {
            // M=1 test
            let weights_m1 = generate_q2_k_weights(1, k);
            let x = generate_q8_k_input(k);
            let mut dst = [0.0f32; 1];

            let start = Instant::now();
            for _ in 0..10000 {
                matmul_q2_k_q8_k(&weights_m1, &x, &mut dst, k, 1);
            }
            let m1_gflops = 2.0 * 1.0 * (k as f64) * 10000.0 / start.elapsed().as_secs_f64() / 1e9;

            // M=4096 test
            let weights_m4096 = generate_q2_k_weights(m_large, k);
            let mut dst_m4096 = vec![0.0f32; m_large];

            let start = Instant::now();
            for _ in 0..100 {
                matmul_q2_k_q8_k(&weights_m4096, &x, &mut dst_m4096, k, m_large);
            }
            let m4096_gflops = 2.0 * (m_large as f64) * (k as f64) * 100.0
                / start.elapsed().as_secs_f64()
                / 1e9;

            let ratio = (m1_gflops + m4096_gflops) / 2.0 / 67.0;
            let status = if ratio >= 1.0 { "✓" } else if ratio >= 0.9 { "≈" } else { "✗" };
            println!(
                "│  Q2_K    │ {:10.1} │ {:13.1} │    {:5.0}%    │   {}     │",
                m1_gflops,
                m4096_gflops,
                ratio * 100.0,
                status
            );
        }

        // Test Q3_K
        {
            // M=1 test
            let weights_m1 = generate_q3_k_weights(1, k);
            let x = generate_q8_k_input(k);
            let mut dst = [0.0f32; 1];

            let start = Instant::now();
            for _ in 0..10000 {
                matmul_q3_k_q8_k(&weights_m1, &x, &mut dst, k, 1);
            }
            let m1_gflops = 2.0 * 1.0 * (k as f64) * 10000.0 / start.elapsed().as_secs_f64() / 1e9;

            // M=4096 test
            let weights_m4096 = generate_q3_k_weights(m_large, k);
            let mut dst_m4096 = vec![0.0f32; m_large];

            let start = Instant::now();
            for _ in 0..100 {
                matmul_q3_k_q8_k(&weights_m4096, &x, &mut dst_m4096, k, m_large);
            }
            let m4096_gflops = 2.0 * (m_large as f64) * (k as f64) * 100.0
                / start.elapsed().as_secs_f64()
                / 1e9;

            let ratio = (m1_gflops + m4096_gflops) / 2.0 / 50.0;
            let status = if ratio >= 1.0 { "✓" } else if ratio >= 0.9 { "≈" } else { "✗" };
            println!(
                "│  Q3_K    │ {:10.1} │ {:13.1} │    {:5.0}%    │   {}     │",
                m1_gflops,
                m4096_gflops,
                ratio * 100.0,
                status
            );
        }

        // Test Q5_K
        {
            // M=1 test
            let weights_m1 = generate_q5_k_weights(1, k);
            let x = generate_q8_k_input(k);
            let mut dst = [0.0f32; 1];

            let start = Instant::now();
            for _ in 0..10000 {
                matmul_q5_k_q8_k(&weights_m1, &x, &mut dst, k, 1);
            }
            let m1_gflops = 2.0 * 1.0 * (k as f64) * 10000.0 / start.elapsed().as_secs_f64() / 1e9;

            // M=4096 test
            let weights_m4096 = generate_q5_k_weights(m_large, k);
            let mut dst_m4096 = vec![0.0f32; m_large];

            let start = Instant::now();
            for _ in 0..100 {
                matmul_q5_k_q8_k(&weights_m4096, &x, &mut dst_m4096, k, m_large);
            }
            let m4096_gflops = 2.0 * (m_large as f64) * (k as f64) * 100.0
                / start.elapsed().as_secs_f64()
                / 1e9;

            let ratio = (m1_gflops + m4096_gflops) / 2.0 / 70.0;
            let status = if ratio >= 1.0 { "✓" } else if ratio >= 0.9 { "≈" } else { "✗" };
            println!(
                "│  Q5_K    │ {:10.1} │ {:13.1} │    {:5.0}%    │   {}     │",
                m1_gflops,
                m4096_gflops,
                ratio * 100.0,
                status
            );
        }

        // Test Q6_K
        {
            // M=1 test
            let weights_m1 = generate_q6_k_weights(1, k);
            let x = generate_q8_k_input(k);
            let mut dst = [0.0f32; 1];

            let start = Instant::now();
            for _ in 0..10000 {
                matmul_q6_k_q8_k(&weights_m1, &x, &mut dst, k, 1);
            }
            let m1_gflops = 2.0 * 1.0 * (k as f64) * 10000.0 / start.elapsed().as_secs_f64() / 1e9;

            // M=4096 test
            let weights_m4096 = generate_q6_k_weights(m_large, k);
            let mut dst_m4096 = vec![0.0f32; m_large];

            let start = Instant::now();
            for _ in 0..100 {
                matmul_q6_k_q8_k(&weights_m4096, &x, &mut dst_m4096, k, m_large);
            }
            let m4096_gflops = 2.0 * (m_large as f64) * (k as f64) * 100.0
                / start.elapsed().as_secs_f64()
                / 1e9;

            let ratio = (m1_gflops + m4096_gflops) / 2.0 / 85.0;
            let status = if ratio >= 1.0 { "✓" } else if ratio >= 0.9 { "≈" } else { "✗" };
            println!(
                "│  Q6_K    │ {:10.1} │ {:13.1} │    {:5.0}%    │   {}     │",
                m1_gflops,
                m4096_gflops,
                ratio * 100.0,
                status
            );
        }

        // Test Q4_0
        {
            // M=1 test
            let weights_m1 = generate_q4_0_weights(1, k);
            let x = generate_q8_0_input(k);
            let mut dst = [0.0f32; 1];

            let start = Instant::now();
            for _ in 0..10000 {
                matmul_q4_0_q8_0(&weights_m1, &x, &mut dst, k, 1);
            }
            let m1_gflops = 2.0 * 1.0 * (k as f64) * 10000.0 / start.elapsed().as_secs_f64() / 1e9;

            // M=4096 test
            let weights_m4096 = generate_q4_0_weights(m_large, k);
            let mut dst_m4096 = vec![0.0f32; m_large];

            let start = Instant::now();
            for _ in 0..100 {
                matmul_q4_0_q8_0(&weights_m4096, &x, &mut dst_m4096, k, m_large);
            }
            let m4096_gflops = 2.0 * (m_large as f64) * (k as f64) * 100.0
                / start.elapsed().as_secs_f64()
                / 1e9;

            println!(
                "│  Q4_0    │ {:10.1} │ {:13.1} │     N/A     │   -     │",
                m1_gflops, m4096_gflops
            );
        }

        // Test Q5_0
        {
            // M=1 test
            let weights_m1 = generate_q5_0_weights(1, k);
            let x = generate_q8_0_input(k);
            let mut dst = [0.0f32; 1];

            let start = Instant::now();
            for _ in 0..10000 {
                matmul_q5_0_q8_0(&weights_m1, &x, &mut dst, k, 1);
            }
            let m1_gflops = 2.0 * 1.0 * (k as f64) * 10000.0 / start.elapsed().as_secs_f64() / 1e9;

            // M=4096 test
            let weights_m4096 = generate_q5_0_weights(m_large, k);
            let mut dst_m4096 = vec![0.0f32; m_large];

            let start = Instant::now();
            for _ in 0..100 {
                matmul_q5_0_q8_0(&weights_m4096, &x, &mut dst_m4096, k, m_large);
            }
            let m4096_gflops = 2.0 * (m_large as f64) * (k as f64) * 100.0
                / start.elapsed().as_secs_f64()
                / 1e9;

            println!(
                "│  Q5_0    │ {:10.1} │ {:13.1} │     N/A     │   -     │",
                m1_gflops, m4096_gflops
            );
        }

        println!("└────────────────────────────────────────────────────────────────┘");
        println!("\n说明:");
        println!("  - M=1 GFLOPS: decode性能（单行matmul）");
        println!("  - M=4096 GFLOPS: batch性能（4096行matmul）");
        println!("  - vs baseline: 相对于llama.cpp vec_dot基准的百分比");
        println!("  - ✓: 超过llama.cpp, ≈: 接近(90%+), ✗: 低于90%");
    }
}

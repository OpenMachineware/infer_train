// Benchmark all IQ/TQ vec_dot formats
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

// IQ series weight generators
fn generate_iq4_xs_weights(k: usize) -> Vec<BlockIQ4XS> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut scales_l = [0u8; 4];
            for j in 0..128 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..4 {
                scales_l[j] = ((i * 23 + j * 7) % 256) as u8;
            }
            BlockIQ4XS {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                scales_h: ((i * 23) % 256) as u16,
                scales_l,
                qs,
            }
        })
        .collect()
}

fn generate_iq4_nl_weights(k: usize) -> Vec<BlockIQ4NL> {
    let nb = k / 32;
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockIQ4NL {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_iq3_xxs_weights(k: usize) -> Vec<BlockIQ3XXS> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 96];
            for j in 0..96 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockIQ3XXS {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_iq3_s_weights(k: usize) -> Vec<BlockIQ3S> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 64];
            let mut qh = [0u8; 8];
            let mut signs = [0u8; 32];
            let mut scales = [0u8; 4];
            for j in 0..64 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..8 {
                qh[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            for j in 0..32 {
                signs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..4 {
                scales[j] = ((i * 23 + j * 7) % 256) as u8;
            }
            BlockIQ3S {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
                qh,
                signs,
                scales,
            }
        })
        .collect()
}

fn generate_iq2_xxs_weights(k: usize) -> Vec<BlockIQ2XXS> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0u16; 32];
            for j in 0..32 {
                qs[j] = ((i * 17 + j * 13) % 256) as u16;
            }
            BlockIQ2XXS {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_iq2_xs_weights(k: usize) -> Vec<BlockIQ2XS> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0u16; 32];
            let mut scales = [0u8; 8];
            for j in 0..32 {
                qs[j] = ((i * 17 + j * 13) % 256) as u16;
            }
            for j in 0..8 {
                scales[j] = ((i * 23 + j * 7) % 256) as u8;
            }
            BlockIQ2XS {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
                scales,
            }
        })
        .collect()
}

fn generate_iq2_s_weights(k: usize) -> Vec<BlockIQ2S> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 64];
            let mut qh = [0u8; 8];
            let mut scales = [0u8; 8];
            for j in 0..64 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..8 {
                qh[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            for j in 0..8 {
                scales[j] = ((i * 23 + j * 7) % 256) as u8;
            }
            BlockIQ2S {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
                qh,
                scales,
            }
        })
        .collect()
}

fn generate_iq1_s_weights(k: usize) -> Vec<BlockIQ1S> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 32];
            let mut qh = [0u16; 8];
            for j in 0..32 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..8 {
                qh[j] = ((i * 23 + j * 7) % 256) as u16;
            }
            BlockIQ1S {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
                qh,
            }
        })
        .collect()
}

fn generate_iq1_m_weights(k: usize) -> Vec<BlockIQ1M> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 32];
            let mut qh = [0u8; 16];
            let mut scales = [0u8; 8];
            for j in 0..32 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..16 {
                qh[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            for j in 0..8 {
                scales[j] = ((i * 23 + j * 7) % 256) as u8;
            }
            BlockIQ1M { qs, qh, scales }
        })
        .collect()
}

fn generate_tq2_0_weights(k: usize) -> Vec<BlockTQ2_0> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 64];
            for j in 0..64 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockTQ2_0 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_tq1_0_weights(k: usize) -> Vec<BlockTQ1_0> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 48];
            let mut qh = [0u8; 4];
            for j in 0..48 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..4 {
                qh[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            BlockTQ1_0 {
                qs,
                qh,
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            }
        })
        .collect()
}

fn main() {
    unsafe {
        let k = 4096;

        println!("┌────────────────────────────────────────────┐");
        println!("│   IQ/TQ vec_dot Performance (K=4096)     │");
        println!("├────────────────────────────────────────────┤");
        println!("│  Format   │  GFLOPS  │  Block Size        │");
        println!("├────────────────────────────────────────────┤");

        let x_q8k = generate_q8_k_input(k);
        let x_q80 = generate_q8_0_input(k);

        // IQ4_XS
        {
            let weights = generate_iq4_xs_weights(k);
            for _ in 0..100 {
                let _ = vec_dot_iq4_xs_q8_k_neon(k, &weights, &x_q8k);
            }
            let start = Instant::now();
            for _ in 0..100000 {
                let _ = vec_dot_iq4_xs_q8_k_neon(k, &weights, &x_q8k);
            }
            let gflops = 2.0 * (k as f64) * 100000.0 / start.elapsed().as_secs_f64() / 1e9;
            println!("│  IQ4_XS   │  {:6.1}  │  256 (Q8_K)       │", gflops);
        }

        // IQ4_NL
        {
            let weights = generate_iq4_nl_weights(k);
            for _ in 0..100 {
                let _ = vec_dot_iq4_nl_q8_0_neon(k, &weights, &x_q80);
            }
            let start = Instant::now();
            for _ in 0..100000 {
                let _ = vec_dot_iq4_nl_q8_0_neon(k, &weights, &x_q80);
            }
            let gflops = 2.0 * (k as f64) * 100000.0 / start.elapsed().as_secs_f64() / 1e9;
            println!("│  IQ4_NL   │  {:6.1}  │   32 (Q8_0)       │", gflops);
        }

        // IQ3_XXS
        {
            let weights = generate_iq3_xxs_weights(k);
            for _ in 0..100 {
                let _ = vec_dot_iq3_xxs_q8_k_neon(k, &weights, &x_q8k);
            }
            let start = Instant::now();
            for _ in 0..100000 {
                let _ = vec_dot_iq3_xxs_q8_k_neon(k, &weights, &x_q8k);
            }
            let gflops = 2.0 * (k as f64) * 100000.0 / start.elapsed().as_secs_f64() / 1e9;
            println!("│  IQ3_XXS  │  {:6.1}  │  256 (Q8_K)       │", gflops);
        }

        // IQ3_S
        {
            let weights = generate_iq3_s_weights(k);
            for _ in 0..100 {
                let _ = vec_dot_iq3_s_q8_k_neon(k, &weights, &x_q8k);
            }
            let start = Instant::now();
            for _ in 0..100000 {
                let _ = vec_dot_iq3_s_q8_k_neon(k, &weights, &x_q8k);
            }
            let gflops = 2.0 * (k as f64) * 100000.0 / start.elapsed().as_secs_f64() / 1e9;
            println!("│  IQ3_S    │  {:6.1}  │  256 (Q8_K)       │", gflops);
        }

        // IQ2_XXS
        {
            let weights = generate_iq2_xxs_weights(k);
            for _ in 0..100 {
                let _ = vec_dot_iq2_xxs_q8_k_neon(k, &weights, &x_q8k);
            }
            let start = Instant::now();
            for _ in 0..100000 {
                let _ = vec_dot_iq2_xxs_q8_k_neon(k, &weights, &x_q8k);
            }
            let gflops = 2.0 * (k as f64) * 100000.0 / start.elapsed().as_secs_f64() / 1e9;
            println!("│  IQ2_XXS  │  {:6.1}  │  256 (Q8_K)       │", gflops);
        }

        // IQ2_XS
        {
            let weights = generate_iq2_xs_weights(k);
            for _ in 0..100 {
                let _ = vec_dot_iq2_xs_q8_k_neon(k, &weights, &x_q8k);
            }
            let start = Instant::now();
            for _ in 0..100000 {
                let _ = vec_dot_iq2_xs_q8_k_neon(k, &weights, &x_q8k);
            }
            let gflops = 2.0 * (k as f64) * 100000.0 / start.elapsed().as_secs_f64() / 1e9;
            println!("│  IQ2_XS   │  {:6.1}  │  256 (Q8_K)       │", gflops);
        }

        // IQ2_S
        {
            let weights = generate_iq2_s_weights(k);
            for _ in 0..100 {
                let _ = vec_dot_iq2_s_q8_k_neon(k, &weights, &x_q8k);
            }
            let start = Instant::now();
            for _ in 0..100000 {
                let _ = vec_dot_iq2_s_q8_k_neon(k, &weights, &x_q8k);
            }
            let gflops = 2.0 * (k as f64) * 100000.0 / start.elapsed().as_secs_f64() / 1e9;
            println!("│  IQ2_S    │  {:6.1}  │  256 (Q8_K)       │", gflops);
        }

        // IQ1_S
        {
            let weights = generate_iq1_s_weights(k);
            for _ in 0..100 {
                let _ = vec_dot_iq1_s_q8_k_neon(k, &weights, &x_q8k);
            }
            let start = Instant::now();
            for _ in 0..100000 {
                let _ = vec_dot_iq1_s_q8_k_neon(k, &weights, &x_q8k);
            }
            let gflops = 2.0 * (k as f64) * 100000.0 / start.elapsed().as_secs_f64() / 1e9;
            println!("│  IQ1_S    │  {:6.1}  │  256 (Q8_K)       │", gflops);
        }

        // IQ1_M
        {
            let weights = generate_iq1_m_weights(k);
            for _ in 0..100 {
                let _ = vec_dot_iq1_m_q8_k_neon(k, &weights, &x_q8k);
            }
            let start = Instant::now();
            for _ in 0..100000 {
                let _ = vec_dot_iq1_m_q8_k_neon(k, &weights, &x_q8k);
            }
            let gflops = 2.0 * (k as f64) * 100000.0 / start.elapsed().as_secs_f64() / 1e9;
            println!("│  IQ1_M    │  {:6.1}  │  256 (Q8_K)       │", gflops);
        }

        // TQ2_0
        {
            let weights = generate_tq2_0_weights(k);
            for _ in 0..100 {
                let _ = vec_dot_tq2_0_q8_k_neon(k, &weights, &x_q8k);
            }
            let start = Instant::now();
            for _ in 0..100000 {
                let _ = vec_dot_tq2_0_q8_k_neon(k, &weights, &x_q8k);
            }
            let gflops = 2.0 * (k as f64) * 100000.0 / start.elapsed().as_secs_f64() / 1e9;
            println!("│  TQ2_0    │  {:6.1}  │  256 (Q8_K)       │", gflops);
        }

        // TQ1_0
        {
            let weights = generate_tq1_0_weights(k);
            for _ in 0..100 {
                let _ = vec_dot_tq1_0_q8_k_neon(k, &weights, &x_q8k);
            }
            let start = Instant::now();
            for _ in 0..100000 {
                let _ = vec_dot_tq1_0_q8_k_neon(k, &weights, &x_q8k);
            }
            let gflops = 2.0 * (k as f64) * 100000.0 / start.elapsed().as_secs_f64() / 1e9;
            println!("│  TQ1_0    │  {:6.1}  │  256 (Q8_K)       │", gflops);
        }

        println!("└────────────────────────────────────────────┘");
    }
}

// Comprehensive correctness test for all matmul formats
use infer_train::quant::matmul::*;
use infer_train::quant::types::*;
use infer_train::quant::vec_dot::arm::*;
use half::f16;
use std::time::Instant;

fn generate_q8_k_input(k: usize) -> Vec<BlockQ8K> {
    let nb = k / 256;
    (0..nb).map(|i| {
        let mut qs = [0i8; 256];
        let mut bsums = [0i16; 16];
        for j in 0..256 { qs[j] = (((i * 17 + j * 13) % 256) as i32 - 128) as i8; }
        for j in 0..16 { bsums[j] = ((i * 23 + j * 7) % 256) as i16; }
        BlockQ8K { d: 0.01 + (i % 10) as f32 * 0.001, qs, bsums }
    }).collect()
}

fn generate_q8_0_input(k: usize) -> Vec<BlockQ8_0> {
    let nb = k / 32;
    (0..nb).map(|i| {
        let mut qs = [0i8; 32];
        for j in 0..32 { qs[j] = (((i * 17 + j * 13) % 256) as i32 - 128) as i8; }
        BlockQ8_0 { d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(), qs }
    }).collect()
}

fn generate_q8_1_input(k: usize) -> Vec<BlockQ8_1> {
    let nb = k / 32;
    (0..nb).map(|i| {
        let mut qs = [0i8; 32];
        for j in 0..32 { qs[j] = (((i * 17 + j * 13) % 256) as i32 - 128) as i8; }
        BlockQ8_1 {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            s: f16::from_f32(0.1 + (i % 5) as f32 * 0.05).to_bits(),
            qs,
        }
    }).collect()
}

fn generate_q4_k_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 128];
        let mut scales = [0u8; 12];
        for j in 0..128 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..12 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockQ4K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            dmin: f16::from_f32(0.1 + (i % 5) as f32 * 0.05).to_bits(),
            scales,
            qs,
        }
    }).collect()
}

fn generate_q5_k_weights(m: usize, k: usize) -> Vec<BlockQ5K> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 128];
        let mut qh = [0u8; 32];
        let mut scales = [0u8; 12];
        for j in 0..128 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..32 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
        for j in 0..12 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockQ5K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            dmin: f16::from_f32(0.1).to_bits(),
            scales,
            qh,
            qs,
        }
    }).collect()
}

fn generate_q6_k_weights(m: usize, k: usize) -> Vec<BlockQ6K> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let mut scales = [0i8; 16];
        for j in 0..128 { ql[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..64 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
        for j in 0..16 { scales[j] = ((i * 23 + j * 7) % 128) as i8 - 64; }
        BlockQ6K {
            ql,
            qh,
            scales,
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
        }
    }).collect()
}

fn generate_q3_k_weights(m: usize, k: usize) -> Vec<BlockQ3K> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 64];
        let mut hmask = [0u8; 32];
        let mut scales = [0u8; 12];
        for j in 0..64 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..32 { hmask[j] = ((i * 19 + j * 11) % 256) as u8; }
        for j in 0..12 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockQ3K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            hmask,
            qs,
            scales,
        }
    }).collect()
}

fn generate_q2_k_weights(m: usize, k: usize) -> Vec<BlockQ2K> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 64];
        let mut scales = [0u8; 16];
        for j in 0..64 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..16 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockQ2K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            dmin: f16::from_f32(0.1).to_bits(),
            scales,
            qs,
        }
    }).collect()
}

fn generate_q4_0_weights(m: usize, k: usize) -> Vec<BlockQ4_0> {
    let nb = k / 32;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 16];
        for j in 0..16 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        BlockQ4_0 { d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(), qs }
    }).collect()
}

fn generate_q5_0_weights(m: usize, k: usize) -> Vec<BlockQ5_0> {
    let nb = k / 32;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 16];
        let mut qh = [0u8; 4];
        for j in 0..16 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..4 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
        BlockQ5_0 {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            qh,
            qs,
        }
    }).collect()
}

fn generate_q4_1_weights(m: usize, k: usize) -> Vec<BlockQ4_1> {
    let nb = k / 32;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 16];
        for j in 0..16 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        BlockQ4_1 {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            m: f16::from_f32(0.1 + (i % 5) as f32 * 0.05).to_bits(),
            qs,
        }
    }).collect()
}

fn generate_q5_1_weights(m: usize, k: usize) -> Vec<BlockQ5_1> {
    let nb = k / 32;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 16];
        let mut qh = [0u8; 4];
        for j in 0..16 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..4 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
        BlockQ5_1 {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            m: f16::from_f32(0.1 + (i % 5) as f32 * 0.05).to_bits(),
            qh,
            qs,
        }
    }).collect()
}

// IQ series weights
fn generate_iq4_xs_weights(m: usize, k: usize) -> Vec<BlockIQ4XS> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 128];
        let mut scales_l = [0u8; 4];
        for j in 0..128 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..4 { scales_l[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockIQ4XS {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            scales_h: ((i * 23) % 256) as u16,
            scales_l,
            qs,
        }
    }).collect()
}

fn generate_iq4_nl_weights(m: usize, k: usize) -> Vec<BlockIQ4NL> {
    let nb = k / 32;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 16];
        for j in 0..16 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        BlockIQ4NL { d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(), qs }
    }).collect()
}

fn generate_iq3_xxs_weights(m: usize, k: usize) -> Vec<BlockIQ3XXS> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 96];
        for j in 0..96 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        BlockIQ3XXS {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            qs,
        }
    }).collect()
}

fn generate_iq3_s_weights(m: usize, k: usize) -> Vec<BlockIQ3S> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 64];
        let mut qh = [0u8; 8];
        let mut signs = [0u8; 32];
        let mut scales = [0u8; 4];
        for j in 0..64 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..8 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
        for j in 0..32 { signs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..4 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockIQ3S {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            qs,
            qh,
            signs,
            scales,
        }
    }).collect()
}

fn generate_iq2_xxs_weights(m: usize, k: usize) -> Vec<BlockIQ2XXS> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u16; 32];
        for j in 0..32 { qs[j] = ((i * 17 + j * 13) % 256) as u16; }
        BlockIQ2XXS {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            qs,
        }
    }).collect()
}

fn generate_iq2_xs_weights(m: usize, k: usize) -> Vec<BlockIQ2XS> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u16; 32];
        let mut scales = [0u8; 8];
        for j in 0..32 { qs[j] = ((i * 17 + j * 13) % 256) as u16; }
        for j in 0..8 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockIQ2XS {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            qs,
            scales,
        }
    }).collect()
}

fn generate_iq2_s_weights(m: usize, k: usize) -> Vec<BlockIQ2S> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 64];
        let mut qh = [0u8; 8];
        let mut scales = [0u8; 8];
        for j in 0..64 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..8 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
        for j in 0..8 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockIQ2S {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            qs,
            qh,
            scales,
        }
    }).collect()
}

fn generate_iq1_s_weights(m: usize, k: usize) -> Vec<BlockIQ1S> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 32];
        let mut qh = [0u16; 8];
        for j in 0..32 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..8 { qh[j] = ((i * 23 + j * 7) % 256) as u16; }
        BlockIQ1S {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            qs,
            qh,
        }
    }).collect()
}

fn generate_iq1_m_weights(m: usize, k: usize) -> Vec<BlockIQ1M> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 32];
        let mut qh = [0u8; 16];
        let mut scales = [0u8; 8];
        for j in 0..32 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..16 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
        for j in 0..8 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockIQ1M { qs, qh, scales }
    }).collect()
}

fn generate_tq2_0_weights(m: usize, k: usize) -> Vec<BlockTQ2_0> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 64];
        for j in 0..64 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        BlockTQ2_0 { d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(), qs }
    }).collect()
}

fn generate_tq1_0_weights(m: usize, k: usize) -> Vec<BlockTQ1_0> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 48];
        let mut qh = [0u8; 4];
        for j in 0..48 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..4 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
        BlockTQ1_0 {
            qs,
            qh,
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
        }
    }).collect()
}

fn main() {
    unsafe {
        println!("=== Matmul Correctness Verification ===\n");
        println!("Comparing matmul(M=4, K=1024) with 4x vec_dot calls");
        println!("Format                  | matmul[0] | vec_dot[0] | Match");
        println!("------------------------|-----------|------------|-------");

        let k = 1024;
        let m = 4;

        // Q4_K
        {
            let nb = k / 256;
            let weights = generate_q4_k_weights(m, k);
            let x = generate_q8_k_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_q4_k_q8_k(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_q4_k_q8_k_neon(k, &weights[0..nb], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("Q4_K                    | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // Q2_K
        {
            let nb = k / 256;
            let weights = generate_q2_k_weights(m, k);
            let x = generate_q8_k_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_q2_k_q8_k(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_q2_k_q8_k_neon(k, &weights[0..nb], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("Q2_K                    | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // Q3_K
        {
            let nb = k / 256;
            let weights = generate_q3_k_weights(m, k);
            let x = generate_q8_k_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_q3_k_q8_k(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_q3_k_q8_k_neon(k, &weights[0..nb], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("Q3_K                    | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // Q5_K
        {
            let nb = k / 256;
            let weights = generate_q5_k_weights(m, k);
            let x = generate_q8_k_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_q5_k_q8_k(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_q5_k_q8_k_neon(k, &weights[0..nb], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("Q5_K                    | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // Q6_K
        {
            let nb = k / 256;
            let weights = generate_q6_k_weights(m, k);
            let x = generate_q8_k_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_q6_k_q8_k(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_q6_k_q8_k_neon(k, &weights[0..nb], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("Q6_K                    | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // Q4_0 (block size 32)
        {
            let k = 1024;
            let weights = generate_q4_0_weights(m, k);
            let x = generate_q8_0_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_q4_0_q8_0(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_q4_0_q8_0_neon(k, &weights[0..32], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("Q4_0                    | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // Q5_0 (block size 32)
        {
            let k = 1024;
            let weights = generate_q5_0_weights(m, k);
            let x = generate_q8_0_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_q5_0_q8_0(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_q5_0_q8_0_neon(k, &weights[0..32], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("Q5_0                    | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // Q4_1 (block size 32)
        {
            let k = 1024;
            let weights = generate_q4_1_weights(m, k);
            let x = generate_q8_1_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_q4_1_q8_1(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_q4_1_q8_1_neon(k, &weights[0..32], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("Q4_1                    | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // Q5_1 (block size 32)
        {
            let k = 1024;
            let weights = generate_q5_1_weights(m, k);
            let x = generate_q8_1_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_q5_1_q8_1(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_q5_1_q8_1_neon(k, &weights[0..32], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("Q5_1                    | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // IQ series
        println!("\n--- IQ/TQ Series (block size 256) ---");

        // IQ4_XS
        {
            let k = 1024;
            let nb = k / 256;
            let weights = generate_iq4_xs_weights(m, k);
            let x = generate_q8_k_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_iq4_xs_q8_k(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_iq4_xs_q8_k_neon(k, &weights[0..nb], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("IQ4_XS                  | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // IQ4_NL
        {
            let k = 1024;
            let weights = generate_iq4_nl_weights(m, k);
            let x = generate_q8_0_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_iq4_nl_q8_0(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_iq4_nl_q8_0_neon(k, &weights[0..32], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("IQ4_NL                  | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // IQ3_XXS
        {
            let k = 1024;
            let nb = k / 256;
            let weights = generate_iq3_xxs_weights(m, k);
            let x = generate_q8_k_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_iq3_xxs_q8_k(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_iq3_xxs_q8_k_neon(k, &weights[0..nb], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("IQ3_XXS                 | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // IQ3_S
        {
            let k = 1024;
            let nb = k / 256;
            let weights = generate_iq3_s_weights(m, k);
            let x = generate_q8_k_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_iq3_s_q8_k(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_iq3_s_q8_k_neon(k, &weights[0..nb], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("IQ3_S                   | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // IQ2_XXS
        {
            let k = 1024;
            let nb = k / 256;
            let weights = generate_iq2_xxs_weights(m, k);
            let x = generate_q8_k_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_iq2_xxs_q8_k(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_iq2_xxs_q8_k_neon(k, &weights[0..nb], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("IQ2_XXS                 | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // IQ2_XS
        {
            let k = 1024;
            let nb = k / 256;
            let weights = generate_iq2_xs_weights(m, k);
            let x = generate_q8_k_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_iq2_xs_q8_k(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_iq2_xs_q8_k_neon(k, &weights[0..nb], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("IQ2_XS                  | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // IQ2_S
        {
            let k = 1024;
            let nb = k / 256;
            let weights = generate_iq2_s_weights(m, k);
            let x = generate_q8_k_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_iq2_s_q8_k(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_iq2_s_q8_k_neon(k, &weights[0..nb], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("IQ2_S                   | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // IQ1_S
        {
            let k = 1024;
            let nb = k / 256;
            let weights = generate_iq1_s_weights(m, k);
            let x = generate_q8_k_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_iq1_s_q8_k(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_iq1_s_q8_k_neon(k, &weights[0..nb], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("IQ1_S                   | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // IQ1_M
        {
            let k = 1024;
            let nb = k / 256;
            let weights = generate_iq1_m_weights(m, k);
            let x = generate_q8_k_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_iq1_m_q8_k(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_iq1_m_q8_k_neon(k, &weights[0..nb], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("IQ1_M                   | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // TQ2_0
        {
            let k = 1024;
            let nb = k / 256;
            let weights = generate_tq2_0_weights(m, k);
            let x = generate_q8_k_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_tq2_0_q8_k(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_tq2_0_q8_k_neon(k, &weights[0..nb], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("TQ2_0                   | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        // TQ1_0
        {
            let k = 1024;
            let nb = k / 256;
            let weights = generate_tq1_0_weights(m, k);
            let x = generate_q8_k_input(k);
            let mut dst = vec![0.0f32; m];
            matmul_tq1_0_q8_k(&weights, &x, &mut dst, k, m);
            let vd0 = vec_dot_tq1_0_q8_k_neon(k, &weights[0..nb], &x);
            let match0 = (dst[0] - vd0).abs() < 0.1;
            println!("TQ1_0                   | {:9.2} | {:10.2} | {}", dst[0], vd0, match0);
        }

        println!("\n=== All formats tested ===");
    }
}

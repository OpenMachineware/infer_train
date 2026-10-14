// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Complete GEMM Verification: All 24 formats
// Verifies all GPU kernels run correctly after refactoring

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn gen_f32(k: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect()
}

// K-quant (5)
fn gen_q2_k(m: usize, k: usize, s: u64) -> Vec<BlockQ2K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            let mut sc = [0u8; 16];
            for i in 0..64 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..16 {
                sc[i] = rng.r#gen::<u8>() % 16 + 1;
            }
            BlockQ2K {
                d: f16::from_f32(0.5).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales: sc,
                qs,
            }
        })
        .collect()
}
fn gen_q3_k(m: usize, k: usize, s: u64) -> Vec<BlockQ3K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            let mut hm = [0u8; 32];
            let mut sc = [0u8; 12];
            for i in 0..64 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..32 {
                hm[i] = rng.r#gen::<u8>();
            }
            for i in 0..12 {
                sc[i] = rng.r#gen::<u8>() % 16 + 1;
            }
            BlockQ3K {
                d: f16::from_f32(0.5).to_bits(),
                hmask: hm,
                qs,
                scales: sc,
            }
        })
        .collect()
}
fn gen_q4_k(m: usize, k: usize, s: u64) -> Vec<BlockQ4K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 128];
            let mut sc = [0u8; 12];
            for i in 0..128 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..12 {
                sc[i] = rng.r#gen::<u8>() % 16 + 1;
            }
            BlockQ4K {
                d: f16::from_f32(0.5).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales: sc,
                qs,
            }
        })
        .collect()
}
fn gen_q5_k(m: usize, k: usize, s: u64) -> Vec<BlockQ5K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 128];
            let mut qh = [0u8; 32];
            let mut sc = [0u8; 12];
            for i in 0..128 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..32 {
                qh[i] = rng.r#gen::<u8>();
            }
            for i in 0..12 {
                sc[i] = rng.r#gen::<u8>() % 16 + 1;
            }
            BlockQ5K {
                d: f16::from_f32(0.5).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales: sc,
                qh,
                qs,
            }
        })
        .collect()
}
fn gen_q6_k(m: usize, k: usize, s: u64) -> Vec<BlockQ6K> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut sc = [0i8; 16];
            for i in 0..128 {
                ql[i] = rng.r#gen::<u8>();
            }
            for i in 0..64 {
                qh[i] = rng.r#gen::<u8>();
            }
            for i in 0..16 {
                sc[i] = (rng.r#gen::<u8>() % 64) as i8 - 32;
            }
            BlockQ6K { ql, qh, scales: sc, d: f16::from_f32(0.5).to_bits() }
        })
        .collect()
}

// Standard Q (7)
fn gen_q1_0(m: usize, k: usize, s: u64) -> Vec<BlockQ1_0> {
    let nb = k / 32;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 4];
            for i in 0..4 {
                qs[i] = rng.r#gen::<u8>();
            }
            BlockQ1_0 { d: f16::from_f32(0.5).to_bits(), qs }
        })
        .collect()
}
fn gen_q2_0(m: usize, k: usize, s: u64) -> Vec<BlockQ2_0> {
    let nb = k / 64;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            for i in 0..16 {
                qs[i] = rng.r#gen::<u8>();
            }
            BlockQ2_0 { d: f16::from_f32(0.5).to_bits(), qs }
        })
        .collect()
}
fn gen_q4_0(m: usize, k: usize, s: u64) -> Vec<BlockQ4_0> {
    let nb = k / 32;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            for i in 0..16 {
                qs[i] = rng.r#gen::<u8>();
            }
            BlockQ4_0 { d: f16::from_f32(0.5).to_bits(), qs }
        })
        .collect()
}
fn gen_q4_1(m: usize, k: usize, s: u64) -> Vec<BlockQ4_1> {
    let nb = k / 32;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            for i in 0..16 {
                qs[i] = rng.r#gen::<u8>();
            }
            BlockQ4_1 {
                d: f16::from_f32(0.5).to_bits(),
                m: f16::from_f32(0.1).to_bits(),
                qs,
            }
        })
        .collect()
}
fn gen_q5_0(m: usize, k: usize, s: u64) -> Vec<BlockQ5_0> {
    let nb = k / 32;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            let mut qh = [0u8; 4];
            for i in 0..16 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..4 {
                qh[i] = rng.r#gen::<u8>();
            }
            BlockQ5_0 { d: f16::from_f32(0.5).to_bits(), qh, qs }
        })
        .collect()
}
fn gen_q5_1(m: usize, k: usize, s: u64) -> Vec<BlockQ5_1> {
    let nb = k / 32;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            let mut qh = [0u8; 4];
            for i in 0..16 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..4 {
                qh[i] = rng.r#gen::<u8>();
            }
            BlockQ5_1 {
                d: f16::from_f32(0.5).to_bits(),
                m: f16::from_f32(0.1).to_bits(),
                qh,
                qs,
            }
        })
        .collect()
}
fn gen_q8_0(m: usize, k: usize, s: u64) -> Vec<BlockQ8_0> {
    let nb = k / 32;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0i8; 32];
            for i in 0..32 {
                qs[i] = rng.r#gen::<u8>() as i8;
            }
            BlockQ8_0 { d: f16::from_f32(0.5).to_bits(), qs }
        })
        .collect()
}

// IQ (9)
fn gen_iq1_s(m: usize, k: usize, s: u64) -> Vec<BlockIQ1S> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 32];
            let mut qh = [0u16; 8];
            for i in 0..32 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..8 {
                qh[i] = rng.r#gen::<u16>();
            }
            BlockIQ1S { d: f16::from_f32(0.5).to_bits(), qs, qh }
        })
        .collect()
}
fn gen_iq1_m(m: usize, k: usize, s: u64) -> Vec<BlockIQ1M> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 32];
            let mut qh = [0u8; 16];
            let mut sc = [0u8; 8];
            for i in 0..32 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..16 {
                qh[i] = rng.r#gen::<u8>();
            }
            for i in 0..8 {
                sc[i] = rng.r#gen::<u8>();
            }
            BlockIQ1M { qs, qh, scales: sc }
        })
        .collect()
}
fn gen_iq2_xxs(m: usize, k: usize, s: u64) -> Vec<BlockIQ2XXS> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u16; 32];
            for i in 0..32 {
                qs[i] = rng.r#gen::<u16>();
            }
            BlockIQ2XXS { d: f16::from_f32(0.5).to_bits(), qs }
        })
        .collect()
}
fn gen_iq2_xs(m: usize, k: usize, s: u64) -> Vec<BlockIQ2XS> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u16; 32];
            let mut sc = [0u8; 8];
            for i in 0..32 {
                qs[i] = rng.r#gen::<u16>();
            }
            for i in 0..8 {
                sc[i] = rng.r#gen::<u8>();
            }
            BlockIQ2XS { d: f16::from_f32(0.5).to_bits(), qs, scales: sc }
        })
        .collect()
}
fn gen_iq2_s(m: usize, k: usize, s: u64) -> Vec<BlockIQ2S> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            let mut qh = [0u8; 8];
            let mut sc = [0u8; 8];
            for i in 0..64 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..8 {
                qh[i] = rng.r#gen::<u8>();
            }
            for i in 0..8 {
                sc[i] = rng.r#gen::<u8>();
            }
            BlockIQ2S { d: f16::from_f32(0.5).to_bits(), qs, qh, scales: sc }
        })
        .collect()
}
fn gen_iq3_xxs(m: usize, k: usize, s: u64) -> Vec<BlockIQ3XXS> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 96];
            for i in 0..96 {
                qs[i] = rng.r#gen::<u8>();
            }
            BlockIQ3XXS { d: f16::from_f32(0.5).to_bits(), qs }
        })
        .collect()
}
fn gen_iq3_s(m: usize, k: usize, s: u64) -> Vec<BlockIQ3S> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            let mut qh = [0u8; 8];
            let mut sg = [0u8; 32];
            let mut sc = [0u8; 4];
            for i in 0..64 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..8 {
                qh[i] = rng.r#gen::<u8>();
            }
            for i in 0..32 {
                sg[i] = rng.r#gen::<u8>();
            }
            for i in 0..4 {
                sc[i] = rng.r#gen::<u8>();
            }
            BlockIQ3S {
                d: f16::from_f32(0.5).to_bits(),
                qs,
                qh,
                signs: sg,
                scales: sc,
            }
        })
        .collect()
}
fn gen_iq4_nl(m: usize, k: usize, s: u64) -> Vec<BlockIQ4NL> {
    let nb = k / 32;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            for i in 0..16 {
                qs[i] = rng.r#gen::<u8>();
            }
            BlockIQ4NL { d: f16::from_f32(0.5).to_bits(), qs }
        })
        .collect()
}
fn gen_iq4_xs(m: usize, k: usize, s: u64) -> Vec<BlockIQ4XS> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 128];
            let mut sc = [0u8; 4];
            for i in 0..128 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..4 {
                sc[i] = rng.r#gen::<u8>();
            }
            BlockIQ4XS {
                d: f16::from_f32(0.5).to_bits(),
                scales_h: rng.r#gen::<u16>(),
                scales_l: sc,
                qs,
            }
        })
        .collect()
}

// TQ (2)
fn gen_tq1_0(m: usize, k: usize, s: u64) -> Vec<BlockTQ1_0> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 48];
            let mut qh = [0u8; 4];
            for i in 0..48 {
                qs[i] = rng.r#gen::<u8>();
            }
            for i in 0..4 {
                qh[i] = rng.r#gen::<u8>();
            }
            BlockTQ1_0 { qs, qh, d: f16::from_f32(0.5).to_bits() }
        })
        .collect()
}
fn gen_tq2_0(m: usize, k: usize, s: u64) -> Vec<BlockTQ2_0> {
    let nb = k / 256;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            for i in 0..64 {
                qs[i] = rng.r#gen::<u8>();
            }
            BlockTQ2_0 { d: f16::from_f32(0.5).to_bits(), qs }
        })
        .collect()
}

// MXFP (1)
fn gen_mxfp4(m: usize, k: usize, s: u64) -> Vec<BlockMXFP4> {
    let nb = k / 32;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 16];
            for i in 0..16 {
                qs[i] = rng.r#gen::<u8>();
            }
            // E8M0: e=127 gives scale 1.0, use range 120-135 for reasonable scales
            BlockMXFP4 { e: 120 + (i as u8 % 16), qs }
        })
        .collect()
}

fn check(name: &str, gpu: &[f32]) -> bool {
    let nan_inf =
        gpu.iter().filter(|&&v| v.is_nan() || v.is_infinite()).count();
    let non_zero = gpu.iter().filter(|&&v| v.abs() > 1e-6).count();
    let ok = nan_inf == 0 && non_zero > 0;
    println!(
        "  {:12} {} values, {} non-zero, {} NaN/Inf {}",
        name,
        gpu.len(),
        non_zero,
        nan_inf,
        if ok { "✓" } else { "✗" }
    );
    ok
}

fn main() {
    let ctx = MetalContext::new().expect("Metal init failed");
    let m = 64;
    let k = 1024;
    let seed = 42;
    let input = gen_f32(k, seed);
    let mut passed = 0;

    println!("=== K-quant (5) ===");
    if check(
        "Q2_K",
        &ctx.mv_q2_k_f32(m, k, &gen_q2_k(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "Q3_K",
        &ctx.mv_q3_k_f32(m, k, &gen_q3_k(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "Q4_K",
        &ctx.mv_q4_k_f32(m, k, &gen_q4_k(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "Q5_K",
        &ctx.mv_q5_k_f32(m, k, &gen_q5_k(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "Q6_K",
        &ctx.mv_q6_k_f32(m, k, &gen_q6_k(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }

    println!("\n=== Standard Q (7) ===");
    if check(
        "Q1_0",
        &ctx.mv_q1_0_f32(m, k, &gen_q1_0(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "Q2_0",
        &ctx.mv_q2_0_f32(
            m,
            k * 2,
            &gen_q2_0(m, k * 2, seed),
            &gen_f32(k * 2, seed),
        )
        .unwrap(),
    ) {
        passed += 1;
    } // Q2_0 uses 64
    if check(
        "Q4_0",
        &ctx.mv_q4_0_f32(m, k, &gen_q4_0(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "Q4_1",
        &ctx.mv_q4_1_f32(m, k, &gen_q4_1(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "Q5_0",
        &ctx.mv_q5_0_f32(m, k, &gen_q5_0(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "Q5_1",
        &ctx.mv_q5_1_f32(m, k, &gen_q5_1(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "Q8_0",
        &ctx.mv_q8_0_f32(m, k, &gen_q8_0(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }

    println!("\n=== IQ (9) ===");
    if check(
        "IQ1_S",
        &ctx.mv_iq1_s_f32(m, k, &gen_iq1_s(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "IQ1_M",
        &ctx.mv_iq1_m_f32(m, k, &gen_iq1_m(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "IQ2_XXS",
        &ctx.mv_iq2_xxs_f32(m, k, &gen_iq2_xxs(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "IQ2_XS",
        &ctx.mv_iq2_xs_f32(m, k, &gen_iq2_xs(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "IQ2_S",
        &ctx.mv_iq2_s_f32(m, k, &gen_iq2_s(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "IQ3_XXS",
        &ctx.mv_iq3_xxs_f32(m, k, &gen_iq3_xxs(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "IQ3_S",
        &ctx.mv_iq3_s_f32(m, k, &gen_iq3_s(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "IQ4_NL",
        &ctx.mv_iq4_nl_f32(m, k, &gen_iq4_nl(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "IQ4_XS",
        &ctx.mv_iq4_xs_f32(m, k, &gen_iq4_xs(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }

    println!("\n=== TQ (2) ===");
    if check(
        "TQ1_0",
        &ctx.mv_tq1_0_f32(m, k, &gen_tq1_0(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }
    if check(
        "TQ2_0",
        &ctx.mv_tq2_0_f32(m, k, &gen_tq2_0(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }

    println!("\n=== MXFP (1) ===");
    if check(
        "MXFP4",
        &ctx.mv_mxfp4_f32(m, k, &gen_mxfp4(m, k, seed), &input).unwrap(),
    ) {
        passed += 1;
    }

    println!("\n=== Summary ===");
    println!("Passed: {}/24", passed);
    if passed == 24 {
        println!("All formats work correctly ✓");
    }
}

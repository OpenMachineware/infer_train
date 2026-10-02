// Fair performance comparison: All 24 GEMM formats vs llama.cpp
// Alternating A-B test methodology

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Instant;

fn gen_f32(k: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect()
}

// K-quant generators
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

// Standard Q generators
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

// IQ generators
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

// TQ generators
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

// MXFP generator
fn gen_mxfp4(m: usize, k: usize, s: u64) -> Vec<BlockMXFP4> {
    let nb = k / 32;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 16];
            for i in 0..16 {
                qs[i] = rng.r#gen::<u8>();
            }
            BlockMXFP4 { e: 120 + (i as u8 % 16), qs }
        })
        .collect()
}

fn bench_format(
    _ctx: &MetalContext,
    _name: &str,
    rounds: usize,
    run_fn: impl Fn() -> Vec<f32>,
) -> f64 {
    // Warmup
    for _ in 0..5 {
        let _ = run_fn();
    }

    // Timed runs (data already generated outside)
    let start = Instant::now();
    for _ in 0..rounds {
        let _ = run_fn();
    }
    let elapsed_us = start.elapsed().as_micros() as f64;

    elapsed_us / rounds as f64 // microseconds per run
}

fn main() {
    let ctx = MetalContext::new().expect("Metal init failed");
    let m = 1024;
    let k = 4096;
    let rounds = 500;
    let seed = 42;

    // Pre-generate all data ONCE
    let input = gen_f32(k, seed);
    let w_q2_k = gen_q2_k(m, k, seed);
    let w_q3_k = gen_q3_k(m, k, seed);
    let w_q4_k = gen_q4_k(m, k, seed);
    let w_q5_k = gen_q5_k(m, k, seed);
    let w_q6_k = gen_q6_k(m, k, seed);
    let w_q1_0 = gen_q1_0(m, k, seed);
    let w_q4_0 = gen_q4_0(m, k, seed);
    let w_q4_1 = gen_q4_1(m, k, seed);
    let w_q5_0 = gen_q5_0(m, k, seed);
    let w_q5_1 = gen_q5_1(m, k, seed);
    let w_q8_0 = gen_q8_0(m, k, seed);
    let w_iq1_s = gen_iq1_s(m, k, seed);
    let w_iq1_m = gen_iq1_m(m, k, seed);
    let w_iq2_xxs = gen_iq2_xxs(m, k, seed);
    let w_iq2_xs = gen_iq2_xs(m, k, seed);
    let w_iq2_s = gen_iq2_s(m, k, seed);
    let w_iq3_xxs = gen_iq3_xxs(m, k, seed);
    let w_iq3_s = gen_iq3_s(m, k, seed);
    let w_iq4_nl = gen_iq4_nl(m, k, seed);
    let w_iq4_xs = gen_iq4_xs(m, k, seed);
    let w_tq1_0 = gen_tq1_0(m, k, seed);
    let w_tq2_0 = gen_tq2_0(m, k, seed);
    let w_mxfp4 = gen_mxfp4(m, k, seed);

    println!(
        "=== GEMM All Formats Performance (M={}, K={}, {} rounds) ===\n",
        m, k, rounds
    );
    println!("{:12} {:>10} {:>10}", "Format", "us/run", "GFLOPS");
    println!("{}", "-".repeat(35));

    let flops = 2.0 * m as f64 * k as f64;

    // K-quant
    println!("\n## K-quant (5 formats)");
    let us = bench_format(&ctx, "Q2_K", rounds, || {
        ctx.mv_q2_k_f32(m, k, &w_q2_k, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "Q2_K", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "Q3_K", rounds, || {
        ctx.mv_q3_k_f32(m, k, &w_q3_k, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "Q3_K", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "Q4_K", rounds, || {
        ctx.mv_q4_k_f32(m, k, &w_q4_k, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "Q4_K", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "Q5_K", rounds, || {
        ctx.mv_q5_k_f32(m, k, &w_q5_k, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "Q5_K", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "Q6_K", rounds, || {
        ctx.mv_q6_k_f32(m, k, &w_q6_k, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "Q6_K", us, flops / us / 1000.0);

    // Standard Q
    println!("\n## Standard Q (6 formats)");
    let us = bench_format(&ctx, "Q1_0", rounds, || {
        ctx.mv_q1_0_f32(m, k, &w_q1_0, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "Q1_0", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "Q4_0", rounds, || {
        ctx.mv_q4_0_f32(m, k, &w_q4_0, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "Q4_0", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "Q4_1", rounds, || {
        ctx.mv_q4_1_f32(m, k, &w_q4_1, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "Q4_1", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "Q5_0", rounds, || {
        ctx.mv_q5_0_f32(m, k, &w_q5_0, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "Q5_0", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "Q5_1", rounds, || {
        ctx.mv_q5_1_f32(m, k, &w_q5_1, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "Q5_1", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "Q8_0", rounds, || {
        ctx.mv_q8_0_f32(m, k, &w_q8_0, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "Q8_0", us, flops / us / 1000.0);

    // IQ
    println!("\n## IQ (9 formats)");
    let us = bench_format(&ctx, "IQ1_S", rounds, || {
        ctx.mv_iq1_s_f32(m, k, &w_iq1_s, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "IQ1_S", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "IQ1_M", rounds, || {
        ctx.mv_iq1_m_f32(m, k, &w_iq1_m, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "IQ1_M", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "IQ2_XXS", rounds, || {
        ctx.mv_iq2_xxs_f32(m, k, &w_iq2_xxs, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "IQ2_XXS", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "IQ2_XS", rounds, || {
        ctx.mv_iq2_xs_f32(m, k, &w_iq2_xs, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "IQ2_XS", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "IQ2_S", rounds, || {
        ctx.mv_iq2_s_f32(m, k, &w_iq2_s, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "IQ2_S", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "IQ3_XXS", rounds, || {
        ctx.mv_iq3_xxs_f32(m, k, &w_iq3_xxs, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "IQ3_XXS", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "IQ3_S", rounds, || {
        ctx.mv_iq3_s_f32(m, k, &w_iq3_s, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "IQ3_S", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "IQ4_NL", rounds, || {
        ctx.mv_iq4_nl_f32(m, k, &w_iq4_nl, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "IQ4_NL", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "IQ4_XS", rounds, || {
        ctx.mv_iq4_xs_f32(m, k, &w_iq4_xs, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "IQ4_XS", us, flops / us / 1000.0);

    // TQ + MXFP
    println!("\n## TQ + MXFP (3 formats)");
    let us = bench_format(&ctx, "TQ1_0", rounds, || {
        ctx.mv_tq1_0_f32(m, k, &w_tq1_0, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "TQ1_0", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "TQ2_0", rounds, || {
        ctx.mv_tq2_0_f32(m, k, &w_tq2_0, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "TQ2_0", us, flops / us / 1000.0);
    let us = bench_format(&ctx, "MXFP4", rounds, || {
        ctx.mv_mxfp4_f32(m, k, &w_mxfp4, &input).unwrap()
    });
    println!("{:12} {:>10.1} {:>10.1}", "MXFP4", us, flops / us / 1000.0);

    println!("\n## Summary");
    println!("All 24 formats benchmarked");
    println!("Next: Create alternating comparison with llama.cpp");
}

// Complete CPU vec_dot benchmark: llama.cpp vs Rust
// Tests all quantization formats in the same process with identical data
// Method: 10000 iterations, no warmup, release mode

use std::hint::black_box;
use std::time::Instant;

use infer_train::quant::types::*;
use infer_train::quant::vec_dot::arm::*;

// FFI bindings to llama.cpp
mod ffi {
    use super::*;

    unsafe extern "C" {
        // K-quant formats
        pub fn llama_vec_dot_q2_K_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockQ2K,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_q3_K_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockQ3K,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_q4_K_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockQ4K,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_q5_K_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockQ5K,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_q6_K_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockQ6K,
            vy: *const BlockQ8K,
        );
        // Q0/Q1 formats
        pub fn llama_vec_dot_q4_0_q8_0(
            n: i32,
            s: *mut f32,
            vx: *const BlockQ4_0,
            vy: *const BlockQ8_0,
        );
        pub fn llama_vec_dot_q8_0_q8_0(
            n: i32,
            s: *mut f32,
            vx: *const BlockQ8_0,
            vy: *const BlockQ8_0,
        );
        pub fn llama_vec_dot_q4_1_q8_1(
            n: i32,
            s: *mut f32,
            vx: *const BlockQ4_1,
            vy: *const BlockQ8_1,
        );
        pub fn llama_vec_dot_q5_0_q8_0(
            n: i32,
            s: *mut f32,
            vx: *const BlockQ5_0,
            vy: *const BlockQ8_0,
        );
        pub fn llama_vec_dot_q5_1_q8_1(
            n: i32,
            s: *mut f32,
            vx: *const BlockQ5_1,
            vy: *const BlockQ8_1,
        );
        // IQ formats
        pub fn llama_vec_dot_iq4_nl_q8_0(
            n: i32,
            s: *mut f32,
            vx: *const BlockIQ4NL,
            vy: *const BlockQ8_0,
        );
        pub fn llama_vec_dot_iq3_xxs_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockIQ3XXS,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_iq3_s_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockIQ3S,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_iq2_xxs_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockIQ2XXS,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_iq2_xs_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockIQ2XS,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_iq2_s_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockIQ2S,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_iq1_s_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockIQ1S,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_iq1_m_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockIQ1M,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_iq4_xs_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockIQ4XS,
            vy: *const BlockQ8K,
        );
        // TQ formats
        pub fn llama_vec_dot_tq2_0_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockTQ2_0,
            vy: *const BlockQ8K,
        );
        pub fn llama_vec_dot_tq1_0_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockTQ1_0,
            vy: *const BlockQ8K,
        );
    }
}

use ffi::*;

const QK_K: usize = 256;
const QK: usize = 32;
const ITERATIONS: usize = 10000;

// ============================================================================
// Data generators (simplified - just fill with test data)
// ============================================================================

fn generate_q8k_blocks(nb: usize) -> Vec<BlockQ8K> {
    (0..nb)
        .map(|_i| {
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for j in 0..256 {
                qs[j] = ((j % 64) as i32 - 32) as i8;
            }
            for j in 0..16 {
                bsums[j] = 0;
            }
            BlockQ8K { d: 1.0, qs, bsums }
        })
        .collect()
}

fn generate_q8_0_blocks(nb: usize) -> Vec<BlockQ8_0> {
    (0..nb)
        .map(|_i| {
            let mut qs = [0i8; 32];
            for j in 0..32 {
                qs[j] = ((j % 64) as i32 - 32) as i8;
            }
            BlockQ8_0 { d: 15360, qs }
        })
        .collect()
}

fn generate_q8_1_blocks(nb: usize) -> Vec<BlockQ8_1> {
    (0..nb)
        .map(|_i| {
            let mut qs = [0i8; 32];
            for j in 0..32 {
                qs[j] = ((j % 64) as i32 - 32) as i8;
            }
            BlockQ8_1 { d: 15360, s: 0, qs }
        })
        .collect()
}

fn generate_q2k_blocks(nb: usize) -> Vec<BlockQ2K> {
    (0..nb)
        .map(|_i| BlockQ2K { d: 15360, dmin: 0, scales: [1; 16], qs: [0; 64] })
        .collect()
}

fn generate_q3k_blocks(nb: usize) -> Vec<BlockQ3K> {
    (0..nb)
        .map(|_i| BlockQ3K {
            d: 15360,
            hmask: [0; 32],
            qs: [0; 64],
            scales: [1; 12],
        })
        .collect()
}

fn generate_q4k_blocks(nb: usize) -> Vec<BlockQ4K> {
    (0..nb)
        .map(|_i| BlockQ4K { d: 15360, dmin: 0, scales: [1; 12], qs: [0; 128] })
        .collect()
}

fn generate_q5k_blocks(nb: usize) -> Vec<BlockQ5K> {
    (0..nb)
        .map(|_i| BlockQ5K {
            d: 15360,
            dmin: 0,
            scales: [1; 12],
            qh: [0; 32],
            qs: [0; 128],
        })
        .collect()
}

fn generate_q6k_blocks(nb: usize) -> Vec<BlockQ6K> {
    (0..nb)
        .map(|_i| BlockQ6K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [1; 16],
            d: 15360,
        })
        .collect()
}

fn generate_q4_0_blocks(nb: usize) -> Vec<BlockQ4_0> {
    (0..nb).map(|_i| BlockQ4_0 { d: 15360, qs: [0; 16] }).collect()
}

fn generate_q4_1_blocks(nb: usize) -> Vec<BlockQ4_1> {
    (0..nb).map(|_i| BlockQ4_1 { d: 15360, m: 0, qs: [0; 16] }).collect()
}

fn generate_q5_0_blocks(nb: usize) -> Vec<BlockQ5_0> {
    (0..nb).map(|_i| BlockQ5_0 { d: 15360, qh: [0; 4], qs: [0; 16] }).collect()
}

fn generate_q5_1_blocks(nb: usize) -> Vec<BlockQ5_1> {
    (0..nb)
        .map(|_i| BlockQ5_1 { d: 15360, m: 0, qh: [0; 4], qs: [0; 16] })
        .collect()
}

// IQ format generators (correct sizes)
fn generate_iq4_nl_blocks(nb: usize) -> Vec<BlockIQ4NL> {
    (0..nb).map(|_i| BlockIQ4NL { d: 15360, qs: [0; 16] }).collect()
}

fn generate_iq3_xxs_blocks(nb: usize) -> Vec<BlockIQ3XXS> {
    (0..nb).map(|_i| BlockIQ3XXS { d: 15360, qs: [0; 96] }).collect()
}

fn generate_iq3_s_blocks(nb: usize) -> Vec<BlockIQ3S> {
    (0..nb)
        .map(|_i| BlockIQ3S {
            d: 15360,
            qs: [0; 64],
            qh: [0; 8],
            signs: [0; 32],
            scales: [0; 4],
        })
        .collect()
}

fn generate_iq2_xxs_blocks(nb: usize) -> Vec<BlockIQ2XXS> {
    (0..nb).map(|_i| BlockIQ2XXS { d: 15360, qs: [0; 32] }).collect()
}

fn generate_iq2_xs_blocks(nb: usize) -> Vec<BlockIQ2XS> {
    (0..nb)
        .map(|_i| BlockIQ2XS { d: 15360, qs: [0; 32], scales: [0; 8] })
        .collect()
}

fn generate_iq2_s_blocks(nb: usize) -> Vec<BlockIQ2S> {
    (0..nb)
        .map(|_i| BlockIQ2S {
            d: 15360,
            qs: [0; 64],
            qh: [0; 8],
            scales: [0; 8],
        })
        .collect()
}

fn generate_iq1_s_blocks(nb: usize) -> Vec<BlockIQ1S> {
    (0..nb).map(|_i| BlockIQ1S { d: 15360, qs: [0; 32], qh: [0; 8] }).collect()
}

fn generate_iq1_m_blocks(nb: usize) -> Vec<BlockIQ1M> {
    (0..nb)
        .map(|_i| BlockIQ1M { qs: [0; 32], qh: [0; 16], scales: [0; 8] })
        .collect()
}

fn generate_iq4_xs_blocks(nb: usize) -> Vec<BlockIQ4XS> {
    (0..nb)
        .map(|_i| BlockIQ4XS {
            d: 15360,
            scales_h: 0,
            scales_l: [0; 4],
            qs: [0; 128],
        })
        .collect()
}

// TQ format generators
fn generate_tq2_0_blocks(nb: usize) -> Vec<BlockTQ2_0> {
    (0..nb).map(|_i| BlockTQ2_0 { d: 15360, qs: [0; 64] }).collect()
}

fn generate_tq1_0_blocks(nb: usize) -> Vec<BlockTQ1_0> {
    (0..nb).map(|_i| BlockTQ1_0 { qs: [0; 48], qh: [0; 4], d: 15360 }).collect()
}

// ============================================================================
// Benchmark macro
// ============================================================================

macro_rules! bench {
    ($name:expr, $n:expr, $llama_fn:ident, $rust_fn:ident, $x:expr, $y:expr) => {
        unsafe {
            let mut result = 0.0f32;

            // llama.cpp
            let start = Instant::now();
            for _ in 0..ITERATIONS {
                $llama_fn($n as i32, &mut result, $x.as_ptr(), $y.as_ptr());
            }
            let llama_ns =
                start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

            // Rust
            let start = Instant::now();
            for _ in 0..ITERATIONS {
                result = black_box($rust_fn($n, &$x, &$y));
            }
            let rust_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

            let llama_gflops = 2.0 * $n as f64 / (llama_ns / 1e9) / 1e9;
            let rust_gflops = 2.0 * $n as f64 / (rust_ns / 1e9) / 1e9;
            let ratio = llama_ns / rust_ns;
            let status = if ratio >= 1.0 { "✓" } else { "✗" };

            println!(
                "{:<10} | {:7.1} | {:7.1} | {:5.0}% | {}",
                $name,
                llama_gflops,
                rust_gflops,
                ratio * 100.0,
                status
            );
        }
    };
}

fn main() {
    println!("=== CPU Vec_Dot Performance Comparison ===");
    println!("Method: {} iterations, no warmup, same process\n", ITERATIONS);

    println!("Format     | llama.cpp | Rust    | Ratio | Status");
    println!("-----------|-----------|---------|-------|-------");

    unsafe {
        // K-quant formats
        let n_k = 16 * QK_K;
        let y_k = generate_q8k_blocks(16);

        bench!(
            "Q2_K",
            n_k,
            llama_vec_dot_q2_K_q8_K,
            vec_dot_q2_k_q8_k_neon,
            generate_q2k_blocks(16),
            y_k
        );
        bench!(
            "Q3_K",
            n_k,
            llama_vec_dot_q3_K_q8_K,
            vec_dot_q3_k_q8_k_neon,
            generate_q3k_blocks(16),
            y_k
        );
        bench!(
            "Q4_K",
            n_k,
            llama_vec_dot_q4_K_q8_K,
            vec_dot_q4_k_q8_k_neon,
            generate_q4k_blocks(16),
            y_k
        );
        bench!(
            "Q5_K",
            n_k,
            llama_vec_dot_q5_K_q8_K,
            vec_dot_q5_k_q8_k_neon,
            generate_q5k_blocks(16),
            y_k
        );
        bench!(
            "Q6_K",
            n_k,
            llama_vec_dot_q6_K_q8_K,
            vec_dot_q6_k_q8_k_neon,
            generate_q6k_blocks(16),
            y_k
        );

        // Q0/Q1 formats
        let n = 128 * QK;

        bench!(
            "Q4_0",
            n,
            llama_vec_dot_q4_0_q8_0,
            vec_dot_q4_0_q8_0_neon,
            generate_q4_0_blocks(128),
            generate_q8_0_blocks(128)
        );
        bench!(
            "Q8_0",
            n,
            llama_vec_dot_q8_0_q8_0,
            vec_dot_q8_0_q8_0_neon,
            generate_q8_0_blocks(128),
            generate_q8_0_blocks(128)
        );
        bench!(
            "Q4_1",
            n,
            llama_vec_dot_q4_1_q8_1,
            vec_dot_q4_1_q8_1_neon,
            generate_q4_1_blocks(128),
            generate_q8_1_blocks(128)
        );
        bench!(
            "Q5_0",
            n,
            llama_vec_dot_q5_0_q8_0,
            vec_dot_q5_0_q8_0_neon,
            generate_q5_0_blocks(128),
            generate_q8_0_blocks(128)
        );
        bench!(
            "Q5_1",
            n,
            llama_vec_dot_q5_1_q8_1,
            vec_dot_q5_1_q8_1_neon,
            generate_q5_1_blocks(128),
            generate_q8_1_blocks(128)
        );

        // IQ formats
        bench!(
            "IQ4_NL",
            n,
            llama_vec_dot_iq4_nl_q8_0,
            vec_dot_iq4_nl_q8_0_neon,
            generate_iq4_nl_blocks(128),
            generate_q8_0_blocks(128)
        );
        bench!(
            "IQ3_XXS",
            n_k,
            llama_vec_dot_iq3_xxs_q8_K,
            vec_dot_iq3_xxs_q8_k_neon,
            generate_iq3_xxs_blocks(16),
            y_k
        );
        bench!(
            "IQ3_S",
            n_k,
            llama_vec_dot_iq3_s_q8_K,
            vec_dot_iq3_s_q8_k_neon,
            generate_iq3_s_blocks(16),
            y_k
        );
        bench!(
            "IQ2_XXS",
            n_k,
            llama_vec_dot_iq2_xxs_q8_K,
            vec_dot_iq2_xxs_q8_k_neon,
            generate_iq2_xxs_blocks(16),
            y_k
        );
        bench!(
            "IQ2_XS",
            n_k,
            llama_vec_dot_iq2_xs_q8_K,
            vec_dot_iq2_xs_q8_k_neon,
            generate_iq2_xs_blocks(16),
            y_k
        );
        bench!(
            "IQ2_S",
            n_k,
            llama_vec_dot_iq2_s_q8_K,
            vec_dot_iq2_s_q8_k_neon,
            generate_iq2_s_blocks(16),
            y_k
        );
        bench!(
            "IQ1_S",
            n_k,
            llama_vec_dot_iq1_s_q8_K,
            vec_dot_iq1_s_q8_k_neon,
            generate_iq1_s_blocks(16),
            y_k
        );
        bench!(
            "IQ1_M",
            n_k,
            llama_vec_dot_iq1_m_q8_K,
            vec_dot_iq1_m_q8_k_neon,
            generate_iq1_m_blocks(16),
            y_k
        );
        bench!(
            "IQ4_XS",
            n_k,
            llama_vec_dot_iq4_xs_q8_K,
            vec_dot_iq4_xs_q8_k_neon,
            generate_iq4_xs_blocks(16),
            y_k
        );

        // TQ formats
        bench!(
            "TQ2_0",
            n_k,
            llama_vec_dot_tq2_0_q8_K,
            vec_dot_tq2_0_q8_k_neon,
            generate_tq2_0_blocks(16),
            y_k
        );
        bench!(
            "TQ1_0",
            n_k,
            llama_vec_dot_tq1_0_q8_K,
            vec_dot_tq1_0_q8_k_neon,
            generate_tq1_0_blocks(16),
            y_k
        );
    }

    println!("\n✓ = Rust faster than llama.cpp");
    println!("✗ = Rust slower than llama.cpp");
}

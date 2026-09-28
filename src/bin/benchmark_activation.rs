// Benchmark activation functions (GELU, GELU-Quick, SILU) vs llama.cpp

use half::f16;
use infer_train::ops::{gelu_f16_table, gelu_quick_f16_table};
use std::arch::aarch64::*;
use std::time::Instant;

const GELU_COEF_A: f32 = 0.044715;
const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const GELU_QUICK_COEF: f32 = -1.702;

// NEON implementations
#[target_feature(enable = "neon")]
unsafe fn vtanhq_f32(x: float32x4_t) -> float32x4_t {
    let abs_x = vabsq_f32(x);
    let one = vdupq_n_f32(1.0);
    let two = vdupq_n_f32(2.0);
    let four = vdupq_n_f32(4.0);

    let large_mask = vcagtq_f32(x, four);

    let two_x = vmulq_f32(two, x);
    let exp_2x = vexpq_f32(two_x);
    let tanh_approx = vsubq_f32(one, vdivq_f32(two, vaddq_f32(exp_2x, one)));

    let sign = vreinterpretq_f32_u32(vorrq_u32(
        vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x80000000)),
        vreinterpretq_u32_f32(one),
    ));

    vbslq_f32(large_mask, sign, tanh_approx)
}

#[target_feature(enable = "neon")]
unsafe fn vexpq_f32(x: float32x4_t) -> float32x4_t {
    let max_input = vdupq_n_f32(88.0);
    let min_input = vdupq_n_f32(-88.0);
    let x_clamped = vminq_f32(vmaxq_f32(x, min_input), max_input);

    let one = vdupq_n_f32(1.0);
    let log2e = vdupq_n_f32(1.4426950408889634);
    let p1 = vdupq_n_f32(1.0);
    let p2 = vdupq_n_f32(0.5);
    let p3 = vdupq_n_f32(0.16666666666666666);

    let fx = vaddq_f32(vmulq_f32(x_clamped, log2e), one);

    let imm = vreinterpretq_s32_f32(fx);
    let imm = vaddq_s32(imm, vdupq_n_s32(0x3f800000));
    let imm = vshlq_n_s32(imm, 23);

    let y = vsubq_f32(fx, one);
    let x2 = vmulq_f32(y, y);
    let x3 = vmulq_f32(x2, y);

    let t = vmlaq_f32(vmlaq_f32(vmlaq_f32(p1, y, p2), x2, p3), x3, vdupq_n_f32(0.041666666666666664));

    vreinterpretq_f32_s32(vmulq_s32(vreinterpretq_s32_f32(t), imm))
}

#[target_feature(enable = "neon")]
unsafe fn gelu_neon_f32(x: &[f32], dst: &mut [f32]) {
    let n = x.len();
    let chunks = n / 4;

    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let sqrt_2_over_pi = vdupq_n_f32(SQRT_2_OVER_PI);
    let gelu_coef = vdupq_n_f32(GELU_COEF_A);

    for i in 0..chunks {
        let vx = vld1q_f32(x.as_ptr().add(i * 4));
        let x2 = vmulq_f32(vx, vx);
        let inner = vfmaq_f32(one, gelu_coef, x2);
        let tanh_arg = vmulq_f32(vmulq_f32(sqrt_2_over_pi, vx), inner);
        let tanh_val = vtanhq_f32(tanh_arg);
        let one_plus_tanh = vaddq_f32(one, tanh_val);
        let result = vmulq_f32(vmulq_f32(half, vx), one_plus_tanh);
        vst1q_f32(dst.as_mut_ptr().add(i * 4), result);
    }

    for i in (chunks * 4)..n {
        let inner = 1.0 + GELU_COEF_A * x[i] * x[i];
        let tanh_arg = SQRT_2_OVER_PI * x[i] * inner;
        dst[i] = 0.5 * x[i] * (1.0 + tanh_arg.tanh());
    }
}

#[target_feature(enable = "neon")]
unsafe fn gelu_quick_neon_f32(x: &[f32], dst: &mut [f32]) {
    let n = x.len();
    let chunks = n / 4;

    let one = vdupq_n_f32(1.0);
    let quick_coef = vdupq_n_f32(GELU_QUICK_COEF);

    for i in 0..chunks {
        let vx = vld1q_f32(x.as_ptr().add(i * 4));
        let arg = vmulq_f32(quick_coef, vx);
        let exp_arg = vexpq_f32(arg);
        let sigmoid = vdivq_f32(one, vaddq_f32(one, exp_arg));
        let result = vmulq_f32(vx, sigmoid);
        vst1q_f32(dst.as_mut_ptr().add(i * 4), result);
    }

    for i in (chunks * 4)..n {
        dst[i] = x[i] / (1.0 + (GELU_QUICK_COEF * x[i]).exp());
    }
}

#[target_feature(enable = "neon")]
unsafe fn silu_neon_f32(x: &[f32], dst: &mut [f32]) {
    let n = x.len();
    let chunks = n / 4;

    let one = vdupq_n_f32(1.0);

    for i in 0..chunks {
        let vx = vld1q_f32(x.as_ptr().add(i * 4));
        let neg_x = vnegq_f32(vx);
        let exp_neg = vexpq_f32(neg_x);
        let one_plus_exp = vaddq_f32(one, exp_neg);
        let result = vdivq_f32(vx, one_plus_exp);
        vst1q_f32(dst.as_mut_ptr().add(i * 4), result);
    }

    for i in (chunks * 4)..n {
        dst[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

// Scalar implementations
fn gelu_scalar_f32(x: &[f32], dst: &mut [f32]) {
    for i in 0..x.len() {
        let inner = 1.0 + GELU_COEF_A * x[i] * x[i];
        let tanh_arg = SQRT_2_OVER_PI * x[i] * inner;
        dst[i] = 0.5 * x[i] * (1.0 + tanh_arg.tanh());
    }
}

fn gelu_quick_scalar_f32(x: &[f32], dst: &mut [f32]) {
    for i in 0..x.len() {
        dst[i] = x[i] / (1.0 + (GELU_QUICK_COEF * x[i]).exp());
    }
}

fn silu_scalar_f32(x: &[f32], dst: &mut [f32]) {
    for i in 0..x.len() {
        dst[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

fn silu_scalar_f16(x: &[f16], dst: &mut [f16]) {
    for i in 0..x.len() {
        let v = x[i].to_f32();
        dst[i] = f16::from_f32(v / (1.0 + (-v).exp()));
    }
}

fn main() {
    println!("Activation Functions Benchmark\n");

    let sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192];
    let iterations = [10000, 5000, 2000, 1000, 500, 200, 100, 50];

    println!("=== F32: NEON vs Scalar ===\n");
    println!("Size     | GELU NEON | GELU Scalar | Speedup | GELU-Q NEON | GELU-Q Scalar | Speedup | SILU NEON | SILU Scalar | Speedup");
    println!("---------|-----------|-------------|---------|-------------|---------------|---------|-----------|-------------|--------");

    for (i, &n) in sizes.iter().enumerate() {
        let iters = iterations[i];

        let x: Vec<f32> = (0..n).map(|j| ((j % 20) as f32 - 10.0)).collect();
        let mut dst = vec![0.0f32; n];

        // GELU
        for _ in 0..10 { unsafe { gelu_neon_f32(&x, &mut dst); } }
        let start = Instant::now();
        for _ in 0..iters { unsafe { gelu_neon_f32(&x, &mut dst); } }
        let neon_gelu = (start.elapsed().as_secs_f64() / iters as f64) * 1_000_000.0;

        for _ in 0..10 { gelu_scalar_f32(&x, &mut dst); }
        let start = Instant::now();
        for _ in 0..iters { gelu_scalar_f32(&x, &mut dst); }
        let scalar_gelu = (start.elapsed().as_secs_f64() / iters as f64) * 1_000_000.0;

        // GELU-Quick
        for _ in 0..10 { unsafe { gelu_quick_neon_f32(&x, &mut dst); } }
        let start = Instant::now();
        for _ in 0..iters { unsafe { gelu_quick_neon_f32(&x, &mut dst); } }
        let neon_gelu_q = (start.elapsed().as_secs_f64() / iters as f64) * 1_000_000.0;

        for _ in 0..10 { gelu_quick_scalar_f32(&x, &mut dst); }
        let start = Instant::now();
        for _ in 0..iters { gelu_quick_scalar_f32(&x, &mut dst); }
        let scalar_gelu_q = (start.elapsed().as_secs_f64() / iters as f64) * 1_000_000.0;

        // SILU
        for _ in 0..10 { unsafe { silu_neon_f32(&x, &mut dst); } }
        let start = Instant::now();
        for _ in 0..iters { unsafe { silu_neon_f32(&x, &mut dst); } }
        let neon_silu = (start.elapsed().as_secs_f64() / iters as f64) * 1_000_000.0;

        for _ in 0..10 { silu_scalar_f32(&x, &mut dst); }
        let start = Instant::now();
        for _ in 0..iters { silu_scalar_f32(&x, &mut dst); }
        let scalar_silu = (start.elapsed().as_secs_f64() / iters as f64) * 1_000_000.0;

        println!("{:8} | {:9.2} | {:11.2} | {:5.2}x | {:11.2} | {:13.2} | {:5.2}x | {:9.2} | {:11.2} | {:5.2}x",
            n, neon_gelu, scalar_gelu, scalar_gelu / neon_gelu,
            neon_gelu_q, scalar_gelu_q, scalar_gelu_q / neon_gelu_q,
            neon_silu, scalar_silu, scalar_silu / neon_silu);
    }

    println!("\n=== F16: Table Lookup vs Scalar ===\n");
    println!("Size     | GELU Table | GELU Scalar | Speedup | GELU-Q Table | GELU-Q Scalar | Speedup | SILU Scalar");
    println!("---------|------------|-------------|---------|--------------|---------------|---------|------------");

    for (i, &n) in sizes.iter().enumerate() {
        let iters = iterations[i];

        let x_f16: Vec<f16> = (0..n).map(|j| f16::from_f32((j % 20) as f32 - 10.0)).collect();
        let mut dst_f16 = vec![f16::ZERO; n];

        // GELU Table
        for _ in 0..10 { gelu_f16_table(&x_f16, &mut dst_f16); }
        let start = Instant::now();
        for _ in 0..iters { gelu_f16_table(&x_f16, &mut dst_f16); }
        let table_gelu = (start.elapsed().as_secs_f64() / iters as f64) * 1_000_000.0;

        // GELU Scalar
        for _ in 0..10 {
            for j in 0..n {
                let v = x_f16[j].to_f32();
                let inner = 1.0 + GELU_COEF_A * v * v;
                let tanh_arg = SQRT_2_OVER_PI * v * inner;
                dst_f16[j] = f16::from_f32(0.5 * v * (1.0 + tanh_arg.tanh()));
            }
        }
        let start = Instant::now();
        for _ in 0..iters {
            for j in 0..n {
                let v = x_f16[j].to_f32();
                let inner = 1.0 + GELU_COEF_A * v * v;
                let tanh_arg = SQRT_2_OVER_PI * v * inner;
                dst_f16[j] = f16::from_f32(0.5 * v * (1.0 + tanh_arg.tanh()));
            }
        }
        let scalar_gelu_f16 = (start.elapsed().as_secs_f64() / iters as f64) * 1_000_000.0;

        // GELU-Quick Table
        for _ in 0..10 { gelu_quick_f16_table(&x_f16, &mut dst_f16); }
        let start = Instant::now();
        for _ in 0..iters { gelu_quick_f16_table(&x_f16, &mut dst_f16); }
        let table_gelu_q = (start.elapsed().as_secs_f64() / iters as f64) * 1_000_000.0;

        // GELU-Quick Scalar
        for _ in 0..10 {
            for j in 0..n {
                let v = x_f16[j].to_f32();
                dst_f16[j] = f16::from_f32(v / (1.0 + (GELU_QUICK_COEF * v).exp()));
            }
        }
        let start = Instant::now();
        for _ in 0..iters {
            for j in 0..n {
                let v = x_f16[j].to_f32();
                dst_f16[j] = f16::from_f32(v / (1.0 + (GELU_QUICK_COEF * v).exp()));
            }
        }
        let scalar_gelu_q_f16 = (start.elapsed().as_secs_f64() / iters as f64) * 1_000_000.0;

        // SILU (no table)
        for _ in 0..10 { silu_scalar_f16(&x_f16, &mut dst_f16); }
        let start = Instant::now();
        for _ in 0..iters { silu_scalar_f16(&x_f16, &mut dst_f16); }
        let scalar_silu_f16 = (start.elapsed().as_secs_f64() / iters as f64) * 1_000_000.0;

        println!("{:8} | {:10.2} | {:11.2} | {:5.2}x | {:12.2} | {:13.2} | {:5.2}x | {:10.2}",
            n, table_gelu, scalar_gelu_f16, scalar_gelu_f16 / table_gelu,
            table_gelu_q, scalar_gelu_q_f16, scalar_gelu_q_f16 / table_gelu_q,
            scalar_silu_f16);
    }
}

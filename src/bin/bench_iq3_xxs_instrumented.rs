// Instrumented IQ3_XXS to measure each step
// Compare with llama.cpp to find bottlenecks

use std::arch::aarch64::*;
use std::arch::asm;
use std::hint::black_box;
use std::ptr;
use std::time::Instant;

use infer_train::quant::types::*;
use infer_train::quant::vec_dot::arm::fp::fp16_to_fp32;
use infer_train::quant::vec_dot::tables::{IQ3XXS_GRID, KEVEN_SIGNS_Q2XS};

mod ffi {
    use super::*;

    unsafe extern "C" {
        pub fn llama_vec_dot_iq3_xxs_q8_K(
            n: i32,
            s: *mut f32,
            vx: *const BlockIQ3XXS,
            vy: *const BlockQ8K,
        );
    }
}

// Helper function
#[target_feature(enable = "neon")]
unsafe fn vld1q_u32_from_scalars(w: u32, x: u32, y: u32, z: u32) -> uint32x4_t {
    vcombine_u32(
        vreinterpret_u32_u64(vcreate_u64((w as u64) | ((x as u64) << 32))),
        vreinterpret_u32_u64(vcreate_u64((y as u64) | ((z as u64) << 32))),
    )
}

// Manual SDOT
#[target_feature(enable = "neon,dotprod")]
unsafe fn instr_vdotq_s32(
    acc: int32x4_t,
    a: int8x16_t,
    b: int8x16_t,
) -> int32x4_t {
    let mut result = acc;
    asm!(
        "sdot {0}.4s, {1}.16b, {2}.16b",
        inout(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, preserves_flags)
    );
    result
}

// Instrumented version
#[target_feature(enable = "neon,dotprod")]
unsafe fn vec_dot_iq3_xxs_instrumented(
    n: usize,
    x: &[BlockIQ3XXS],
    y: &[BlockQ8K],
) -> [f64; 8] {
    let nb = n / QK_K;
    let grid = IQ3XXS_GRID.as_ptr();
    let signs = KEVEN_SIGNS_Q2XS.as_ptr() as *const i8;

    // Timing accumulators (in nanoseconds)
    let mut t_q8_load = 0.0;
    let mut t_aux_load = 0.0;
    let mut t_grid_lookup = 0.0;
    let mut t_sign_extract = 0.0;
    let mut t_sign_apply = 0.0;
    let mut t_dot = 0.0;
    let mut t_accum = 0.0;
    let mut t_overhead = 0.0;

    let half = 0.5f32;
    let zero = vdupq_n_s32(0);

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let mut qs_ptr = x[i].qs.as_ptr();
        // BlockIQ3XXS layout: qs[0..64] = quantized values, qs[64..96] = gas values
        let mut gas_ptr = qs.add(64);
        let mut q8_ptr = y[i].qs.as_ptr();

        let mut sumf1 = 0.0f32;
        let mut sumf2 = 0.0f32;

        for _ in 0..4 {
            // Step 1: Q8 load
            let t0 = Instant::now();
            let q8b = vld1q_s8_x4(black_box(q8));
            t_q8_load += t0.elapsed().as_nanos() as f64;
            q8_ptr = q8_ptr.add(64);

            // Step 2: aux32 load
            let t1 = Instant::now();
            let aux32_0 = ptr::read_unaligned(black_box(gas as *const u32));
            let aux32_1 =
                ptr::read_unaligned(black_box(gas.add(4) as *const u32));
            t_aux_load += t1.elapsed().as_nanos() as f64;
            gas_ptr = gas_ptr.add(8);

            // Step 3: Grid lookup
            let t2 = Instant::now();
            let v0 = vld1q_u32_from_scalars(
                *black_box(grid.add(*qs.add(0) as usize)),
                *black_box(grid.add(*qs.add(1) as usize)),
                *black_box(grid.add(*qs.add(2) as usize)),
                *black_box(grid.add(*qs.add(3) as usize)),
            );
            let v1 = vld1q_u32_from_scalars(
                *black_box(grid.add(*qs.add(4) as usize)),
                *black_box(grid.add(*qs.add(5) as usize)),
                *black_box(grid.add(*qs.add(6) as usize)),
                *black_box(grid.add(*qs.add(7) as usize)),
            );
            let v2 = vld1q_u32_from_scalars(
                *black_box(grid.add(*qs.add(8) as usize)),
                *black_box(grid.add(*qs.add(9) as usize)),
                *black_box(grid.add(*qs.add(10) as usize)),
                *black_box(grid.add(*qs.add(11) as usize)),
            );
            let v3 = vld1q_u32_from_scalars(
                *black_box(grid.add(*qs.add(12) as usize)),
                *black_box(grid.add(*qs.add(13) as usize)),
                *black_box(grid.add(*qs.add(14) as usize)),
                *black_box(grid.add(*qs.add(15) as usize)),
            );
            t_grid_lookup += t2.elapsed().as_nanos() as f64;
            qs_ptr = qs_ptr.add(16);

            // Step 4: Sign extraction using ubfx
            let t3 = Instant::now();
            let (s0, s1, s2, s3): (u64, u64, u64, u64);
            let (s4, s5, s6, s7): (u64, u64, u64, u64);
            asm!(
                "ubfx {s0}, {aux0}, #0, #7",
                "ubfx {s1}, {aux0}, #7, #7",
                "ubfx {s2}, {aux0}, #14, #7",
                "ubfx {s3}, {aux0}, #21, #7",
                "ubfx {s4}, {aux1}, #0, #7",
                "ubfx {s5}, {aux1}, #7, #7",
                "ubfx {s6}, {aux1}, #14, #7",
                "ubfx {s7}, {aux1}, #21, #7",
                aux0 = in(reg) aux32_0 as u64,
                aux1 = in(reg) aux32_1 as u64,
                s0 = out(reg) s0,
                s1 = out(reg) s1,
                s2 = out(reg) s2,
                s3 = out(reg) s3,
                s4 = out(reg) s4,
                s5 = out(reg) s5,
                s6 = out(reg) s6,
                s7 = out(reg) s7,
                options(pure, nomem, preserves_flags)
            );
            t_sign_extract += t3.elapsed().as_nanos() as f64;

            // Step 5: Sign application
            let t4 = Instant::now();
            let q3s_0 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(black_box(signs.add(s0 as usize * 8))),
                    vld1_s8(black_box(signs.add(s1 as usize * 8))),
                ),
                vreinterpretq_s8_u32(v0),
            );
            let q3s_1 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(black_box(signs.add(s2 as usize * 8))),
                    vld1_s8(black_box(signs.add(s3 as usize * 8))),
                ),
                vreinterpretq_s8_u32(v1),
            );
            let q3s_2 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(black_box(signs.add(s4 as usize * 8))),
                    vld1_s8(black_box(signs.add(s5 as usize * 8))),
                ),
                vreinterpretq_s8_u32(v2),
            );
            let q3s_3 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(black_box(signs.add(s6 as usize * 8))),
                    vld1_s8(black_box(signs.add(s7 as usize * 8))),
                ),
                vreinterpretq_s8_u32(v3),
            );
            t_sign_apply += t4.elapsed().as_nanos() as f64;

            // Step 6: Dot products
            let t5 = Instant::now();
            let p1 = instr_vdotq_s32(
                instr_vdotq_s32(zero, q3s_0, q8b.0),
                q3s_1,
                q8b.1,
            );
            let p2 = instr_vdotq_s32(
                instr_vdotq_s32(zero, q3s_2, q8b.2),
                q3s_3,
                q8b.3,
            );
            t_dot += t5.elapsed().as_nanos() as f64;

            // Step 7: Accumulation
            let t6 = Instant::now();
            sumf1 += vaddvq_s32(p1) as f32 * (half + (aux32_0 >> 28) as f32);
            sumf2 += vaddvq_s32(p2) as f32 * (half + (aux32_1 >> 28) as f32);
            t_accum += t6.elapsed().as_nanos() as f64;
        }

        let _ = d * (sumf1 + sumf2);
    }

    let total = t_q8_load
        + t_aux_load
        + t_grid_lookup
        + t_sign_extract
        + t_sign_apply
        + t_dot
        + t_accum;
    t_overhead = 0.0; // Will be calculated by caller

    [
        t_q8_load,
        t_aux_load,
        t_grid_lookup,
        t_sign_extract,
        t_sign_apply,
        t_dot,
        t_accum,
        total,
    ]
}

fn main() {
    unsafe {
        let n = 16 * QK_K;
        let x = generate_iq3_xxs_blocks(16);
        let y = generate_q8k_blocks(16);

        let iterations = 1000;

        // Warmup
        for _ in 0..100 {
            let mut result = 0.0f32;
            ffi::llama_vec_dot_iq3_xxs_q8_K(
                n as i32,
                &mut result,
                x.as_ptr(),
                y.as_ptr(),
            );
            vec_dot_iq3_xxs_instrumented(n, &x, &y);
        }

        // Instrumented Rust
        println!("=== Rust Instrumented (1000 iterations) ===");
        let mut totals = [0.0; 8];
        for _ in 0..iterations {
            let times = vec_dot_iq3_xxs_instrumented(n, &x, &y);
            for i in 0..8 {
                totals[i] += times[i];
            }
        }

        println!(
            "Step                    | Total (ns) | Per iter (ns) | % of total"
        );
        println!(
            "------------------------|------------|---------------|----------"
        );
        let grand_total = totals[7];
        let steps = [
            "Q8 load",
            "aux32 load",
            "Grid lookup",
            "Sign extract",
            "Sign apply",
            "Dot product",
            "Accumulate",
        ];
        for (i, step) in steps.iter().enumerate() {
            let total = totals[i];
            let per_iter = total / iterations as f64;
            let pct = (total / grand_total) * 100.0;
            println!(
                "{:<23} | {:10.1} | {:13.2} | {:6.1}%",
                step, total, per_iter, pct
            );
        }
        println!(
            "------------------------|------------|---------------|----------"
        );
        println!(
            "{:<23} | {:10.1} | {:13.2} | {:6.1}%",
            "TOTAL",
            grand_total,
            grand_total / iterations as f64,
            100.0
        );

        // Baseline llama.cpp (total time only)
        println!("\n=== llama.cpp Baseline (1000 iterations) ===");
        let start = Instant::now();
        for _ in 0..iterations {
            let mut result = 0.0f32;
            ffi::llama_vec_dot_iq3_xxs_q8_K(
                n as i32,
                &mut result,
                x.as_ptr(),
                y.as_ptr(),
            );
        }
        let llama_total = start.elapsed().as_nanos() as f64;
        println!(
            "llama.cpp total: {:.1} ns ({:.2} ns/iter)",
            llama_total,
            llama_total / iterations as f64
        );
        println!(
            "Rust total:      {:.1} ns ({:.2} ns/iter)",
            grand_total,
            grand_total / iterations as f64
        );
        println!(
            "Ratio:           {:.1}%",
            (llama_total / grand_total) * 100.0
        );
    }
}

fn generate_iq3_xxs_blocks(count: usize) -> Vec<BlockIQ3XXS> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut blocks = Vec::with_capacity(count);

    for _ in 0..count {
        let mut qs = [0u8; 96];
        for j in 0..96 {
            qs[j] = rng.r#gen();
        }
        let d = half::f16::from_f32(rng.gen_range(0.1..2.0)).to_bits();

        blocks.push(BlockIQ3XXS { qs, d });
    }
    blocks
}

fn generate_q8k_blocks(count: usize) -> Vec<BlockQ8K> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut blocks = Vec::with_capacity(count);

    for _ in 0..count {
        let mut qs = [0i8; QK_K];
        let mut bsums = [0i16; QK_K / 16];
        for j in 0..QK_K {
            qs[j] = rng.gen_range(-128..127);
        }
        for j in 0..QK_K / 16 {
            let sum: i32 =
                qs[j * 16..(j + 1) * 16].iter().map(|&x| x as i32).sum();
            bsums[j] = sum as i16;
        }
        let d = rng.gen_range(0.1..2.0);

        blocks.push(BlockQ8K { qs, bsums, d });
    }
    blocks
}

fn instr_vdotq_s32(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    use std::arch::asm;
    let mut result = acc;
    asm!(
        "sdot {0}.4s, {1}.16b, {2}.16b",
        inout(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, preserves_flags)
    );
    result
}

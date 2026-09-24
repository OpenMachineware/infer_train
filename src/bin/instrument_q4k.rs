#![allow(unused_unsafe)]

use infer_train::quant::types::*;
use std::arch::aarch64::*;
use std::time::Instant;

const QK_K: usize = 256;
const ITERATIONS: usize = 100000;

type vreg = int32x4_t;

#[inline(always)]
unsafe fn vdotq_s32_manual(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    use std::arch::asm;
    let mut result = acc;
    asm!(
        "sdot {0}.4s, {1}.16b, {2}.16b",
        inout(vreg) result,
        in(vreg) a,
        in(vreg) b,
    );
    result
}

unsafe fn vld1q_u8_x2(ptr: *const u8) -> (uint8x16_t, uint8x16_t) {
    (vld1q_u8(ptr), vld1q_u8(ptr.add(16)))
}

unsafe fn vld1q_s8_x2(ptr: *const i8) -> (int8x16_t, int8x16_t) {
    (vld1q_s8(ptr), vld1q_s8(ptr.add(16)))
}

fn generate_q8_k(k: usize) -> Vec<BlockQ8K> {
    let nb = k / QK_K;
    let mut blocks = Vec::with_capacity(nb);
    for b in 0..nb {
        let mut qs = [0i8; QK_K];
        for i in 0..QK_K {
            qs[i] = ((i + b * QK_K) % 256) as i8;
        }
        let mut bsums = [0i16; 16];
        for i in 0..16 {
            let start = i * 16;
            let mut sum: i32 = 0;
            for j in 0..16 {
                sum += qs[start + j] as i32;
            }
            bsums[i] = sum as i16;
        }
        blocks.push(BlockQ8K { d: 1.0, qs, bsums });
    }
    blocks
}

fn generate_q4_k(k: usize) -> Vec<BlockQ4K> {
    let nb = k / QK_K;
    let mut blocks = Vec::with_capacity(nb);
    for b in 0..nb {
        let mut scales = [0u8; 12];
        for i in 0..12 {
            scales[i] = ((i + b) % 64) as u8;
        }
        let qs = [((b % 256) as u8); 128];
        blocks.push(BlockQ4K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.5).to_bits(),
            scales,
            qs,
        });
    }
    blocks
}

fn main() {
    println!("Q4_K Instrumentation Analysis (time per step)");
    println!("==============================================");
    println!();

    let k = 4096;
    let x = generate_q8_k(k);
    let y = generate_q4_k(k);
    let nb = k / QK_K;

    // Warmup
    for _ in 0..1000 {
        unsafe {
            for i in 0..nb {
                let x_i = y.get_unchecked(i);
                let y_i = x.get_unchecked(i);
                let _ = y_i.d * half::f16::from_bits(x_i.d).to_f32();
            }
        }
    }

    // Full kernel baseline
    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        unsafe {
            let _ = infer_train::quant::vec_dot::arm::vec_dot_q4_k_q8_k_neon(k, &y, &x);
        }
    }
    let baseline_ns = t0.elapsed().as_nanos();

    // Step 1: Scale decode only
    let t1 = Instant::now();
    for _ in 0..ITERATIONS {
        unsafe {
            for i in 0..nb {
                let x_i = y.get_unchecked(i);
                let d = x[0].d * half::f16::from_bits(x_i.d).to_f32();
                let dmin = x[0].d * half::f16::from_bits(x_i.dmin).to_f32();
                std::hint::black_box((d, dmin));
            }
        }
    }
    let scale_decode_ns = t1.elapsed().as_nanos();

    // Step 2: Min calculation only
    let t2 = Instant::now();
    for _ in 0..ITERATIONS {
        unsafe {
            for i in 0..nb {
                let y_i = x.get_unchecked(i);
                let q8sums = vpaddq_s16(vld1q_s16(y_i.bsums.as_ptr()), vld1q_s16(y_i.bsums.as_ptr().add(8)));
                std::hint::black_box(q8sums);
            }
        }
    }
    let min_calc_ns = t2.elapsed().as_nanos();

    // Step 3: Inner loop only (load + compute + scale multiply + accumulate)
    let t3 = Instant::now();
    for _ in 0..ITERATIONS {
        unsafe {
            for i in 0..nb {
                let x_i = y.get_unchecked(i);
                let y_i = x.get_unchecked(i);
                let m4b = vdupq_n_u8(0x0F);
                let vzero = vdupq_n_s32(0);

                // Decode scales (matching llama.cpp)
                let mut utmp = [0u32; 4];
                std::ptr::copy_nonoverlapping(x_i.scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);

                const KMASK1: u32 = 0x3f3f3f3f;
                const KMASK2: u32 = 0x0f0f0f0f;
                const KMASK3: u32 = 0x03030303;

                utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
                utmp[0] &= KMASK1;
                let scales = utmp.as_ptr() as *const u8;

                let mut q4 = x_i.qs.as_ptr();
                let mut q8 = y_i.qs.as_ptr();
                let mut sc = scales;

                let mut sumi1 = 0i32;
                let mut sumi2 = 0i32;

                for _ in 0..(QK_K / 64) {
                    let q4bits = vld1q_u8_x2(q4);
                    let q8bytes = vld1q_s8_x2(q8);

                    let q4l_0 = vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b));
                    let q4l_1 = vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b));
                    let p1 = vdotq_s32_manual(vdotq_s32_manual(vzero, q4l_0, q8bytes.0), q4l_1, q8bytes.1);
                    sumi1 += vaddvq_s32(p1) * *sc as i32;

                    let q8bytes_1 = vld1q_s8_x2(q8.add(32));
                    let q4h_0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4));
                    let q4h_1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4));
                    let p2 = vdotq_s32_manual(vdotq_s32_manual(vzero, q4h_0, q8bytes_1.0), q4h_1, q8bytes_1.1);
                    sumi2 += vaddvq_s32(p2) * *sc.add(1) as i32;

                    q4 = q4.add(32);
                    q8 = q8.add(64);
                    sc = sc.add(2);
                }
                std::hint::black_box((sumi1, sumi2));
            }
        }
    }
    let inner_loop_ns = t3.elapsed().as_nanos();

    let total_per_iter = baseline_ns as f64 / ITERATIONS as f64;

    println!("K = {} (nb = {})", k, nb);
    println!("Iterations: {}", ITERATIONS);
    println!();
    println!("Step                      | Total (ns) | Per block (ns) | Percent");
    println!("--------------------------+------------+----------------+--------");
    println!("Full kernel               | {:10} | {:14.1} | 100.0%", baseline_ns, total_per_iter / nb as f64);
    println!("  Scale decode            | {:10} | {:14.1} | {:5.1}%",
        scale_decode_ns,
        scale_decode_ns as f64 / ITERATIONS as f64 / nb as f64,
        scale_decode_ns as f64 / baseline_ns as f64 * 100.0);
    println!("  Min calculation         | {:10} | {:14.1} | {:5.1}%",
        min_calc_ns,
        min_calc_ns as f64 / ITERATIONS as f64 / nb as f64,
        min_calc_ns as f64 / baseline_ns as f64 * 100.0);
    println!("  Inner loop (load+dot)   | {:10} | {:14.1} | {:5.1}%",
        inner_loop_ns,
        inner_loop_ns as f64 / ITERATIONS as f64 / nb as f64,
        inner_loop_ns as f64 / baseline_ns as f64 * 100.0);
}

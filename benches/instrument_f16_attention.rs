// Instrument F16 attention to find bottleneck
use std::arch::aarch64::*;
use std::time::Instant;

const NUM_ITER: usize = 1000;
const HEAD_DIM: usize = 128;

fn main() {
    let q_f32: Vec<f32> = vec![0.5; HEAD_DIM];
    let q_f16: Vec<u16> =
        (0..HEAD_DIM).map(|_| half::f16::from_f32(0.5).to_bits()).collect();
    let k_f16: Vec<u16> =
        (0..HEAD_DIM).map(|_| half::f16::from_f32(0.3).to_bits()).collect();
    let v_f16: Vec<u16> =
        (0..HEAD_DIM).map(|_| half::f16::from_f32(0.4).to_bits()).collect();

    unsafe {
        // Test 1: FP16 dot product
        let mut total_dot = 0u64;
        for _ in 0..NUM_ITER {
            let start = Instant::now();
            let _score = vec_dot_f16_test(&q_f16, &k_f16, HEAD_DIM);
            total_dot += start.elapsed().as_nanos() as u64;
        }
        println!(
            "FP16 vec_dot: {} ns/iter (head_dim={})",
            total_dot / NUM_ITER as u64,
            HEAD_DIM
        );

        // Test 2: FP16 vec_mad
        let mut vkq = v_f16.clone();
        let mut total_mad = 0u64;
        for _ in 0..NUM_ITER {
            let start = Instant::now();
            vec_mad_f16_test(&mut vkq, &v_f16, 0.5, HEAD_DIM);
            total_mad += start.elapsed().as_nanos() as u64;
        }
        println!("FP16 vec_mad: {} ns/iter", total_mad / NUM_ITER as u64);

        // Test 3: FP16 vec_scale
        let mut total_scale = 0u64;
        for _ in 0..NUM_ITER {
            let start = Instant::now();
            vec_scale_f16_test(&mut vkq, 0.5, HEAD_DIM);
            total_scale += start.elapsed().as_nanos() as u64;
        }
        println!("FP16 vec_scale: {} ns/iter", total_scale / NUM_ITER as u64);

        // Test 4: FP32 -> FP16 conversion
        let mut total_conv = 0u64;
        let mut q_f16_out: Vec<u16> = vec![0; HEAD_DIM];
        for _ in 0..NUM_ITER {
            let start = Instant::now();
            for d in (0..HEAD_DIM).step_by(8) {
                let q0 = vld1q_f32(q_f32.as_ptr().add(d));
                let q1 = vld1q_f32(q_f32.as_ptr().add(d + 4));
                let q_f16_low = vcvt_f16_f32(q0);
                let q_f16_vec = vcvt_high_f16_f32(q_f16_low, q1);
                vst1q_u16(
                    q_f16_out.as_mut_ptr().add(d),
                    vreinterpretq_u16_f16(q_f16_vec),
                );
            }
            total_conv += start.elapsed().as_nanos() as u64;
        }
        println!("FP32->FP16 conv: {} ns/iter", total_conv / NUM_ITER as u64);
    }
}

#[target_feature(enable = "neon")]
unsafe fn vec_dot_f16_test(q: &[u16], k: &[u16], n: usize) -> f32 {
    let q_ptr = q.as_ptr() as *const float16x8_t;
    let k_ptr = k.as_ptr() as *const float16x8_t;

    let zero_f16 = vreinterpretq_f16_u16(vdupq_n_u16(0));
    let mut sum0 = zero_f16;
    let mut sum1 = zero_f16;
    let mut sum2 = zero_f16;
    let mut sum3 = zero_f16;

    let np = n & !31;
    let mut i = 0;

    // Use ldp-style loading via inline assembly for better performance
    while i < np {
        // Load pairs (ldp equivalent) - 4 vectors per iteration
        let (q0, q1) = load_pair_f16(q_ptr.add(i / 8));
        let (q2, q3) = load_pair_f16(q_ptr.add(i / 8 + 2));
        let (k0, k1) = load_pair_f16(k_ptr.add(i / 8));
        let (k2, k3) = load_pair_f16(k_ptr.add(i / 8 + 2));

        sum0 = vfmaq_f16_asm(sum0, q0, k0);
        sum1 = vfmaq_f16_asm(sum1, q1, k1);
        sum2 = vfmaq_f16_asm(sum2, q2, k2);
        sum3 = vfmaq_f16_asm(sum3, q3, k3);

        i += 32;
    }

    sum0 = vaddq_f16_asm(sum0, sum1);
    sum2 = vaddq_f16_asm(sum2, sum3);
    sum0 = vaddq_f16_asm(sum0, sum2);

    let sum_f32_0 = vcvt_f32_f16(vget_low_f16(sum0));
    let sum_f32_1 = vcvt_high_f32_f16(sum0);
    vaddvq_f32(vaddq_f32(sum_f32_0, sum_f32_1))
}

/// Load pair of float16x8_t
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_pair_f16(ptr: *const float16x8_t) -> (float16x8_t, float16x8_t) {
    // Use vld1q for simplicity - ldp with vector registers has strict ordering requirements
    let v0 = vld1q_f16(ptr as *const f16);
    let v1 = vld1q_f16(ptr.add(1) as *const f16);
    (v0, v1)
}

#[target_feature(enable = "neon")]
unsafe fn vec_mad_f16_test(y: &mut [u16], x: &[u16], v: f32, n: usize) {
    let y_ptr = y.as_mut_ptr();
    let x_ptr = x.as_ptr();
    let v_bits = half::f16::from_f32(v).to_bits();
    let v_f16 = vreinterpretq_f16_u16(vdupq_n_u16(v_bits));

    let np = n & !7;
    let mut i = 0;

    while i < np {
        let x8 = vreinterpretq_f16_u16(vld1q_u16(x_ptr.add(i)));
        let y8 = vreinterpretq_f16_u16(vld1q_u16(y_ptr.add(i)));
        let result = vfmaq_f16_asm(y8, x8, v_f16);
        vst1q_u16(y_ptr.add(i), vreinterpretq_u16_f16(result));
        i += 8;
    }
}

#[target_feature(enable = "neon")]
unsafe fn vec_scale_f16_test(y: &mut [u16], v: f32, n: usize) {
    let y_ptr = y.as_mut_ptr();
    let v_bits = half::f16::from_f32(v).to_bits();
    let v_f16 = vreinterpretq_f16_u16(vdupq_n_u16(v_bits));

    let np = n & !7;
    let mut i = 0;

    while i < np {
        let y8 = vreinterpretq_f16_u16(vld1q_u16(y_ptr.add(i)));
        let result = vmulq_f16_asm(y8, v_f16);
        vst1q_u16(y_ptr.add(i), vreinterpretq_u16_f16(result));
        i += 8;
    }
}

use std::arch::asm;

#[target_feature(enable = "neon")]
#[inline]
unsafe fn vfmaq_f16_asm(
    a: float16x8_t,
    b: float16x8_t,
    c: float16x8_t,
) -> float16x8_t {
    let result: float16x8_t;
    asm!(
        "fmla {0:v}.8h, {1:v}.8h, {2:v}.8h",
        inout(vreg) a => result,
        in(vreg) b,
        in(vreg) c,
        options(pure, nomem, nostack)
    );
    result
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn vaddq_f16_asm(a: float16x8_t, b: float16x8_t) -> float16x8_t {
    let result: float16x8_t;
    asm!(
        "fadd {0:v}.8h, {1:v}.8h, {2:v}.8h",
        lateout(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, nostack)
    );
    result
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn vmulq_f16_asm(a: float16x8_t, b: float16x8_t) -> float16x8_t {
    let result: float16x8_t;
    asm!(
        "fmul {0:v}.8h, {1:v}.8h, {2:v}.8h",
        lateout(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, nostack)
    );
    result
}

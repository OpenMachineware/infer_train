// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use std::arch::aarch64::*;
use std::arch::asm;
use std::time::Instant;

const NUM_ITER: usize = 1000000;
const SIZE: usize = 128;

fn main() {
    let mut a: Vec<f32> = vec![0.5; SIZE];
    let mut b: Vec<f32> = vec![0.3; SIZE];

    // Modify data each iteration to prevent optimization
    let mut sum_intrinsics: f32 = 0.0;
    let mut sum_asm: f32 = 0.0;

    unsafe {
        let t1 = Instant::now();
        for iter in 0..NUM_ITER {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();

            let a0 = vld1q_f32(a_ptr);
            let a1 = vld1q_f32(a_ptr.add(4));
            let b0 = vld1q_f32(b_ptr);
            let b1 = vld1q_f32(b_ptr.add(4));
            let sum0 = vfmaq_f32(vdupq_n_f32(0.0), a0, b0);
            let sum1 = vfmaq_f32(vdupq_n_f32(0.0), a1, b1);

            let a2 = vld1q_f32(a_ptr.add(8));
            let a3 = vld1q_f32(a_ptr.add(12));
            let b2 = vld1q_f32(b_ptr.add(8));
            let b3 = vld1q_f32(b_ptr.add(12));
            let sum2 = vfmaq_f32(vdupq_n_f32(0.0), a2, b2);
            let sum3 = vfmaq_f32(vdupq_n_f32(0.0), a3, b3);

            sum_intrinsics = vaddvq_f32(vaddq_f32(
                vaddq_f32(sum0, sum1),
                vaddq_f32(sum2, sum3),
            ));

            // Touch data to prevent optimization
            a[0] = iter as f32;
        }
        let intrinsics_ns = t1.elapsed().as_nanos() as f64 / NUM_ITER as f64;

        let t2 = Instant::now();
        for iter in 0..NUM_ITER {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let mut score: f32 = 0.0;
            asm!(
                "movi v0.2d, #0\nmovi v1.2d, #0\nmovi v2.2d, #0\nmovi v3.2d, #0",
                "ldp q4, q5, [{0}]\nldp q6, q7, [{1}]\nfmla v0.4s, v4.4s, v6.4s\nfmla v1.4s, v5.4s, v7.4s",
                "ldp q4, q5, [{0}, #32]\nldp q6, q7, [{1}, #32]\nfmla v2.4s, v4.4s, v6.4s\nfmla v3.4s, v5.4s, v7.4s",
                "fadd v0.4s, v0.4s, v1.4s\nfadd v2.4s, v2.4s, v3.4s\nfadd v0.4s, v0.4s, v2.4s\naddv s0, v0.4s\nstr s0, [{2}]",
                in(reg) a_ptr, in(reg) b_ptr, in(reg) &mut score as *mut f32 as *const f32,
                out("v0") _, options(nostack)
            );
            sum_asm = score;
            b[0] = iter as f32;
        }
        let asm_ns = t2.elapsed().as_nanos() as f64 / NUM_ITER as f64;

        println!(
            "Intrinsics: {:.2} ns (sum={})",
            intrinsics_ns, sum_intrinsics
        );
        println!("ASM ldp:    {:.2} ns (sum={})", asm_ns, sum_asm);
        println!("Ratio:      {:.2}x", intrinsics_ns / asm_ns);
    }
}

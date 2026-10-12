// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test fcvt vs ucvtf for FP16 conversion
use std::arch::aarch64::*;
use std::time::Instant;

#[inline(always)]
unsafe fn fp16_to_fp32_fcvt(bits: u16) -> f32 {
    let f16_vec = vdup_n_u16(bits);
    let f16 = vreinterpret_f16_u16(f16_vec);
    let f32_vec = vcvt_f32_f16(f16);
    vgetq_lane_f32(f32_vec, 0)
}

#[inline(always)]
unsafe fn fp16_to_fp32_ucvtf(bits: u16) -> f32 {
    let mut result = 0.0f32;
    std::arch::asm!(
        "mov {0:h}, {1:w}",
        "ucvtf {0:s}, {0:s}",
        inout(vreg) result,
        in(reg) bits as u32,
    );
    result
}

fn main() {
    unsafe {
        let values: Vec<u16> =
            (0..10000).map(|i| ((i % 1000) + 15360) as u16).collect(); // Valid FP16 range

        // Warmup
        for _ in 0..10000 {
            for &v in &values {
                std::hint::black_box(fp16_to_fp32_fcvt(v));
            }
        }

        // Test fcvt
        let start = Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..10000 {
            for &v in &values {
                sum += fp16_to_fp32_fcvt(v);
            }
        }
        let elapsed_fcvt = start.elapsed().as_nanos();

        // Warmup
        for _ in 0..10000 {
            for &v in &values {
                std::hint::black_box(fp16_to_fp32_ucvtf(v));
            }
        }

        // Test ucvtf
        let start = Instant::now();
        let mut sum2 = 0.0f32;
        for _ in 0..10000 {
            for &v in &values {
                sum2 += fp16_to_fp32_ucvtf(v);
            }
        }
        let elapsed_ucvtf = start.elapsed().as_nanos();

        println!("fcvt:   {} ns", elapsed_fcvt);
        println!("ucvtf:  {} ns", elapsed_ucvtf);
        println!(
            "Ratio:  {:.2}%",
            elapsed_ucvtf as f64 / elapsed_fcvt as f64 * 100.0
        );
        println!("Sum check: {} {}", sum, sum2);
    }
}

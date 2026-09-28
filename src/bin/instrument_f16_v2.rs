// Direct assembly test without function call overhead
use std::arch::asm;
use std::time::Instant;

const NUM_ITER: usize = 1000;
const HEAD_DIM: usize = 128;

fn main() {
    let q_f16: Vec<u16> = vec![0x3800; HEAD_DIM];
    let k_f16: Vec<u16> = vec![0x3400; HEAD_DIM];

    unsafe {
        let q_ptr = q_f16.as_ptr();
        let k_ptr = k_f16.as_ptr();
        let mut score: f32 = 0.0;

        // Test: Full vec_dot in inline assembly
        let start = Instant::now();
        for _ in 0..NUM_ITER {
            // Inline everything
            asm!(
                // Zero accumulators
                "movi v0.2d, #0",
                "movi v1.2d, #0",
                "movi v2.2d, #0",
                "movi v3.2d, #0",

                // Loop: load 4 pairs and FMA
                "ldp q4, q5, [{0}]",
                "ldp q6, q7, [{1}]",
                "fmla v0.8h, v4.8h, v6.8h",
                "fmla v1.8h, v5.8h, v7.8h",

                "ldp q4, q5, [{0}, #32]",
                "ldp q6, q7, [{1}, #32]",
                "fmla v2.8h, v4.8h, v6.8h",
                "fmla v3.8h, v5.8h, v7.8h",

                // Reduce
                "fadd v0.8h, v0.8h, v1.8h",
                "fadd v2.8h, v2.8h, v3.8h",
                "fadd v0.8h, v0.8h, v2.8h",

                // Convert to FP32 and sum
                "fcvtl2 v1.4s, v0.8h",
                "fcvtl v0.4s, v0.4h",
                "fadd v0.4s, v0.4s, v1.4s",
                "addv s0, v0.4s",

                in(reg) q_ptr,
                in(reg) k_ptr,
                out("v0") score,
                options(nostack)
            );
        }
        let elapsed = start.elapsed();
        let ns_per_iter = elapsed.as_nanos() as u64 / NUM_ITER as u64;
        println!("Pure ASM vec_dot: {} ns/iter", ns_per_iter);

        // Also test just the timing overhead
        let start = Instant::now();
        for _ in 0..NUM_ITER {
            asm!("nop", options(nostack));
        }
        let elapsed = start.elapsed();
        let ns_per_iter = elapsed.as_nanos() as u64 / NUM_ITER as u64;
        println!("Empty loop overhead: {} ns/iter", ns_per_iter);
    }
}

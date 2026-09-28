use std::arch::asm;
use std::time::Instant;

const NUM_ITER: usize = 100000;
const HEAD_DIM: usize = 128;

static mut DUMMY: f32 = 0.0;

#[target_feature(enable = "neon")]
unsafe fn vec_dot_target_feature(q_ptr: *const u16, k_ptr: *const u16) -> f32 {
    let mut score: f32 = 0.0;
    asm!(
        "movi v0.2d, #0\nmovi v1.2d, #0\nmovi v2.2d, #0\nmovi v3.2d, #0",
        "ldp q4, q5, [{0}]\nldp q6, q7, [{1}]\nfmla v0.8h, v4.8h, v6.8h\nfmla v1.8h, v5.8h, v7.8h",
        "ldp q4, q5, [{0}, #32]\nldp q6, q7, [{1}, #32]\nfmla v2.8h, v4.8h, v6.8h\nfmla v3.8h, v5.8h, v7.8h",
        "fadd v0.8h, v0.8h, v1.8h\nfadd v2.8h, v2.8h, v3.8h\nfadd v0.8h, v0.8h, v2.8h",
        "fcvtl2 v1.4s, v0.8h\nfcvtl v0.4s, v0.4h\nfadd v0.4s, v0.4s, v1.4s\naddv s0, v0.4s",
        in(reg) q_ptr,
        in(reg) k_ptr,
        out("v0") score,
        options(nostack)
    );
    score
}

fn main() {
    let q_f16: Vec<u16> = vec![0x3800; HEAD_DIM];
    let k_f16: Vec<u16> = vec![0x3400; HEAD_DIM];

    unsafe {
        let q_ptr = q_f16.as_ptr();
        let k_ptr = k_f16.as_ptr();

        // Warmup
        for _ in 0..1000 {
            DUMMY = vec_dot_target_feature(q_ptr, k_ptr);
        }

        // Test
        let start = Instant::now();
        for _ in 0..NUM_ITER {
            DUMMY = vec_dot_target_feature(q_ptr, k_ptr);
        }
        let elapsed = start.elapsed();
        println!(
            "#[target_feature]: {} ns/iter",
            elapsed.as_nanos() as u64 / NUM_ITER as u64
        );
    }
}

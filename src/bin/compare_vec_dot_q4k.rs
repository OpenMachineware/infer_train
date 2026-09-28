// Compare Rust vs llama.cpp vec_dot performance directly
// Calls both implementations with identical data
use std::arch::aarch64::*;
use std::ffi::c_void;
use std::time::Instant;

// Rust types
#[repr(C)]
#[derive(Clone, Copy)]
struct BlockQ4K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qs: [u8; 128],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockQ8K {
    d: f32,
    qs: [i8; 256],
    bsums: [i16; 16],
}

// External llama.cpp function
extern "C" {
    fn ggml_vec_dot_q4_K_q8_K(
        n: i32,
        s: *mut f32,
        bs: usize,
        vx: *const c_void,
        bx: usize,
        vy: *const c_void,
        by: usize,
        nrc: i32,
    );
}

// Our Rust implementation
mod rust_impl {
    use super::*;

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn vec_dot_q4_k_q8_k_neon(
        n: usize,
        x: &[BlockQ4K],
        y: &[BlockQ8K],
    ) -> f32 {
        let m4b = vdupq_n_u8(0x0F);
        let vzero = vdupq_n_s32(0);

        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        let nb = n / 256;
        let mut sum = 0.0f32;

        for i in 0..nb {
            let x_i = x.get_unchecked(i);
            let y_i = y.get_unchecked(i);

            let d = y_i.d * half::f16::from_bits(x_i.d).to_f32();
            let dmin = y_i.d * half::f16::from_bits(x_i.dmin).to_f32();

            let q8sums = vpaddq_s16(
                vld1q_s16(y_i.bsums.as_ptr()),
                vld1q_s16(y_i.bsums.as_ptr().add(8)),
            );

            let mut utmp = [0u32; 4];
            std::ptr::copy_nonoverlapping(
                x_i.scales.as_ptr(),
                utmp.as_mut_ptr() as *mut u8,
                12,
            );

            let mut mins8 = vdup_n_u32(0);
            mins8 = vset_lane_u32(utmp[1] & KMASK1, mins8, 0);
            mins8 = vset_lane_u32(
                ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4),
                mins8,
                1,
            );

            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[0] &= KMASK1;

            let mins =
                vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
            let prod = vaddq_s32(
                vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
                vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
            );
            let min_sum = vaddvq_s32(prod);

            let sc = std::slice::from_raw_parts(utmp.as_ptr() as *const u8, 8);

            let mut q4 = x_i.qs.as_ptr();
            let mut q8 = y_i.qs.as_ptr();

            let mut sumi1 = 0i32;
            let mut sumi2 = 0i32;

            for _ in 0..4 {
                let q4bits = vld1q_u8_x2(q4);
                let q8bytes = vld1q_s8_x2(q8);

                let q4l_0 = vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b));
                let q4l_1 = vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b));

                let p1 = vdotq_s32(
                    vdotq_s32(vzero, q4l_0, q8bytes.0),
                    q4l_1,
                    q8bytes.1,
                );
                sumi1 += vaddvq_s32(p1) * sc[0] as i32;

                let q8bytes_1 = vld1q_s8_x2(q8.add(32));
                let q4h_0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4));
                let q4h_1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4));

                let p2 = vdotq_s32(
                    vdotq_s32(vzero, q4h_0, q8bytes_1.0),
                    q4h_1,
                    q8bytes_1.1,
                );
                sumi2 += vaddvq_s32(p2) * sc[1] as i32;

                q4 = q4.add(32);
                q8 = q8.add(64);
            }

            sum -= dmin * min_sum as f32;
            sum += d * (sumi1 + sumi2) as f32;
        }

        sum
    }
}

// Manual SDOT via inline assembly (same as we use)
#[target_feature(enable = "neon,dotprod")]
unsafe fn vdotq_s32(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
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

fn generate_q4_k_data(nb: usize) -> Vec<BlockQ4K> {
    (0..nb)
        .map(|i| {
            let mut scales = [0u8; 12];
            let mut qs = [0u8; 128];
            for j in 0..12 {
                scales[j] = ((i * 17 + j * 13) % 64) as u8;
            }
            for j in 0..128 {
                qs[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            BlockQ4K {
                d: half::f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: half::f16::from_f32(0.1 + (i % 5) as f32 * 0.05)
                    .to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn generate_q8_k_data(nb: usize) -> Vec<BlockQ8K> {
    (0..nb)
        .map(|i| {
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for j in 0..256 {
                qs[j] = (((i * 17 + j * 13) % 256) as i32 - 128) as i8;
            }
            for j in 0..16 {
                bsums[j] = ((i * 23 + j * 7) % 256) as i16;
            }
            BlockQ8K { d: 0.01 + (i % 10) as f32 * 0.001, qs, bsums }
        })
        .collect()
}

/// Warm up CPU
fn warmup() {
    let mut sum = 0.0f64;
    for i in 0..50_000_000 {
        sum = sum.sqrt() + i as f64;
    }
    if sum < 0.0 {
        println!("This should never print");
    }
}

fn main() {
    println!("Q4_K Performance Comparison: Rust vs llama.cpp");
    println!("================================================\n");

    // Warm up
    println!("Warming up CPU...");
    warmup();
    std::thread::sleep(std::time::Duration::from_millis(500));

    let n = 4096;
    let nb = n / 256;
    let iterations = 1000;

    let x = generate_q4_k_data(nb);
    let y = generate_q8_k_data(nb);

    // Test Rust implementation
    println!("\nTesting Rust implementation...");
    let start = Instant::now();
    let mut result_rust = 0.0f32;
    for _ in 0..iterations {
        unsafe {
            result_rust = rust_impl::vec_dot_q4_k_q8_k_neon(n, &x, &y);
        }
    }
    let elapsed_rust = start.elapsed();
    let ops_rust = iterations as f64 * 2.0 * n as f64;
    let gflops_rust = ops_rust / elapsed_rust.as_secs_f64() / 1e9;

    println!("  Result: {}", result_rust);
    println!("  Time: {:.2} ms", elapsed_rust.as_secs_f64() * 1000.0);
    println!("  Performance: {:.2} GFLOPS", gflops_rust);

    // Test llama.cpp implementation
    println!("\nTesting llama.cpp implementation...");
    let start = Instant::now();
    let mut result_llama = 0.0f32;
    for _ in 0..iterations {
        unsafe {
            ggml_vec_dot_q4_K_q8_K(
                n as i32,
                &mut result_llama,
                0,
                x.as_ptr() as *const c_void,
                0,
                y.as_ptr() as *const c_void,
                0,
                1,
            );
        }
    }
    let elapsed_llama = start.elapsed();
    let ops_llama = iterations as f64 * 2.0 * n as f64;
    let gflops_llama = ops_llama / elapsed_llama.as_secs_f64() / 1e9;

    println!("  Result: {}", result_llama);
    println!("  Time: {:.2} ms", elapsed_llama.as_secs_f64() * 1000.0);
    println!("  Performance: {:.2} GFLOPS", gflops_llama);

    // Comparison
    println!("\n{}", "=".repeat(50));
    println!("Comparison:");
    println!("  Rust:     {:.2} GFLOPS", gflops_rust);
    println!("  llama.cpp: {:.2} GFLOPS", gflops_llama);
    println!(
        "  Ratio:    {:.1}% of llama.cpp",
        gflops_rust / gflops_llama * 100.0
    );
    println!(
        "  Correctness: {}",
        if (result_rust - result_llama).abs() < 0.01 {
            "✓ Match"
        } else {
            "✗ Mismatch"
        }
    );
}

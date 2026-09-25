// Benchmark CPU Softmax vs llama.cpp
// llama.cpp uses NEON SIMD with custom exp approximation

use infer_train::ops::softmax::softmax_f32_dispatch;

// llama.cpp-style scalar softmax for comparison
fn softmax_scalar(x: &[f32], y: &mut [f32]) -> f32 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }

    // Find max
    let mut max = x[0];
    for &xi in &x[1..] {
        if xi > max {
            max = xi;
        }
    }

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for i in 0..n {
        let val = (x[i] - max).exp();
        y[i] = val;
        sum += val;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for yi in y.iter_mut() {
        *yi *= inv_sum;
    }

    sum
}

// llama.cpp NEON softmax (from vec.cpp)
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn softmax_neon_llama(x: &[f32], y: &mut [f32]) -> f32 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }

    // Find max
    let mut max = f32::NEG_INFINITY;
    let mut i = 0;
    for chunk in x.chunks_exact(4) {
        let v = vld1q_f32(chunk.as_ptr());
        let chunk_max = vmaxvq_f32(v);
        if chunk_max > max {
            max = chunk_max;
        }
        i += 4;
    }
    for &xi in &x[i..] {
        if xi > max {
            max = xi;
        }
    }

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    i = 0;
    for (x_chunk, y_chunk) in x.chunks_exact(4).zip(y.chunks_exact_mut(4)) {
        let vx = vld1q_f32(x_chunk.as_ptr());
        let vmax = vdupq_n_f32(max);
        let vsub = vsubq_f32(vx, vmax);
        let vexp = ggml_v_expf(vsub);
        vst1q_f32(y_chunk.as_mut_ptr(), vexp);
        sum += vaddvq_f32(vexp);
        i += 4;
    }
    for j in i..n {
        let val = (x[j] - max).exp();
        y[j] = val;
        sum += val;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    i = 0;
    for y_chunk in y.chunks_exact_mut(4) {
        let vy = vld1q_f32(y_chunk.as_ptr());
        let vinv = vdupq_n_f32(inv_sum);
        let vscaled = vmulq_f32(vy, vinv);
        vst1q_f32(y_chunk.as_mut_ptr(), vscaled);
        i += 4;
    }
    for yi in y[i..].iter_mut() {
        *yi *= inv_sum;
    }

    sum
}

// llama.cpp's ggml_v_expf implementation
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn ggml_v_expf(x: float32x4_t) -> float32x4_t {
    let r = vdupq_n_f32(12582912.0);
    let z = vfmaq_f32(r, x, vdupq_n_f32(1.4426950408889634));
    let n = vsubq_f32(z, r);
    let b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0.6931381225585938)), n, vdupq_n_f32(1.908214e-10));
    let e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    let k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1.0f32))));
    let c = vcagtq_f32(n, vdupq_n_f32(126.0f32));
    let u = vmulq_f32(b, b);
    let j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0.9999990463256836), b),
        vfmaq_f32(
            vfmaq_f32(vdupq_n_f32(0.4999985098838806), vdupq_n_f32(0.1666666169166565), b),
            vfmaq_f32(vdupq_n_f32(0.04166661170125008), vdupq_n_f32(0.008333338060230017), b),
            u,
        ),
        u,
    );
    if vpaddd_u64(vreinterpretq_u64_u32(c)) == 0 {
        return vfmaq_f32(k, j, k);
    }
    let d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000u32));
    let s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000u32)));
    let s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    vbslq_f32(
        vcagtq_f32(n, vdupq_n_f32(192.0f32)),
        vmulq_f32(s1, s1),
        vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)),
    )
}

fn main() {
    println!("=== CPU Softmax: Our Implementation vs llama.cpp ===\n");

    // Test dimensions: typical attention sizes
    let sizes = vec![64, 128, 256, 512, 1024, 2048, 4096];
    let iterations = 10000;

    println!("Note: Both use NEON SIMD with same exp approximation\n");
    println!("{:<10} {:>15} {:>15} {:>10}", "Size", "Ours (ns)", "llama (ns)", "Perf");
    println!("{}", "-".repeat(55));

    // Warmup CPU
    let warmup_x: Vec<f32> = (0..1024).map(|i| (i as f32 % 100.0) / 10.0).collect();
    let mut warmup_y = vec![0.0f32; 1024];
    for _ in 0..1000 {
        let _ = softmax_f32_dispatch(&warmup_x, &mut warmup_y);
    }
    std::thread::sleep(std::time::Duration::from_millis(100));

    for size in &sizes {
        let x: Vec<f32> = (0..*size).map(|i| ((i % 100) as f32) / 10.0).collect();
        let mut y_ours = vec![0.0f32; *size];
        let mut y_llama = vec![0.0f32; *size];

        // Benchmark our implementation
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = softmax_f32_dispatch(&x, &mut y_ours);
        }
        let t_ours = start.elapsed().as_nanos() as f64 / iterations as f64;

        // Benchmark llama.cpp-style
        #[cfg(target_arch = "aarch64")]
        {
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = unsafe { softmax_neon_llama(&x, &mut y_llama) };
            }
            let t_llama = start.elapsed().as_nanos() as f64 / iterations as f64;

            let ratio = t_llama / t_ours;
            println!("{:<10} {:>15.2} {:>15.2} {:>10.1}%", size, t_ours, t_llama, ratio * 100.0);
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = softmax_scalar(&x, &mut y_llama);
            }
            let t_llama = start.elapsed().as_nanos() as f64 / iterations as f64;
            println!("{:<10} {:>15.2} {:>15.2} (scalar)", size, t_ours, t_llama);
        }
    }

    // Correctness check
    println!("\n=== Correctness ===");
    let test_x: Vec<f32> = (0..128).map(|i| (i as f32 % 50.0) / 10.0).collect();
    let mut y_ours = vec![0.0f32; 128];
    let mut y_llama = vec![0.0f32; 128];

    let _ = softmax_f32_dispatch(&test_x, &mut y_ours);
    #[cfg(target_arch = "aarch64")]
    let _ = unsafe { softmax_neon_llama(&test_x, &mut y_llama) };
    #[cfg(not(target_arch = "aarch64"))]
    let _ = softmax_scalar(&test_x, &mut y_llama);

    let mut max_diff = 0.0f32;
    for i in 0..128 {
        let diff = (y_ours[i] - y_llama[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    // Verify sum = 1
    let sum: f32 = y_ours.iter().sum();
    println!("Sum of probabilities: {:.6}", sum);
    println!("Max diff vs llama.cpp: {:.2e}", max_diff);
}

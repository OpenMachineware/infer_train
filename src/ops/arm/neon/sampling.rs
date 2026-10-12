// NEON SIMD implementations for sampling operations

use std::arch::aarch64::*;

/// NEON optimized argmax
#[target_feature(enable = "neon")]
pub unsafe fn argmax_f32(data: &[f32]) -> usize {
    let n = data.len();
    let mut max_val = vdupq_n_f32(f32::NEG_INFINITY);
    let mut max_idx = vdupq_n_u32(0);

    // Process 4 elements at a time
    let n4 = n - n % 4;
    for i in (0..n4).step_by(4) {
        let vals = vld1q_f32(data.as_ptr().add(i));
        let idxs = vld1q_u32(
            [i as u32, (i + 1) as u32, (i + 2) as u32, (i + 3) as u32].as_ptr(),
        );

        // Compare: if vals > max_val, use new idxs
        let gt = vcgtq_f32(vals, max_val);
        max_val = vbslq_f32(gt, vals, max_val);
        max_idx = vbslq_u32(gt, idxs, max_idx);
    }

    // Find max among the 4 lanes
    let max_vals: [f32; 4] = std::mem::transmute(max_val);
    let max_idxs: [u32; 4] = std::mem::transmute(max_idx);

    let mut best_idx = max_idxs[0] as usize;
    let mut best_val = max_vals[0];
    for i in 1..4 {
        if max_vals[i] > best_val {
            best_val = max_vals[i];
            best_idx = max_idxs[i] as usize;
        }
    }

    // Handle tail
    for i in n4..n {
        if data[i] > best_val {
            best_val = data[i];
            best_idx = i;
        }
    }

    best_idx
}

/// Initialize indices for argsort using NEON
#[target_feature(enable = "neon")]
pub unsafe fn init_indices_neon(indices: &mut [i32]) {
    let n = indices.len();
    let ptr = indices.as_mut_ptr();
    let step = vdupq_n_s32(16);
    let mut v = vdupq_n_s32(0);
    let mut i = 0usize;

    while i + 16 <= n {
        vst1q_s32(ptr.add(i), v);
        v = vaddq_s32(v, step);
        i += 16;
    }

    while i < n {
        *ptr.add(i) = i as i32;
        i += 1;
    }
}

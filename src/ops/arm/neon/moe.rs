// MoE operator NEON implementations

use std::arch::aarch64::*;

/// Matrix multiplication with expert selection (NEON optimized)
/// For each expert, select corresponding rows and compute GEMV
/// weights: [n_experts, n_out, n_in] expert weights
/// expert_ids: [n_tokens] expert assignment per token
/// input: [n_tokens, n_in] input vectors
/// output: [n_tokens, n_out] output vectors
#[target_feature(enable = "neon")]
pub unsafe fn mul_mat_id_f32_neon(
    weights: &[f32],
    _n_experts: usize,
    n_out: usize,
    n_in: usize,
    expert_ids: &[i32],
    input: &[f32],
    output: &mut [f32],
) {
    let expert_size = n_out * n_in;

    for (token_idx, &expert_id) in expert_ids.iter().enumerate() {
        let expert = expert_id as usize;
        let w_offset = expert * expert_size;
        let w = weights.as_ptr().add(w_offset);

        let x_offset = token_idx * n_in;
        let x = input.as_ptr().add(x_offset);

        let y_offset = token_idx * n_out;
        let y = output.as_mut_ptr().add(y_offset);

        // GEMV: y = W^T * x (each row of W dot x)
        for i in 0..n_out {
            let row = w.add(i * n_in);

            // NEON dot product
            let mut sum_vec = vdupq_n_f32(0.0);
            let mut j = 0;

            // Process 16 elements at a time (4 x float32x4_t)
            while j + 16 <= n_in {
                let v0 = vld1q_f32(row.add(j));
                let v1 = vld1q_f32(row.add(j + 4));
                let v2 = vld1q_f32(row.add(j + 8));
                let v3 = vld1q_f32(row.add(j + 12));

                let x0 = vld1q_f32(x.add(j));
                let x1 = vld1q_f32(x.add(j + 4));
                let x2 = vld1q_f32(x.add(j + 8));
                let x3 = vld1q_f32(x.add(j + 12));

                sum_vec = vmlaq_f32(sum_vec, v0, x0);
                sum_vec = vmlaq_f32(sum_vec, v1, x1);
                sum_vec = vmlaq_f32(sum_vec, v2, x2);
                sum_vec = vmlaq_f32(sum_vec, v3, x3);

                j += 16;
            }

            // Process 4 elements at a time
            while j + 4 <= n_in {
                let v = vld1q_f32(row.add(j));
                let xv = vld1q_f32(x.add(j));
                sum_vec = vmlaq_f32(sum_vec, v, xv);
                j += 4;
            }

            // Horizontal sum
            let mut sum = vaddvq_f32(sum_vec);

            // Tail
            while j < n_in {
                sum += *row.add(j) * *x.add(j);
                j += 1;
            }

            *y.add(i) = sum;
        }
    }
}

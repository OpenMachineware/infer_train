//! MoE (Mixture of Experts) operators: GET_ROWS, SET_ROWS, ADD_ID, MUL_MAT_ID

/// Get specific rows from a matrix
/// src: [n_rows, n_cols] matrix
/// indices: [n_indices] row indices
/// dst: [n_indices, n_cols] output
pub fn get_rows_f32(
    src: &[f32],
    n_cols: usize,
    indices: &[i32],
    dst: &mut [f32],
) {
    for (out_row, &idx) in indices.iter().enumerate() {
        let src_row = idx as usize;
        let src_offset = src_row * n_cols;
        let dst_offset = out_row * n_cols;
        dst[dst_offset..dst_offset + n_cols]
            .copy_from_slice(&src[src_offset..src_offset + n_cols]);
    }
}

/// Set rows in a matrix
/// src: [n_updates, n_cols] values
/// indices: [n_updates] row indices
/// dst: [n_rows, n_cols] matrix to update
pub fn set_rows_f32(
    src: &[f32],
    n_cols: usize,
    indices: &[i32],
    dst: &mut [f32],
) {
    for (upd_row, &idx) in indices.iter().enumerate() {
        let dst_row = idx as usize;
        let src_offset = upd_row * n_cols;
        let dst_offset = dst_row * n_cols;
        dst[dst_offset..dst_offset + n_cols]
            .copy_from_slice(&src[src_offset..src_offset + n_cols]);
    }
}

/// Add with broadcasting by ID
/// src: [n_ids, n_cols] values
/// ids: [n_ids] id for each row
/// dst: [max_id + 1, n_cols] accumulator
pub fn add_id_f32(src: &[f32], n_cols: usize, ids: &[i32], dst: &mut [f32]) {
    for (row, &id) in ids.iter().enumerate() {
        let dst_row = id as usize;
        let src_row = &src[row * n_cols..][..n_cols];
        let dst_row_slice = &mut dst[dst_row * n_cols..][..n_cols];

        // Use zip iterator for potential vectorization
        for (d, s) in dst_row_slice.iter_mut().zip(src_row.iter()) {
            *d += s;
        }
    }
}

/// Matrix multiplication with expert selection
/// For each expert, select corresponding rows and compute GEMV
/// weights: [n_experts, n_out, n_in] expert weights
/// expert_ids: [n_tokens] expert assignment per token
/// input: [n_tokens, n_in] input vectors
/// output: [n_tokens, n_out] output vectors
pub fn mul_mat_id_f32(
    weights: &[f32],
    n_experts: usize,
    n_out: usize,
    n_in: usize,
    expert_ids: &[i32],
    input: &[f32],
    output: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            crate::ops::arm::neon::mul_mat_id_f32_neon(
                weights, n_experts, n_out, n_in, expert_ids, input, output,
            )
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        mul_mat_id_f32_scalar(
            weights, n_experts, n_out, n_in, expert_ids, input, output,
        )
    }
}

/// Scalar fallback for mul_mat_id_f32
#[cfg(not(target_arch = "aarch64"))]
fn mul_mat_id_f32_scalar(
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

        let x_offset = token_idx * n_in;
        let y_offset = token_idx * n_out;

        // GEMV: y = W^T * x (each row of W dot x)
        for i in 0..n_out {
            let row_offset = w_offset + i * n_in;
            let mut sum = 0.0f32;
            for j in 0..n_in {
                sum += weights[row_offset + j] * input[x_offset + j];
            }
            output[y_offset + i] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_rows() {
        let src = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let indices = [1, 0]; // Get row 1, then row 0
        let mut dst = [0.0f32; 6];
        get_rows_f32(&src, 3, &indices, &mut dst);
        assert_eq!(dst, [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_set_rows() {
        let src = [7.0, 8.0, 9.0]; // 1x3
        let indices = [0];
        let mut dst = [0.0f32; 6]; // 2x3
        set_rows_f32(&src, 3, &indices, &mut dst);
        assert_eq!(dst[0..3], [7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_mul_mat_id() {
        // 2 experts, each 2x2 weight matrix
        let weights = [
            1.0, 0.0, 0.0, 1.0, // expert 0: identity
            2.0, 0.0, 0.0, 2.0,
        ]; // expert 1: 2x
        let expert_ids = [0, 1]; // token 0 uses expert 0, token 1 uses expert 1
        let input = [1.0, 2.0, 3.0, 4.0]; // 2 tokens, each 2 values
        let mut output = [0.0f32; 4];

        mul_mat_id_f32(&weights, 2, 2, 2, &expert_ids, &input, &mut output);

        // Token 0: expert 0 (identity) * [1,2] = [1, 2]
        // Token 1: expert 1 (2x) * [3,4] = [6, 8]
        assert_eq!(output, [1.0, 2.0, 6.0, 8.0]);
    }
}

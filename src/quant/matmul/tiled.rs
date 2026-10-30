// SPDX-License-Identifier: AGPL-3.0-or-later
// Tiled GEMV/GEMM implementation matching llama.cpp architecture
//
// Key design from llama.cpp:
// 1. 256x256 macrotile, 16x16 microtile
// 2. Per-thread workspace (512KB) for tile buffers
// 3. Narrow path for decode (batch <= 16) with long contiguous reads
// 4. Accumulator buffer to avoid direct dst writes

use crate::quant::types::{BlockQ5K, BlockQ8K, QK_K};

/// Tile dimensions matching llama.cpp
pub const TILE_K: usize = 256; // one QK_K block
pub const TILE_ROWS: usize = 256; // max window rows
pub const MICRO: usize = 16; // microtile edge

/// Per-thread workspace for tiled matmul
/// Size: ~512KB per thread
#[repr(C, align(64))]
pub struct TiledWorkspace {
    /// Weight tile (src0): quantized weights unpacked to uint8
    pub src0_q: [u8; TILE_ROWS * TILE_K],
    /// Weight scales (per-row)
    pub src0_d: [f32; TILE_ROWS],
    /// Weight mins (for Q4_K, Q5_K with mins)
    pub src0_dmin: [f32; TILE_ROWS],
    /// Input tile (src1): Q8_K data unpacked
    pub src1_q: [i8; TILE_ROWS * TILE_K],
    /// Input scales (per-row)
    pub src1_d: [f32; TILE_ROWS],
    /// Input sums (per-16-block)
    pub src1_bsums: [i32; (TILE_K / 16) * TILE_ROWS],
    /// Accumulator buffer
    pub acc: [f32; TILE_ROWS * TILE_ROWS],
}

impl TiledWorkspace {
    pub fn new() -> Self {
        Self {
            src0_q: [0u8; TILE_ROWS * TILE_K],
            src0_d: [0.0f32; TILE_ROWS],
            src0_dmin: [0.0f32; TILE_ROWS],
            src1_q: [0i8; TILE_ROWS * TILE_K],
            src1_d: [0.0f32; TILE_ROWS],
            src1_bsums: [0i32; (TILE_K / 16) * TILE_ROWS],
            acc: [0.0f32; TILE_ROWS * TILE_ROWS],
        }
    }

    /// Clear accumulator
    #[inline]
    pub fn clear_acc(&mut self, n_rows: usize, n_cols: usize) {
        for i in 0..n_rows * n_cols {
            self.acc[i] = 0.0;
        }
    }
}

impl Default for TiledWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Global thread-local workspace pool
thread_local! {
    static WORKSPACE: std::cell::RefCell<TiledWorkspace> =
        std::cell::RefCell::new(TiledWorkspace::new());
}

/// Tiled Q5_K x Q8_K GEMM
/// For GEMV, set n = 1 (single input vector)
///
/// Parameters:
/// - weights: weight matrix (m x k/QK_K blocks)
/// - x: input matrix (n x k/QK_K blocks), n=1 for GEMV
/// - dst: output matrix (m x n elements)
/// - k: K dimension (number of elements per row)
/// - m: M dimension (number of output rows)
/// - n: N dimension (number of input vectors)
#[target_feature(enable = "neon")]
pub unsafe fn matmul_q5k_q8k_tiled(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
    n: usize,
) {
    let nb = k / QK_K;

    // Determine path: narrow (N <= 16) or standard
    // Narrow path is for decode with small batch
    if n <= MICRO {
        // Narrow path: optimize for decode
        // Use longer k_extent for better memory bandwidth
        matmul_q5k_q8k_tiled_narrow(weights, x, dst, k, m, n, nb);
    } else {
        // Standard path: tiled 256x256
        matmul_q5k_q8k_tiled_standard(weights, x, dst, k, m, n, nb);
    }
}

/// Narrow path: N <= 16 (small batch), optimize for decode
/// Key optimization: process K in larger chunks for better memory bandwidth
#[target_feature(enable = "neon")]
unsafe fn matmul_q5k_q8k_tiled_narrow(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    _k: usize,
    m: usize,
    n: usize,
    nb: usize,
) {
    // For narrow path, use longer contiguous K reads
    // Calculate k_extent based on N to maximize bandwidth
    let num_k = if n <= 4 { 4 } else { 2 }; // number of 256-element slabs

    WORKSPACE.with(|ws_cell| {
        let mut ws = ws_cell.borrow_mut();

        // Process each input vector
        for iv in 0..n {
            let x_row = &x[iv * nb..(iv + 1) * nb];

            // Process each output row
            for row in 0..m {
                // Clear accumulator for this output element
                ws.acc[row] = 0.0;
            }

            // Process in k_extent chunks
            for k0 in (0..nb).step_by(num_k) {
                let k_end = (k0 + num_k).min(nb);
                let actual_k_blocks = k_end - k0;
                let partial_k = actual_k_blocks * QK_K;

                // Compute partial dot products for all output rows
                for row in 0..m {
                    let w_slice = &weights[row * nb + k0..row * nb + k_end];
                    let x_slice = &x_row[k0..k_end];

                    let dot =
                        crate::quant::vec_dot::arm::vec_dot_q5_k_q8_k_neon(
                            partial_k, w_slice, x_slice,
                        );
                    ws.acc[row] += dot;
                }
            }

            // Store result from accumulator
            for row in 0..m {
                dst[row * n + iv] = ws.acc[row];
            }
        }
    });
}

/// Standard tiled path: 256x256 tiles
/// For larger batches
#[target_feature(enable = "neon")]
unsafe fn matmul_q5k_q8k_tiled_standard(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    _k: usize,
    m: usize,
    n: usize,
    nb: usize,
) {
    WORKSPACE.with(|ws_cell| {
        let mut ws = ws_cell.borrow_mut();

        // Process 256x256 macrotiles
        for iir1 in (0..n).step_by(TILE_ROWS) {
            let ir1_end = (iir1 + TILE_ROWS).min(n);
            let _n_src1 = ir1_end - iir1;

            for iir0 in (0..m).step_by(TILE_ROWS) {
                let ir0_end = (iir0 + TILE_ROWS).min(m);
                let n_src0 = ir0_end - iir0;

                // Clear accumulator for this macrotile
                ws.clear_acc(n_src0, ir1_end - iir1);

                // Process K dimension in 256-element slabs (one block at a time)
                for ib in 0..nb {
                    // Run 16x16 microtiles
                    for ir1 in (iir1..ir1_end).step_by(MICRO) {
                        for ir0 in (iir0..ir0_end).step_by(MICRO) {
                            // Run microkernel
                            for r1 in ir1..ir1.min(ir1 + MICRO) {
                                for r0 in ir0..ir0.min(ir0 + MICRO) {
                                    let w_slice = &weights[r0 * nb + ib..r0 * nb + ib + 1];
                                    let x_slice = &x[r1 * nb + ib..r1 * nb + ib + 1];
                                    let dot = crate::quant::vec_dot::arm::vec_dot_q5_k_q8_k_neon(
                                        QK_K, // single block
                                        w_slice,
                                        x_slice,
                                    );
                                    ws.acc[(r1 - iir1) * TILE_ROWS + (r0 - iir0)] += dot;
                                }
                            }
                        }
                    }
                }

                // Store accumulator to dst
                for ir1 in iir1..ir1_end {
                    for ir0 in iir0..ir0_end {
                        dst[ir0 * n + ir1] = ws.acc[(ir1 - iir1) * TILE_ROWS + (ir0 - iir0)];
                    }
                }
            }
        }
    });
}

/// GEMV wrapper (single vector)
#[target_feature(enable = "neon")]
pub unsafe fn gemv_q5k_q8k_tiled(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    matmul_q5k_q8k_tiled(weights, x, dst, k, m, 1);
}

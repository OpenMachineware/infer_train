use crate::quant::types::*;
use crate::quant::vec_dot::arm::*;

/// Quantized matrix-vector multiplication: dst = weights @ x
///
/// weights: [K, N] in quantized format (row-major)
/// x: [K] in Q8_K format
/// dst: [N] output in FP32
pub unsafe fn matmul_q4_k_q8_k(
    weights: &[BlockQ4K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    ne00: usize,  // N (output dimension)
    ne01: usize,  // K (input dimension, number of weight rows)
) {
    // Block tiling: 16x16
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK_K;  // Number of blocks per row
    let ne01_blocks = ne01;  // Number of rows (each weight row is one block)

    // Temporary buffer for block results
    let mut tmp = [0.0f32; 32];

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_block = &x[ir1..ir1 + 1];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    tmp[ir0 - iir0] = vec_dot_q4_k_q8_k_neon(QK_K, w_block, x_block);
                }

                // Copy results to dst
                for (i, &val) in tmp[..(iir0 + BLCK_0).min(ne00_blocks) - iir0].iter().enumerate() {
                    dst[ir1 * ne00_blocks + iir0 + i] = val;
                }
            }
        }
    }
}

/// Q6_K × Q8_K matrix-vector multiplication
pub unsafe fn matmul_q6_k_q8_k(
    weights: &[BlockQ6K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK_K;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_block = &x[ir1..ir1 + 1];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_q6_k_q8_k_neon(QK_K, w_block, x_block);
                }
            }
        }
    }
}

/// Q5_K × Q8_K matrix-vector multiplication
pub unsafe fn matmul_q5_k_q8_k(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK_K;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_block = &x[ir1..ir1 + 1];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_q5_k_q8_k_neon(QK_K, w_block, x_block);
                }
            }
        }
    }
}

/// Q3_K × Q8_K matrix-vector multiplication
pub unsafe fn matmul_q3_k_q8_k(
    weights: &[BlockQ3K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK_K;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_block = &x[ir1..ir1 + 1];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_q3_k_q8_k_neon(QK_K, w_block, x_block);
                }
            }
        }
    }
}

/// Q2_K × Q8_K matrix-vector multiplication
pub unsafe fn matmul_q2_k_q8_k(
    weights: &[BlockQ2K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK_K;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_block = &x[ir1..ir1 + 1];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_q2_k_q8_k_neon(QK_K, w_block, x_block);
                }
            }
        }
    }
}

/// Q4_0 × Q8_0 matrix-vector multiplication
pub unsafe fn matmul_q4_0_q8_0(
    weights: &[BlockQ4_0],
    x: &[BlockQ8_0],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK4_0;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_row = &x[ir1 * ne00_blocks..];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    let x_block = &x_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_q4_0_q8_0_neon(QK4_0, w_block, x_block);
                }
            }
        }
    }
}

/// Q5_0 × Q8_0 matrix-vector multiplication
pub unsafe fn matmul_q5_0_q8_0(
    weights: &[BlockQ5_0],
    x: &[BlockQ8_0],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK8_0;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_row = &x[ir1 * ne00_blocks..];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    let x_block = &x_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_q5_0_q8_0_neon(QK8_0, w_block, x_block);
                }
            }
        }
    }
}

/// Q4_1 × Q8_1 matrix-vector multiplication
pub unsafe fn matmul_q4_1_q8_1(
    weights: &[BlockQ4_1],
    x: &[BlockQ8_1],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK4_0;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_row = &x[ir1 * ne00_blocks..];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    let x_block = &x_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_q4_1_q8_1_neon(QK4_0, w_block, x_block);
                }
            }
        }
    }
}

/// Q5_1 × Q8_1 matrix-vector multiplication
pub unsafe fn matmul_q5_1_q8_1(
    weights: &[BlockQ5_1],
    x: &[BlockQ8_1],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK8_0;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_row = &x[ir1 * ne00_blocks..];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    let x_block = &x_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_q5_1_q8_1_neon(QK8_0, w_block, x_block);
                }
            }
        }
    }
}

// ===== IQ series matmul =====

/// IQ4_XS × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq4_xs_q8_k(
    weights: &[BlockIQ4XS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK_K;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_block = &x[ir1..ir1 + 1];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_iq4_xs_q8_k_neon(QK_K, w_block, x_block);
                }
            }
        }
    }
}

/// IQ4_NL × Q8_0 matrix-vector multiplication
pub unsafe fn matmul_iq4_nl_q8_0(
    weights: &[BlockIQ4NL],
    x: &[BlockQ8_0],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK4_0;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_row = &x[ir1 * ne00_blocks..];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    let x_block = &x_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_iq4_nl_q8_0_neon(QK4_0, w_block, x_block);
                }
            }
        }
    }
}

/// IQ3_XXS × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq3_xxs_q8_k(
    weights: &[BlockIQ3XXS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK_K;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_block = &x[ir1..ir1 + 1];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_iq3_xxs_q8_k_neon(QK_K, w_block, x_block);
                }
            }
        }
    }
}

/// IQ3_S × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq3_s_q8_k(
    weights: &[BlockIQ3S],
    x: &[BlockQ8K],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK_K;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_block = &x[ir1..ir1 + 1];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_iq3_s_q8_k_neon(QK_K, w_block, x_block);
                }
            }
        }
    }
}

/// IQ2_XXS × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq2_xxs_q8_k(
    weights: &[BlockIQ2XXS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK_K;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_block = &x[ir1..ir1 + 1];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_iq2_xxs_q8_k_neon(QK_K, w_block, x_block);
                }
            }
        }
    }
}

/// IQ2_XS × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq2_xs_q8_k(
    weights: &[BlockIQ2XS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK_K;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_block = &x[ir1..ir1 + 1];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_iq2_xs_q8_k_neon(QK_K, w_block, x_block);
                }
            }
        }
    }
}

/// IQ2_S × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq2_s_q8_k(
    weights: &[BlockIQ2S],
    x: &[BlockQ8K],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK_K;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_block = &x[ir1..ir1 + 1];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_iq2_s_q8_k_neon(QK_K, w_block, x_block);
                }
            }
        }
    }
}

/// IQ1_S × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq1_s_q8_k(
    weights: &[BlockIQ1S],
    x: &[BlockQ8K],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK_K;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_block = &x[ir1..ir1 + 1];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_iq1_s_q8_k_neon(QK_K, w_block, x_block);
                }
            }
        }
    }
}

/// IQ1_M × Q8_K matrix-vector multiplication
pub unsafe fn matmul_iq1_m_q8_k(
    weights: &[BlockIQ1M],
    x: &[BlockQ8K],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK_K;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_block = &x[ir1..ir1 + 1];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_iq1_m_q8_k_neon(QK_K, w_block, x_block);
                }
            }
        }
    }
}

// ===== TQ series matmul =====

/// TQ2_0 × Q8_K matrix-vector multiplication
pub unsafe fn matmul_tq2_0_q8_k(
    weights: &[BlockTQ2_0],
    x: &[BlockQ8K],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK_K;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_block = &x[ir1..ir1 + 1];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_tq2_0_q8_k_neon(QK_K, w_block, x_block);
                }
            }
        }
    }
}

/// TQ1_0 × Q8_K matrix-vector multiplication
pub unsafe fn matmul_tq1_0_q8_k(
    weights: &[BlockTQ1_0],
    x: &[BlockQ8K],
    dst: &mut [f32],
    ne00: usize,
    ne01: usize,
) {
    const BLCK_0: usize = 16;
    const BLCK_1: usize = 16;

    let ne00_blocks = ne00 / QK_K;

    for iir1 in (0..ne01).step_by(BLCK_1) {
        for iir0 in (0..ne00_blocks).step_by(BLCK_0) {
            for ir1 in iir1..(iir1 + BLCK_1).min(ne01) {
                let weight_row = &weights[ir1 * ne00_blocks..];
                let x_block = &x[ir1..ir1 + 1];

                for ir0 in iir0..(iir0 + BLCK_0).min(ne00_blocks) {
                    let w_block = &weight_row[ir0..ir0 + 1];
                    dst[ir1 * ne00_blocks + ir0] = vec_dot_tq1_0_q8_k_neon(QK_K, w_block, x_block);
                }
            }
        }
    }
}

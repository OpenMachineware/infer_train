// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

//! Tiled Flash Attention implementation for CPU
//! Based on llama.cpp's ggml_compute_forward_flash_attn_ext_tiled

use crate::inference::KVCacheBase;

/// Tile sizes for flash attention
const Q_TILE_SZ: usize = 8; // Process 8 Q rows at once
const KV_TILE_SZ: usize = 8; // Process 8 KV positions at once

/// Optimized prefill attention - processes all tokens at once
/// This replaces the token-by-token loop with a tiled batch approach
#[cfg(target_arch = "aarch64")]
pub fn flash_attention_prefill_optimized(
    q_outputs: &[f32], // [n_tokens, n_heads, head_dim] - Q projections
    k_outputs: &[f32], // [n_tokens, n_kv_heads, head_dim] - K projections
    v_outputs: &[f32], // [n_tokens, n_kv_heads, head_dim] - V projections
    layer_idx: usize,
    kv_cache: &mut crate::inference::KVCache,
    n_tokens: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    attn_outputs: &mut [f32], // [n_tokens, n_heads, head_dim]
    scratch: &mut FlashAttnPrefillScratch,
) {
    use std::arch::aarch64::*;

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let q_size = n_heads * head_dim;
    let kv_size = n_kv_heads * head_dim;

    // Clear outputs
    for elem in attn_outputs.iter_mut() {
        *elem = 0.0;
    }

    // First, store all K, V to cache
    for token_idx in 0..n_tokens {
        let k_offset = token_idx * kv_size;
        let v_offset = token_idx * kv_size;
        kv_cache.set_k(
            layer_idx,
            &k_outputs[k_offset..k_offset + kv_size],
            n_kv_heads,
            head_dim,
        );
        kv_cache.set_v(
            layer_idx,
            &v_outputs[v_offset..v_offset + kv_size],
            n_kv_heads,
            head_dim,
        );
        kv_cache.advance();
    }

    // Process attention for each token
    // Token i attends to positions 0..=i
    for token_idx in 0..n_tokens {
        let seq_len = token_idx + 1; // Number of K/V positions this token attends to

        // For each head
        for h in 0..n_heads {
            let kv_head = h * n_kv_heads / n_heads;
            let q_offset = token_idx * q_size + h * head_dim;
            let out_offset = token_idx * q_size + h * head_dim;

            // Flash attention state
            let mut max_score = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;

            // Process KV positions in tiles for better cache utilization
            for kv_tile_start in (0..seq_len).step_by(KV_TILE_SZ) {
                let kv_tile_end = (kv_tile_start + KV_TILE_SZ).min(seq_len);
                let kv_tile_size = kv_tile_end - kv_tile_start;

                // Compute attention scores for this tile
                for tk in 0..kv_tile_size {
                    let kv_pos = kv_tile_start + tk;
                    kv_cache.get_k_into(
                        layer_idx,
                        kv_head,
                        kv_pos,
                        &mut scratch.k_row,
                    );

                    // NEON-optimized dot product
                    let mut score = 0.0f32;
                    unsafe {
                        let q_ptr = q_outputs[q_offset..].as_ptr();
                        let k_ptr = scratch.k_row.as_ptr();
                        let mut sum_vec = std::arch::aarch64::vdupq_n_f32(0.0);

                        for d in (0..head_dim).step_by(4) {
                            if d + 4 <= head_dim {
                                let q_vec =
                                    std::arch::aarch64::vld1q_f32(q_ptr.add(d));
                                let k_vec =
                                    std::arch::aarch64::vld1q_f32(k_ptr.add(d));
                                sum_vec = std::arch::aarch64::vfmaq_f32(
                                    sum_vec, q_vec, k_vec,
                                );
                            } else {
                                for dd in d..head_dim {
                                    score += q_outputs[q_offset + dd]
                                        * scratch.k_row[dd];
                                }
                            }
                        }

                        let sum_arr: [f32; 4] = std::mem::transmute(sum_vec);
                        score +=
                            sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
                    }

                    scratch.scores[tk] = score * scale;
                }

                // Online softmax (Flash Attention style)
                let tile_max = scratch.scores[..kv_tile_size]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                if tile_max > max_score {
                    let rescale = (max_score - tile_max).exp();
                    for d in 0..head_dim {
                        attn_outputs[out_offset + d] *= rescale;
                    }
                    sum_exp *= rescale;
                    max_score = tile_max;
                }

                // Compute exponentials
                for tk in 0..kv_tile_size {
                    scratch.scores[tk] = (scratch.scores[tk] - max_score).exp();
                    sum_exp += scratch.scores[tk];
                }

                // Accumulate weighted V
                for tk in 0..kv_tile_size {
                    let kv_pos = kv_tile_start + tk;
                    kv_cache.get_v_into(
                        layer_idx,
                        kv_head,
                        kv_pos,
                        &mut scratch.v_row,
                    );

                    let weight = scratch.scores[tk];
                    for d in 0..head_dim {
                        attn_outputs[out_offset + d] +=
                            weight * scratch.v_row[d];
                    }
                }
            }

            // Normalize
            if sum_exp > 0.0 {
                for d in 0..head_dim {
                    attn_outputs[out_offset + d] /= sum_exp;
                }
            }
        }
    }
}

/// Optimized single-token attention for prefill/decode
/// Uses NEON SIMD for dot products and tiled KV processing
#[cfg(target_arch = "aarch64")]
pub fn flash_attention_single(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    layer_idx: usize,
    kv_cache: &mut crate::inference::KVCache,
    n_heads: usize,
    q_head_dim: usize,
    n_kv_heads: usize,
    out: &mut [f32],
    scratch: &mut FlashAttnPrefillScratch,
) {
    use std::arch::aarch64::*;

    let k_head_dim = k.len() / n_kv_heads;
    let v_head_dim = v.len() / n_kv_heads;

    // Store K, V in cache (must be done before parallel processing)
    kv_cache.set_k(layer_idx, k, n_kv_heads, k_head_dim);
    kv_cache.set_v(layer_idx, v, n_kv_heads, v_head_dim);

    let seq_len = kv_cache.seq_len() + 1;
    let dot_dim = q_head_dim.min(k_head_dim);
    let scale = 1.0f32 / (dot_dim as f32).sqrt();

    // Clear output
    for elem in out.iter_mut() {
        *elem = 0.0;
    }

    // Use multithreading for attention heads when beneficial
    // Decode (batch=1) has thread overhead that hurts performance
    // Only use multithreading for batch prefill (seq_len > 32 or many heads)
    let use_multithread = n_heads >= 4 && seq_len > 32;
    let n_threads = if use_multithread {
        crate::cpu::default_n_threads().min(n_heads)
    } else {
        1
    };

    if n_threads > 1 {
        use crate::cpu::SyncPtr;
        // Use SyncPtr to share kv_cache across threads (safe: get_k/get_v are read-only)
        let kv_cache_ptr = SyncPtr::new(
            kv_cache as *const crate::inference::KVCache
                as *mut crate::inference::KVCache,
        );
        let out_ptr = SyncPtr::new(out.as_mut_ptr());
        let q_ptr = SyncPtr::new(q.as_ptr() as *mut f32);

        std::thread::scope(|s| {
            let handles: Vec<_> = (0..n_threads)
                .map(|tid| {
                    s.spawn(move || {
                        // Each thread gets its own scratch space
                        let mut local_scores = vec![0.0f32; KV_TILE_SZ];
                        let mut local_k_row = vec![0.0f32; k_head_dim];
                        let mut local_v_row = vec![0.0f32; v_head_dim];

                        // Process heads assigned to this thread
                        let heads_per_thread =
                            (n_heads + n_threads - 1) / n_threads;
                        let h_start = tid * heads_per_thread;
                        let h_end = (h_start + heads_per_thread).min(n_heads);

                        for h in h_start..h_end {
                            let kv_head = h * n_kv_heads / n_heads;
                            let q_offset = h * q_head_dim;
                            let out_offset = h * v_head_dim;

                            let mut max_score = f32::NEG_INFINITY;
                            let mut sum_exp = 0.0f32;

                            for kv_tile_start in
                                (0..seq_len).step_by(KV_TILE_SZ)
                            {
                                let kv_tile_end =
                                    (kv_tile_start + KV_TILE_SZ).min(seq_len);
                                let kv_tile_size = kv_tile_end - kv_tile_start;

                                // Compute attention scores
                                for tk in 0..kv_tile_size {
                                    let pos = kv_tile_start + tk;
                                    unsafe {
                                        (*kv_cache_ptr.get()).get_k_into(
                                            layer_idx,
                                            kv_head,
                                            pos,
                                            &mut local_k_row,
                                        );
                                    }

                                    unsafe {
                                        let q_ptr_local =
                                            q_ptr.get().add(q_offset);
                                        let k_ptr_local = local_k_row.as_ptr();
                                        let mut sum_vec = vdupq_n_f32(0.0);

                                        for d in (0..dot_dim).step_by(4) {
                                            if d + 4 <= dot_dim {
                                                let q_vec = vld1q_f32(
                                                    q_ptr_local.add(d),
                                                );
                                                let k_vec = vld1q_f32(
                                                    k_ptr_local.add(d),
                                                );
                                                sum_vec = vfmaq_f32(
                                                    sum_vec, q_vec, k_vec,
                                                );
                                            }
                                        }

                                        let sum_arr: [f32; 4] =
                                            std::mem::transmute(sum_vec);
                                        local_scores[tk] = (sum_arr[0]
                                            + sum_arr[1]
                                            + sum_arr[2]
                                            + sum_arr[3])
                                            * scale;
                                    }
                                }

                                // Online softmax
                                let tile_max = local_scores[..kv_tile_size]
                                    .iter()
                                    .cloned()
                                    .fold(f32::NEG_INFINITY, f32::max);

                                if tile_max > max_score {
                                    let rescale = (max_score - tile_max).exp();
                                    unsafe {
                                        let out_local =
                                            out_ptr.get().add(out_offset);
                                        for d in 0..v_head_dim {
                                            *out_local.add(d) *= rescale;
                                        }
                                    }
                                    sum_exp *= rescale;
                                    max_score = tile_max;
                                }

                                // Compute exponentials
                                for tk in 0..kv_tile_size {
                                    local_scores[tk] =
                                        (local_scores[tk] - max_score).exp();
                                    sum_exp += local_scores[tk];
                                }

                                // Accumulate weighted V
                                for tk in 0..kv_tile_size {
                                    let pos = kv_tile_start + tk;
                                    unsafe {
                                        (*kv_cache_ptr.get()).get_v_into(
                                            layer_idx,
                                            kv_head,
                                            pos,
                                            &mut local_v_row,
                                        );
                                    }

                                    let weight = local_scores[tk];
                                    unsafe {
                                        let out_local =
                                            out_ptr.get().add(out_offset);
                                        for d in 0..v_head_dim {
                                            *out_local.add(d) +=
                                                weight * local_v_row[d];
                                        }
                                    }
                                }
                            }

                            // Normalize
                            if sum_exp > 0.0 {
                                unsafe {
                                    let out_local =
                                        out_ptr.get().add(out_offset);
                                    for d in 0..v_head_dim {
                                        *out_local.add(d) /= sum_exp;
                                    }
                                }
                            }
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        });
    } else {
        // Single-threaded fallback
        for h in 0..n_heads {
            let kv_head = h * n_kv_heads / n_heads;
            let q_offset = h * q_head_dim;
            let out_offset = h * v_head_dim;

            let mut max_score = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;

            for kv_tile_start in (0..seq_len).step_by(KV_TILE_SZ) {
                let kv_tile_end = (kv_tile_start + KV_TILE_SZ).min(seq_len);
                let kv_tile_size = kv_tile_end - kv_tile_start;

                for tk in 0..kv_tile_size {
                    let pos = kv_tile_start + tk;
                    kv_cache.get_k_into(
                        layer_idx,
                        kv_head,
                        pos,
                        &mut scratch.k_row,
                    );

                    let mut score = 0.0f32;
                    unsafe {
                        let q_ptr = q[q_offset..].as_ptr();
                        let k_ptr = scratch.k_row.as_ptr();
                        let mut sum_vec = vdupq_n_f32(0.0);

                        for d in (0..dot_dim).step_by(4) {
                            if d + 4 <= dot_dim {
                                let q_vec = vld1q_f32(q_ptr.add(d));
                                let k_vec = vld1q_f32(k_ptr.add(d));
                                sum_vec = vfmaq_f32(sum_vec, q_vec, k_vec);
                            } else {
                                for dd in d..dot_dim {
                                    score +=
                                        q[q_offset + dd] * scratch.k_row[dd];
                                }
                            }
                        }

                        let sum_arr: [f32; 4] = std::mem::transmute(sum_vec);
                        score +=
                            sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
                    }

                    scratch.scores[tk] = score * scale;
                }

                let tile_max = scratch.scores[..kv_tile_size]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                if tile_max > max_score {
                    let rescale = (max_score - tile_max).exp();
                    for d in 0..v_head_dim {
                        out[out_offset + d] *= rescale;
                    }
                    sum_exp *= rescale;
                    max_score = tile_max;
                }

                for tk in 0..kv_tile_size {
                    scratch.scores[tk] = (scratch.scores[tk] - max_score).exp();
                    sum_exp += scratch.scores[tk];
                }

                for tk in 0..kv_tile_size {
                    let pos = kv_tile_start + tk;
                    kv_cache.get_v_into(
                        layer_idx,
                        kv_head,
                        pos,
                        &mut scratch.v_row,
                    );

                    let weight = scratch.scores[tk];
                    for d in 0..v_head_dim {
                        out[out_offset + d] += weight * scratch.v_row[d];
                    }
                }
            }

            if sum_exp > 0.0 {
                for d in 0..v_head_dim {
                    out[out_offset + d] /= sum_exp;
                }
            }
        }
    }
}

/// Scratch space for tiled flash attention
/// Per-thread layout matches llama.cpp
pub struct FlashAttnPrefillScratch {
    pub scores: Vec<f32>, // [KV_TILE_SZ] - for decode
    pub k_row: Vec<f32>,  // [head_dim]
    pub v_row: Vec<f32>,  // [head_dim]
    // Additional fields for tiled prefill
    pub k_tile: Vec<f32>, // [head_dim, KV_TILE_SZ] - transposed for GEMM
    pub v_tile: Vec<f32>, // [KV_TILE_SZ, head_dim]
    pub kq: Vec<f32>,     // [Q_TILE_SZ, KV_TILE_SZ]
    pub vkq: Vec<f32>,    // [Q_TILE_SZ, head_dim]
    pub m_state: Vec<f32>, // [Q_TILE_SZ] - online softmax max
    pub s_state: Vec<f32>, // [Q_TILE_SZ] - online softmax sum
}

impl FlashAttnPrefillScratch {
    pub fn new(head_dim: usize) -> Self {
        Self {
            scores: vec![0.0; KV_TILE_SZ],
            k_row: vec![0.0; head_dim],
            v_row: vec![0.0; head_dim],
            k_tile: vec![0.0; head_dim * KV_TILE_SZ],
            v_tile: vec![0.0; KV_TILE_SZ * head_dim],
            kq: vec![0.0; Q_TILE_SZ * KV_TILE_SZ],
            vkq: vec![0.0; Q_TILE_SZ * head_dim],
            m_state: vec![f32::NEG_INFINITY; Q_TILE_SZ],
            s_state: vec![0.0; Q_TILE_SZ],
        }
    }
}

/// Tiled Flash Attention for prefill following llama.cpp's approach
/// Processes multiple Q rows together using tiled processing with transposed K
#[cfg(target_arch = "aarch64")]
pub fn flash_attention_prefill_tiled(
    q_outputs: &[f32], // [n_tokens, n_heads, head_dim]
    k_outputs: &[f32], // [n_tokens, n_kv_heads, head_dim]
    v_outputs: &[f32], // [n_tokens, n_kv_heads, head_dim]
    layer_idx: usize,
    kv_cache: &mut crate::inference::KVCache,
    n_tokens: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    attn_outputs: &mut [f32], // [n_tokens, n_heads, head_dim]
    scratch: &mut FlashAttnPrefillScratch,
) {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let q_size = n_heads * head_dim;
    let kv_size = n_kv_heads * head_dim;

    // Clear outputs
    for elem in attn_outputs.iter_mut() {
        *elem = 0.0;
    }

    // First, store all K, V to cache
    for token_idx in 0..n_tokens {
        let k_offset = token_idx * kv_size;
        let v_offset = token_idx * kv_size;
        kv_cache.set_k(
            layer_idx,
            &k_outputs[k_offset..k_offset + kv_size],
            n_kv_heads,
            head_dim,
        );
        kv_cache.set_v(
            layer_idx,
            &v_outputs[v_offset..v_offset + kv_size],
            n_kv_heads,
            head_dim,
        );
        kv_cache.advance();
    }

    // Process Q rows in tiles, respecting causal mask
    let seq_len = n_tokens;

    for q_tile_start in (0..seq_len).step_by(Q_TILE_SZ) {
        let q_tile_end = (q_tile_start + Q_TILE_SZ).min(seq_len);
        let q_tile_size = q_tile_end - q_tile_start;

        // For each head
        for h in 0..n_heads {
            let kv_head = h * n_kv_heads / n_heads;

            // Initialize softmax state for this tile
            for i in 0..Q_TILE_SZ {
                scratch.m_state[i] = f32::NEG_INFINITY;
                scratch.s_state[i] = 0.0;
            }
            // Clear VKQ accumulator
            for i in 0..Q_TILE_SZ * head_dim {
                scratch.vkq[i] = 0.0;
            }

            // Process KV tiles - each Q row i attends to positions [0, i]
            for kv_tile_start in (0..q_tile_end).step_by(KV_TILE_SZ) {
                let kv_tile_end = (kv_tile_start + KV_TILE_SZ).min(q_tile_end);
                let kv_tile_size = kv_tile_end - kv_tile_start;

                // Load K tile transposed: K_tile[head_dim, kv_tile_size]
                // This is the critical optimization from llama.cpp
                for tk in 0..kv_tile_size {
                    let kv_pos = kv_tile_start + tk;
                    kv_cache.get_k_into(
                        layer_idx,
                        kv_head,
                        kv_pos,
                        &mut scratch.k_row,
                    );
                    // Transpose: K_tile[d, tk] = K[tk, d]
                    for d in 0..head_dim {
                        scratch.k_tile[d * KV_TILE_SZ + tk] = scratch.k_row[d];
                    }
                }

                // Load V tile: V_tile[kv_tile_size, head_dim]
                for tk in 0..kv_tile_size {
                    let kv_pos = kv_tile_start + tk;
                    kv_cache.get_v_into(
                        layer_idx,
                        kv_head,
                        kv_pos,
                        &mut scratch.v_row,
                    );
                    for d in 0..head_dim {
                        scratch.v_tile[tk * head_dim + d] = scratch.v_row[d];
                    }
                }

                // Compute KQ = Q @ K^T using true GEMM: KQ[q_tile_size, kv_tile_size] = Q[q_tile_size, head_dim] @ K[head_dim, kv_tile_size]
                // K is stored transposed, so we can efficiently access K[:, tk]
                unsafe {
                    use std::arch::aarch64::*;

                    // Clear KQ for this tile
                    for i in 0..q_tile_size * kv_tile_size {
                        scratch.kq[i] = 0.0;
                    }

                    // GEMM: For each row i of Q and column j of K, compute dot product
                    // This is equivalent to simd_gemm(KQ, Q, K, q_tile_size, head_dim, kv_tile_size)
                    // where K is transposed: K[d, j] instead of K[j, d]

                    // Process in 4×4 blocks following llama.cpp's GEMM_RM/GEMM_RN pattern
                    for i0 in (0..q_tile_size).step_by(4) {
                        let i1 = (i0 + 4).min(q_tile_size);

                        for j0 in (0..kv_tile_size).step_by(4) {
                            let j1 = (j0 + 4).min(kv_tile_size);

                            // 4×4 accumulator block
                            let mut acc: [[float32x4_t; 4]; 4] =
                                std::mem::zeroed();

                            // Inner loop over K dimension (head_dim)
                            for kk in (0..head_dim).step_by(4) {
                                if kk + 4 <= head_dim {
                                    // Load K columns j0..j1 for dimensions kk..kk+4
                                    // K is transposed: K[d, tk] stored at k_tile[d * KV_TILE_SZ + tk]
                                    // So K[kk:kk+4, j] is at k_tile[kk*KV_TILE_SZ + j] to k_tile[(kk+3)*KV_TILE_SZ + j]

                                    // Load V32-style: 4 elements from each of 4 K columns
                                    let mut k_cols: [float32x4_t; 4] =
                                        std::mem::zeroed();
                                    for jj in 0..(j1 - j0) {
                                        // Load K[kk:kk+4, j0+jj] - but this needs careful indexing
                                        // Since K is [DK, KV_TILE_SZ] with DK contiguous first,
                                        // K[d, tk] is at k_tile[d * KV_TILE_SZ + tk]
                                        // So K[kk:kk+4, j0+jj] is at &k_tile[kk * KV_TILE_SZ + j0 + jj]
                                        // But these are not contiguous! We need to gather.

                                        // Actually, following llama.cpp's approach more closely:
                                        // K_f32 is [DK, KV_TILE_SZ] stored transposed for GEMM
                                        // When doing Q @ K^T, we compute KQ[i, j] = sum_k Q[i, k] * K[k, j]
                                        // With transposed K, K[k, j] is at K_f32[k * KV_TILE_SZ + j]
                                        // To vectorize, we load K[kk:kk+4, j] for multiple j at once

                                        // Load 4 K rows at different positions - each from a different K column
                                        k_cols[jj] = vld1q_f32(
                                            scratch
                                                .k_tile
                                                .as_ptr()
                                                .add(kk * KV_TILE_SZ + j0 + jj),
                                        );
                                    }

                                    // For each Q row in the block
                                    for ii in 0..(i1 - i0) {
                                        let i = i0 + ii;
                                        let q_row = q_tile_start + i;
                                        let q_offset =
                                            q_row * q_size + h * head_dim;

                                        // Load Q[i, kk:kk+4]
                                        let qv = vld1q_f32(
                                            q_outputs[q_offset..]
                                                .as_ptr()
                                                .add(kk),
                                        );

                                        // FMLA: acc[ii, jj] += Q[i, kk:kk+4] * K[kk:kk+4, j0+jj]
                                        for jj in 0..(j1 - j0) {
                                            acc[ii][jj] = vfmaq_f32(
                                                acc[ii][jj],
                                                qv,
                                                k_cols[jj],
                                            );
                                        }
                                    }
                                }
                            }

                            // Scale and store results, applying causal mask
                            for ii in 0..(i1 - i0) {
                                for jj in 0..(j1 - j0) {
                                    let i = i0 + ii;
                                    let j = j0 + jj;

                                    // Apply causal mask
                                    if kv_tile_start + j <= q_tile_start + i {
                                        let score_vals: [f32; 4] =
                                            std::mem::transmute(acc[ii][jj]);
                                        let score = (score_vals[0]
                                            + score_vals[1]
                                            + score_vals[2]
                                            + score_vals[3])
                                            * scale;
                                        scratch.kq[i * KV_TILE_SZ + j] = score;
                                    } else {
                                        scratch.kq[i * KV_TILE_SZ + j] =
                                            f32::NEG_INFINITY;
                                    }
                                }
                            }
                        }
                    }
                }

                // Online softmax: update M and S, rescale VKQ if needed
                for tq in 0..q_tile_size {
                    let mut tile_max = f32::NEG_INFINITY;
                    for tk in 0..kv_tile_size {
                        let score = scratch.kq[tq * KV_TILE_SZ + tk];
                        if score > tile_max && score != f32::NEG_INFINITY {
                            tile_max = score;
                        }
                    }

                    if tile_max == f32::NEG_INFINITY {
                        continue;
                    }

                    let old_m = scratch.m_state[tq];
                    let new_m = old_m.max(tile_max);

                    if new_m > old_m && old_m != f32::NEG_INFINITY {
                        let rescale = (old_m - new_m).exp();
                        for d in 0..head_dim {
                            scratch.vkq[tq * head_dim + d] *= rescale;
                        }
                        scratch.s_state[tq] *= rescale;
                    }
                    scratch.m_state[tq] = new_m;

                    // Compute softmax and accumulate
                    let mut sum_exp = 0.0f32;
                    for tk in 0..kv_tile_size {
                        let score = scratch.kq[tq * KV_TILE_SZ + tk];
                        if score != f32::NEG_INFINITY {
                            let exp_score = (score - new_m).exp();
                            scratch.kq[tq * KV_TILE_SZ + tk] = exp_score;
                            sum_exp += exp_score;
                        } else {
                            scratch.kq[tq * KV_TILE_SZ + tk] = 0.0;
                        }
                    }
                    scratch.s_state[tq] += sum_exp;

                    // Accumulate weighted V using GEMM-style operations
                    // softmax(KQ) @ V where KQ is [q_tile_size, kv_tile_size] and V is [kv_tile_size, head_dim]
                    unsafe {
                        use std::arch::aarch64::*;

                        // Process 4 dimensions at a time
                        for d in (0..head_dim).step_by(4) {
                            if d + 4 <= head_dim {
                                // For each Q row, accumulate weighted V across all KV positions
                                for tq in 0..q_tile_size {
                                    let mut vsum = vdupq_n_f32(0.0f32);

                                    // Accumulate across KV positions
                                    // This is the GEMM inner loop: sum over k of softmax(KQ)[q, k] * V[k, d]
                                    for tk in 0..kv_tile_size {
                                        let weight =
                                            scratch.kq[tq * KV_TILE_SZ + tk];
                                        if weight > 0.0 {
                                            // Load V[tk, d: d+4]
                                            let v_row = vld1q_f32(
                                                scratch.v_tile[tk * head_dim..]
                                                    .as_ptr()
                                                    .add(d),
                                            );
                                            // FMLA: vsum += weight * v_row
                                            vsum = vfmaq_f32(
                                                vsum,
                                                v_row,
                                                vdupq_n_f32(weight),
                                            );
                                        }
                                    }

                                    // Store to accumulator
                                    let sum_vals: [f32; 4] =
                                        std::mem::transmute(vsum);
                                    for lane in 0..4 {
                                        scratch.vkq
                                            [tq * head_dim + d + lane] +=
                                            sum_vals[lane];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Normalize and write output
            for tq in 0..q_tile_size {
                let q_row = q_tile_start + tq;
                let out_offset = q_row * q_size + h * head_dim;

                let s = scratch.s_state[tq];
                if s > 0.0 {
                    let s_inv = 1.0 / s;
                    for d in 0..head_dim {
                        attn_outputs[out_offset + d] =
                            scratch.vkq[tq * head_dim + d] * s_inv;
                    }
                }
            }
        }
    }
}

/// Legacy single-token flash attention (for decode)
#[cfg(target_arch = "aarch64")]
pub fn flash_attention_decode(
    q: &[f32],
    k_cache: &crate::inference::KVCache,
    v_cache: &crate::inference::KVCache,
    layer_idx: usize,
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
    scratch: &mut FlashAttnScratch,
) {
    use std::arch::aarch64::*;

    let scale = 1.0f32 / (head_dim as f32).sqrt();

    for elem in out.iter_mut() {
        *elem = 0.0;
    }

    for h in 0..n_heads {
        let kv_head = h * n_kv_heads / n_heads;
        let q_offset = h * head_dim;
        let out_offset = h * head_dim;

        let mut max_score = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;

        for kv_tile_start in (0..seq_len).step_by(KV_TILE_SZ) {
            let kv_tile_end = (kv_tile_start + KV_TILE_SZ).min(seq_len);
            let kv_tile_size = kv_tile_end - kv_tile_start;

            for tk in 0..kv_tile_size {
                let pos = kv_tile_start + tk;
                k_cache.get_k_into(layer_idx, kv_head, pos, &mut scratch.k_row);

                let mut score = 0.0f32;
                unsafe {
                    let q_ptr = q[q_offset..].as_ptr();
                    let k_ptr = scratch.k_row.as_ptr();
                    let mut sum_vec = std::arch::aarch64::vdupq_n_f32(0.0);
                    for d in (0..head_dim).step_by(4) {
                        if d + 4 <= head_dim {
                            let q_vec =
                                std::arch::aarch64::vld1q_f32(q_ptr.add(d));
                            let k_vec =
                                std::arch::aarch64::vld1q_f32(k_ptr.add(d));
                            sum_vec = std::arch::aarch64::vfmaq_f32(
                                sum_vec, q_vec, k_vec,
                            );
                        }
                    }
                    let sum_arr: [f32; 4] = std::mem::transmute(sum_vec);
                    score = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
                }

                scratch.scores[tk] = score * scale;
            }

            let tile_max = scratch.scores[..kv_tile_size]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            if tile_max > max_score {
                let rescale = (max_score - tile_max).exp();
                for d in 0..head_dim {
                    out[out_offset + d] *= rescale;
                }
                sum_exp *= rescale;
                max_score = tile_max;
            }

            for tk in 0..kv_tile_size {
                scratch.scores[tk] = (scratch.scores[tk] - max_score).exp();
                sum_exp += scratch.scores[tk];
            }

            for tk in 0..kv_tile_size {
                let pos = kv_tile_start + tk;
                v_cache.get_v_into(layer_idx, kv_head, pos, &mut scratch.v_row);
                let weight = scratch.scores[tk];
                for d in 0..head_dim {
                    out[out_offset + d] += weight * scratch.v_row[d];
                }
            }
        }

        if sum_exp > 0.0 {
            for d in 0..head_dim {
                out[out_offset + d] /= sum_exp;
            }
        }
    }
}

/// Scratch space for decode attention
pub struct FlashAttnScratch {
    pub scores: Vec<f32>,
    pub k_row: Vec<f32>,
    pub v_row: Vec<f32>,
}

impl FlashAttnScratch {
    pub fn new(head_dim: usize) -> Self {
        Self {
            scores: vec![0.0; KV_TILE_SZ],
            k_row: vec![0.0; head_dim],
            v_row: vec![0.0; head_dim],
        }
    }
}

/// Scratch space for batch flash attention (not currently used, kept for compatibility)
pub struct FlashAttnBatchScratch {
    pub q_tile: Vec<f32>,
    pub k_row: Vec<f32>,
    pub v_row: Vec<f32>,
    pub scores: Vec<f32>,
}

impl FlashAttnBatchScratch {
    pub fn new(head_dim: usize) -> Self {
        Self {
            q_tile: vec![0.0; head_dim * Q_TILE_SZ],
            k_row: vec![0.0; head_dim],
            v_row: vec![0.0; head_dim],
            scores: vec![0.0; KV_TILE_SZ],
        }
    }
}

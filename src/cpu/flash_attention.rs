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

/// Scratch space for prefill attention
pub struct FlashAttnPrefillScratch {
    pub scores: Vec<f32>, // [KV_TILE_SZ]
    pub k_row: Vec<f32>,  // [head_dim]
    pub v_row: Vec<f32>,  // [head_dim]
}

impl FlashAttnPrefillScratch {
    pub fn new(head_dim: usize) -> Self {
        Self {
            scores: vec![0.0; KV_TILE_SZ],
            k_row: vec![0.0; head_dim],
            v_row: vec![0.0; head_dim],
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

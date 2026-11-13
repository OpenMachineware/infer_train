// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

//! Tiled Flash Attention implementation for CPU
//! Based on llama.cpp's ggml_compute_forward_flash_attn_ext_tiled

use crate::inference::KVCacheBase;

/// Tile sizes for flash attention
/// Match llama.cpp's tile sizes for optimal cache utilization
const Q_TILE_SZ: usize = 64; // Process 64 Q rows at once (was 8)
const KV_TILE_SZ: usize = 64; // Process 64 KV positions at once (was 8)

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
    // Decode (single query) should ALWAYS use single-thread to avoid thread overhead
    // For decode, the query length is 1 (n_q = 1), which is the key indicator
    // Only use multithreading for batch prefill where we have many queries
    // Since this function is for single query (flash_attention_single), always use single-thread
    // The prefill path uses a different function (flash_attention_prefill_tiled)
    let _use_multithread = false;
    let n_threads = 1;

    if n_threads > 1 {
        // PROBLEM: Creating new Vec per thread causes cache thrashing
        // This should use pre-allocated scratch like the single-thread version
        use crate::cpu::SyncPtr;
        // Use SyncPtr to share kv_cache across threads (safe: get_k/get_v are read-only)
        let kv_cache_ptr = SyncPtr::new(
            kv_cache as *const crate::inference::KVCache
                as *mut crate::inference::KVCache,
        );
        let out_ptr = SyncPtr::new(out.as_mut_ptr());
        let q_ptr = SyncPtr::new(q.as_ptr() as *mut f32);

        let pool = crate::cpu::global_pool(n_threads);
        use std::sync::atomic::Ordering;

        pool.parallel(move |tid| {
            // Each thread gets its own scratch space
            let mut local_scores = vec![0.0f32; KV_TILE_SZ];
            let mut local_k_row = vec![0.0f32; k_head_dim];
            let mut local_v_row = vec![0.0f32; v_head_dim];

            // Work stealing: threads grab heads dynamically
            let mut h = tid; // Start with thread ID

            while h < n_heads {
                let kv_head = h * n_kv_heads / n_heads;
                let q_offset = h * q_head_dim;
                let out_offset = h * v_head_dim;

                let mut max_score = f32::NEG_INFINITY;
                let mut sum_exp = 0.0f32;

                for kv_tile_start in (0..seq_len).step_by(KV_TILE_SZ) {
                    let kv_tile_end = (kv_tile_start + KV_TILE_SZ).min(seq_len);
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
                            let q_ptr_local = q_ptr.get().add(q_offset);
                            let k_ptr_local = local_k_row.as_ptr();
                            let mut sum_vec = vdupq_n_f32(0.0);

                            for d in (0..dot_dim).step_by(4) {
                                if d + 4 <= dot_dim {
                                    let q_vec = vld1q_f32(q_ptr_local.add(d));
                                    let k_vec = vld1q_f32(k_ptr_local.add(d));
                                    sum_vec = vfmaq_f32(sum_vec, q_vec, k_vec);
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
                            let out_local = out_ptr.get().add(out_offset);
                            for d in 0..v_head_dim {
                                *out_local.add(d) *= rescale;
                            }
                        }
                        sum_exp *= rescale;
                        max_score = tile_max;
                    }

                    // Compute exponentials
                    for tk in 0..kv_tile_size {
                        local_scores[tk] = (local_scores[tk] - max_score).exp();
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
                            let out_local = out_ptr.get().add(out_offset);
                            for d in 0..v_head_dim {
                                *out_local.add(d) += weight * local_v_row[d];
                            }
                        }
                    }
                }

                // Normalize
                if sum_exp > 0.0 {
                    unsafe {
                        let out_local = out_ptr.get().add(out_offset);
                        for d in 0..v_head_dim {
                            *out_local.add(d) /= sum_exp;
                        }
                    }
                }

                // Work stealing: get next head
                h = pool.state().current_chunk.fetch_add(1, Ordering::Relaxed);
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
    pub q_tile: Vec<f32>, // [Q_TILE_SZ, head_dim] - Q tile for GEMM
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
            q_tile: vec![0.0; Q_TILE_SZ * head_dim],
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
/// Multi-threaded: distributes heads across threads for parallel processing
///
/// Key optimization: Reads K/V directly from input tensors, bypassing KV cache
/// This matches llama.cpp's approach for prefill performance
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
    scratch: &mut [FlashAttnPrefillScratch], // Per-thread scratch space
) {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let q_size = n_heads * head_dim;
    let kv_size = n_kv_heads * head_dim;

    // Clear outputs
    for elem in attn_outputs.iter_mut() {
        *elem = 0.0;
    }

    // Store K, V to cache for decode (but don't advance - that's done at a higher level)
    // This follows llama.cpp: prefill reads from input tensors, not cache
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
    }

    // Use multithreading for prefill (n_heads >= 4 or n_tokens >= 8)
    let n_threads = if n_heads >= 4 || n_tokens >= 8 {
        crate::cpu::default_n_threads().min(n_heads * n_tokens)
    } else {
        1
    };

    // Total number of Q rows = n_tokens * n_heads
    // Distribute these rows across threads (llama.cpp's approach)
    let total_q_rows = n_tokens * n_heads;

    if n_threads > 1 {
        // Multi-threaded: distribute Q rows across threads with work stealing
        // Use std::thread::scope to avoid barrier deadlock
        use crate::cpu::SyncPtr;
        use std::sync::atomic::AtomicUsize;

        let out_ptr = SyncPtr::new(attn_outputs.as_mut_ptr());
        let q_ptr = SyncPtr::new(q_outputs.as_ptr() as *mut f32);
        let k_ptr = SyncPtr::new(k_outputs.as_ptr() as *mut f32);
        let v_ptr = SyncPtr::new(v_outputs.as_ptr() as *mut f32);
        let scratch_ptr = SyncPtr::new(scratch.as_mut_ptr());

        let current_chunk = AtomicUsize::new(n_threads);
        use std::sync::atomic::Ordering;

        std::thread::scope(|s| {
            for tid in 0..n_threads {
                let current_chunk = &current_chunk;

                s.spawn(move || {
                    // Each thread gets its own pre-allocated scratch space
                    let local_scratch =
                        unsafe { &mut *scratch_ptr.get().add(tid) };

                    // Work stealing: threads grab rows dynamically
                    let mut ir = tid; // Start with thread ID

                    while ir < total_q_rows {
                        // Convert linear index to (token_idx, head_idx)
                        let token_idx = ir / n_heads;
                        let h = ir % n_heads;
                        let kv_head = h * n_kv_heads / n_heads;

                        // Number of rows in this tile
                        let tile_rows = Q_TILE_SZ
                            .min(n_heads - h)
                            .min(n_tokens - token_idx);

                        process_q_tile(
                            q_ptr.get(),
                            k_ptr.get(),
                            v_ptr.get(),
                            n_kv_heads,
                            head_dim,
                            kv_head,
                            h,
                            token_idx,
                            tile_rows,
                            n_tokens,
                            q_size,
                            kv_size,
                            scale,
                            out_ptr.get(),
                            local_scratch,
                        );

                        // Work stealing: get next row
                        ir = current_chunk.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        });
    } else {
        // Single-threaded fallback
        for token_idx in 0..n_tokens {
            for h in 0..n_heads {
                let kv_head = h * n_kv_heads / n_heads;

                process_q_tile(
                    q_outputs.as_ptr(),
                    k_outputs.as_ptr(),
                    v_outputs.as_ptr(),
                    n_kv_heads,
                    head_dim,
                    kv_head,
                    h,
                    token_idx,
                    1, // Single row in single-threaded mode
                    n_tokens,
                    q_size,
                    kv_size,
                    scale,
                    attn_outputs.as_mut_ptr(),
                    &mut scratch[0], // Use first element for single-threaded
                );
            }
        }
    }
}

/// Process a tile of Q rows (following llama.cpp's tiled approach)
/// Multiple Q rows can be processed together for cache efficiency
#[cfg(target_arch = "aarch64")]
fn process_q_tile(
    q_outputs: *const f32,
    k_outputs: *const f32,
    v_outputs: *const f32,
    _n_kv_heads: usize,
    head_dim: usize,
    kv_head: usize,
    h: usize,         // Starting head index
    token_idx: usize, // Starting token index
    tile_rows: usize, // Number of Q rows in this tile
    seq_len: usize,   // Total sequence length
    q_size: usize,
    kv_size: usize,
    scale: f32,
    attn_outputs: *mut f32,
    scratch: &mut FlashAttnPrefillScratch,
) {
    use std::arch::aarch64::*;

    unsafe {
        let q_outputs = std::slice::from_raw_parts(q_outputs, q_size * seq_len);
        let k_outputs =
            std::slice::from_raw_parts(k_outputs, kv_size * seq_len);
        let v_outputs =
            std::slice::from_raw_parts(v_outputs, kv_size * seq_len);
        let attn_outputs =
            std::slice::from_raw_parts_mut(attn_outputs, q_size * seq_len);

        // Initialize softmax state for this tile
        for i in 0..Q_TILE_SZ {
            scratch.m_state[i] = f32::NEG_INFINITY;
            scratch.s_state[i] = 0.0;
        }
        // Clear VKQ accumulator
        for i in 0..Q_TILE_SZ * head_dim {
            scratch.vkq[i] = 0.0;
        }

        // Each Q row at position (token_idx + i) attends to positions [0, token_idx + i]
        let max_kv_pos = token_idx + tile_rows;

        // Process KV tiles
        for kv_tile_start in (0..max_kv_pos).step_by(KV_TILE_SZ) {
            let kv_tile_end = (kv_tile_start + KV_TILE_SZ).min(max_kv_pos);
            let kv_tile_size = kv_tile_end - kv_tile_start;

            // Load K tile transposed: K_tile[head_dim, kv_tile_size]
            for tk in 0..kv_tile_size {
                let kv_pos = kv_tile_start + tk;
                let k_offset = kv_pos * kv_size + kv_head * head_dim;

                // Transpose: K_tile[d, tk] = K[tk, d]
                for d in 0..head_dim {
                    scratch.k_tile[d * KV_TILE_SZ + tk] =
                        k_outputs[k_offset + d];
                }
            }

            // Load V tile: V_tile[kv_tile_size, head_dim]
            for tk in 0..kv_tile_size {
                let kv_pos = kv_tile_start + tk;
                let v_offset = kv_pos * kv_size + kv_head * head_dim;

                for d in 0..head_dim {
                    scratch.v_tile[tk * head_dim + d] = v_outputs[v_offset + d];
                }
            }

            // Compute KQ = Q @ K^T using optimized SIMD GEMM
            // K is stored transposed: K_tile[head_dim, kv_tile_size]

            // Clear KQ for this tile
            scratch.kq[..tile_rows * kv_tile_size].fill(0.0);

            // Load Q tile for this head range using memcpy-equivalent
            for tq in 0..tile_rows {
                let q_row = token_idx + tq;
                let q_offset = q_row * q_size + h * head_dim;
                scratch.q_tile[tq * head_dim..(tq + 1) * head_dim]
                    .copy_from_slice(&q_outputs[q_offset..q_offset + head_dim]);
            }

            // GEMM: KQ[tile_rows, kv_tile_size] = Q[tile_rows, head_dim] @ K_tile[head_dim, kv_tile_size]
            crate::quant::gemm::kernel::sgemm(
                scratch.kq.as_mut_ptr(),
                scratch.q_tile.as_ptr(),
                scratch.k_tile.as_ptr(),
                tile_rows,
                head_dim,
                kv_tile_size,
            );

            // Apply scale and causal mask
            for i in 0..tile_rows {
                let q_pos = token_idx + i;
                for j in 0..kv_tile_size {
                    let kv_pos = kv_tile_start + j;
                    if kv_pos <= q_pos {
                        scratch.kq[i * KV_TILE_SZ + j] *= scale;
                    } else {
                        scratch.kq[i * KV_TILE_SZ + j] = f32::NEG_INFINITY;
                    }
                }
            }

            // Online softmax: update M and S, rescale VKQ if needed
            for tq in 0..tile_rows {
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

                // Accumulate weighted V
                for d in (0..head_dim).step_by(4) {
                    if d + 4 <= head_dim {
                        let mut vsum = vdupq_n_f32(0.0f32);

                        for tk in 0..kv_tile_size {
                            let weight = scratch.kq[tq * KV_TILE_SZ + tk];
                            if weight > 0.0 {
                                let v_row = vld1q_f32(
                                    scratch.v_tile[tk * head_dim..]
                                        .as_ptr()
                                        .add(d),
                                );
                                vsum =
                                    vfmaq_f32(vsum, v_row, vdupq_n_f32(weight));
                            }
                        }

                        // Store to accumulator
                        let sum_vals: [f32; 4] = std::mem::transmute(vsum);
                        for lane in 0..4 {
                            scratch.vkq[tq * head_dim + d + lane] +=
                                sum_vals[lane];
                        }
                    }
                }
            }
        }

        // Normalize and write output
        for tq in 0..tile_rows {
            let q_row = token_idx + tq;
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

                let score;
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

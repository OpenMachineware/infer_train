// Flash Attention for Metal
// Based on llama.cpp's fa.metal template structure
// Initially supports F16, will extend to quantized formats

#include <metal_stdlib>
using namespace metal;

// Constants
constant short N_SIMDWIDTH = 32;

// Simple Flash Attention for F16 (decode case: batch=1)
// Computes: softmax(Q * K^T / sqrt(d)) * V
// For decode, Q is [1, head_dim], K/V are [seq_len, head_dim]
kernel void kernel_flash_attn_f16_decode(
    device const half * q,
    device const half * k,
    device const half * v,
    device const half * mask,
    device       half * dst,
    constant uint & head_dim,
    constant uint & seq_len,
    constant float & scale,
    uint tid [[thread_index_in_simdgroup]]) {

    // Each SIMD group processes one query for all KV positions
    // tid is the thread index within the SIMD group (0-31)

    // Thread 0 maintains softmax state
    float max_val = -INFINITY;
    float sum_exp = 0.0;

    // Calculate how many dimensions each thread handles
    // For head_dim=128, SIMD width=32: 128/32=4 dimensions per thread
    constexpr uint MAX_DIMS_PER_THREAD = 8;  // Support up to 256 head_dim
    uint dims_per_thread = (head_dim + N_SIMDWIDTH - 1) / N_SIMDWIDTH;
    uint effective_dims = min(dims_per_thread, MAX_DIMS_PER_THREAD);

    // Output accumulators - one per dimension
    float output[MAX_DIMS_PER_THREAD] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Process each KV position sequentially
    for (uint s = 0; s < seq_len; ++s) {
        // Compute Q dot K[s]
        // All threads cooperate on the dot product
        float score = 0.0;
        for (uint i = 0; i < effective_dims; ++i) {
            uint d = tid + i * N_SIMDWIDTH;
            if (d < head_dim) {
                float q_val = float(q[d]);
                float k_val = float(k[s * head_dim + d]);
                score += q_val * k_val;
            }
        }

        // Reduce across SIMD group - all threads now have same score
        score = simd_sum(score);
        score *= scale;

        // Apply mask if present
        if (mask != nullptr) {
            float mask_val = float(mask[s]);
            if (mask_val == -INFINITY) {
                score = -INFINITY;
            }
        }

        // Online softmax update - only thread 0 manages state
        float new_max = max_val;
        if (tid == 0) {
            new_max = max(max_val, score);
        }
        // Broadcast new max to all threads
        new_max = simd_broadcast(new_max, 0);

        float weight = 0.0;
        if (score > -INFINITY) {
            // Compute correction factor for previous outputs
            // Skip correction on first iteration (max_val is -inf)
            if (new_max > max_val && max_val > -INFINITY) {
                // Rescale previous outputs and sum_exp
                float correction = exp(max_val - new_max);
                if (tid == 0) {
                    sum_exp *= correction;
                }
                for (uint i = 0; i < effective_dims; ++i) {
                    output[i] *= correction;
                }
            }

            // Compute weight for this position
            weight = exp(score - new_max);
            if (tid == 0) {
                sum_exp += weight;
            }
        }

        // Accumulate weighted V - each dimension separately
        if (weight > 0.0) {
            for (uint i = 0; i < effective_dims; ++i) {
                uint d = tid + i * N_SIMDWIDTH;
                if (d < head_dim) {
                    float v_val = float(v[s * head_dim + d]);
                    output[i] += weight * v_val;
                }
            }
        }

        // Update max for next iteration
        max_val = new_max;
    }

    // Broadcast sum_exp from thread 0 to all threads
    sum_exp = simd_broadcast(sum_exp, 0);

    // Normalize and write output - each dimension gets its own value
    for (uint i = 0; i < effective_dims; ++i) {
        uint d = tid + i * N_SIMDWIDTH;
        if (d < head_dim) {
            dst[d] = half(output[i] / sum_exp);
        }
    }
}

// Batch Flash Attention for F16
// Q: [batch, head_dim], K/V: [batch, seq_len, head_dim] (batch-independent KV)
// Each threadgroup handles one batch item
kernel void kernel_flash_attn_f16_batch(
    device const half * q,
    device const half * k,
    device const half * v,
    device const half * mask,
    device       half * dst,
    constant uint & batch_size,
    constant uint & head_dim,
    constant uint & seq_len,
    constant float & scale,
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_simdgroup]]) {

    // tgpig.x = batch index
    const uint batch_idx = tgpig.x;

    // Pointers for this batch
    device const half * q_batch = q + batch_idx * head_dim;
    device const half * k_batch = k;  // KV is shared across batch
    device const half * v_batch = v;
    device const half * mask_batch = mask ? mask + batch_idx * seq_len : nullptr;
    device half * dst_batch = dst + batch_idx * head_dim;

    float max_val = -INFINITY;
    float sum_exp = 0.0;

    // Calculate how many dimensions each thread handles
    constexpr uint MAX_DIMS_PER_THREAD = 8;
    uint dims_per_thread = (head_dim + N_SIMDWIDTH - 1) / N_SIMDWIDTH;
    uint effective_dims = min(dims_per_thread, MAX_DIMS_PER_THREAD);
    float output[MAX_DIMS_PER_THREAD] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (uint s = 0; s < seq_len; ++s) {
        float score = 0.0;

        for (uint i = 0; i < effective_dims; ++i) {
            uint d = tid + i * N_SIMDWIDTH;
            if (d < head_dim) {
                float q_val = float(q_batch[d]);
                float k_val = float(k_batch[s * head_dim + d]);
                score += q_val * k_val;
            }
        }

        score = simd_sum(score);
        score *= scale;

        if (mask_batch != nullptr) {
            float mask_val = float(mask_batch[s]);
            if (mask_val == -INFINITY) {
                score = -INFINITY;
            }
        }

        float new_max = max_val;
        if (tid == 0) {
            new_max = max(max_val, score);
        }
        new_max = simd_broadcast(new_max, 0);

        float weight = 0.0;
        if (score > -INFINITY) {
            if (new_max > max_val && max_val > -INFINITY) {
                float correction = exp(max_val - new_max);
                if (tid == 0) {
                    sum_exp *= correction;
                }
                for (uint i = 0; i < effective_dims; ++i) {
                    output[i] *= correction;
                }
            }

            weight = exp(score - new_max);
            if (tid == 0) {
                sum_exp += weight;
            }
        }

        if (weight > 0.0) {
            for (uint i = 0; i < effective_dims; ++i) {
                uint d = tid + i * N_SIMDWIDTH;
                if (d < head_dim) {
                    float v_val = float(v_batch[s * head_dim + d]);
                    output[i] += weight * v_val;
                }
            }
        }

        max_val = new_max;
    }

    sum_exp = simd_broadcast(sum_exp, 0);

    for (uint i = 0; i < effective_dims; ++i) {
        uint d = tid + i * N_SIMDWIDTH;
        if (d < head_dim) {
            dst_batch[d] = half(output[i] / sum_exp);
        }
    }
}

// Multi-head Flash Attention for F16
// Q: [batch, n_heads, head_dim]
// K: [seq_len, n_kv_heads, head_dim] or [seq_len, n_kv_heads, head_dim] for GQA
// V: same as K
// Output: [batch, n_heads, head_dim]
kernel void kernel_flash_attn_f16_multihead(
    device const half * q,
    device const half * k,
    device const half * v,
    device const half * mask,
    device       half * dst,
    constant uint & batch_size,
    constant uint & n_heads,
    constant uint & n_kv_heads,
    constant uint & head_dim,
    constant uint & seq_len,
    constant float & scale,
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_simdgroup]]) {

    // tgpig.x = batch index, tgpig.y = head index
    const uint batch_idx = tgpig.x;
    const uint head_idx = tgpig.y;

    // GQA: multiple query heads share one KV head
    const uint kv_head_idx = head_idx * n_kv_heads / n_heads;

    // Pointers
    device const half * q_head = q + (batch_idx * n_heads + head_idx) * head_dim;
    device const half * k_head = k + kv_head_idx * head_dim;
    device const half * v_head = v + kv_head_idx * head_dim;
    device const half * mask_head = mask ? mask + batch_idx * seq_len : nullptr;
    device half * dst_head = dst + (batch_idx * n_heads + head_idx) * head_dim;

    float max_val = -INFINITY;
    float sum_exp = 0.0;

    // Calculate how many dimensions each thread handles
    constexpr uint MAX_DIMS_PER_THREAD = 8;
    uint dims_per_thread = (head_dim + N_SIMDWIDTH - 1) / N_SIMDWIDTH;
    uint effective_dims = min(dims_per_thread, MAX_DIMS_PER_THREAD);
    float output[MAX_DIMS_PER_THREAD] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (uint s = 0; s < seq_len; ++s) {
        float score = 0.0;

        for (uint i = 0; i < effective_dims; ++i) {
            uint d = tid + i * N_SIMDWIDTH;
            if (d < head_dim) {
                float q_val = float(q_head[d]);
                float k_val = float(k_head[s * n_kv_heads * head_dim + d]);
                score += q_val * k_val;
            }
        }

        score = simd_sum(score);
        score *= scale;

        if (mask_head != nullptr) {
            float mask_val = float(mask_head[s]);
            if (mask_val == -INFINITY) {
                score = -INFINITY;
            }
        }

        float new_max = max_val;
        if (tid == 0) {
            new_max = max(max_val, score);
        }
        new_max = simd_broadcast(new_max, 0);

        float weight = 0.0;
        if (score > -INFINITY) {
            if (new_max > max_val && max_val > -INFINITY) {
                float correction = exp(max_val - new_max);
                if (tid == 0) {
                    sum_exp *= correction;
                }
                for (uint i = 0; i < effective_dims; ++i) {
                    output[i] *= correction;
                }
            }

            weight = exp(score - new_max);
            if (tid == 0) {
                sum_exp += weight;
            }
        }

        if (weight > 0.0) {
            for (uint i = 0; i < effective_dims; ++i) {
                uint d = tid + i * N_SIMDWIDTH;
                if (d < head_dim) {
                    float v_val = float(v_head[s * n_kv_heads * head_dim + d]);
                    output[i] += weight * v_val;
                }
            }
        }

        max_val = new_max;
    }

    sum_exp = simd_broadcast(sum_exp, 0);

    for (uint i = 0; i < effective_dims; ++i) {
        uint d = tid + i * N_SIMDWIDTH;
        if (d < head_dim) {
            dst_head[d] = half(output[i] / sum_exp);
        }
    }
}

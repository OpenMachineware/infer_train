#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/math/matmul.hpp"
#include "infer_train/math/slice.hpp"
#include "infer_train/math/reshape.hpp"
#include "infer_train/nn/softmax.hpp"
#include <cmath>
#include <vector>

namespace infer_train {

// ============================================================
// Causal Mask（因果掩码，下三角）
// ============================================================
template<typename T>
Tensor<T> causal_mask(size_t seq_len) {
    using Conv = DTypeConverter<T>;
    Tensor<T> mask({seq_len, seq_len});
    float neg_inf = -1e9f;
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            float val = (j > i) ? neg_inf : 0.0f;
            mask.data[i * seq_len + j] = Conv::from_float(val);
        }
    }
    return mask;
}

// ============================================================
// Scaled Dot Product Attention
// ============================================================
template<typename T>
Tensor<T> scaled_dot_product_attention(
    const Tensor<T>& query,
    const Tensor<T>& key,
    const Tensor<T>& value,
    const Tensor<T>* mask = nullptr,
    float scale = -1.0f,
    bool is_causal = false,
    float dropout_p = 0.0f
) {
    // Remove warning
    (void)dropout_p;

    if (query.shape.size() != 3 || key.shape.size() != 3 || value.shape.size() != 3) {
        return Tensor<T>();
    }

    if (query.shape[2] != key.shape[2] || key.shape[1] != value.shape[1]) {
        return Tensor<T>();
    }

    using Conv = DTypeConverter<T>;

    size_t batch = query.shape[0];
    size_t seq_len_q = query.shape[1];
    size_t seq_len_k = key.shape[1];
    size_t head_dim = query.shape[2];

    Tensor<T> output({batch, seq_len_q, head_dim});

    // 对每个 batch 单独处理
    for (size_t b = 0; b < batch; ++b) {
        // 提取当前 batch 的 Q/K/V
        // 用 slice 沿第0维切
        Tensor<T> q_b = slice(query, 0, b, b + 1);  // (1, seq_len_q, head_dim)
        Tensor<T> k_b = slice(key, 0, b, b + 1);    // (1, seq_len_k, head_dim)
        Tensor<T> v_b = slice(value, 0, b, b + 1);  // (1, seq_len_v, head_dim)

        // squeeze 掉 batch 维度 (1 -> 0)
        // 但 slice 后 shape 是 (1, seq_len, head_dim)，matmul 需要 2D
        // 所以直接 reshape 成 2D
        // 或者让 matmul 支持 3D？先 reshape
        // 简单处理：复制数据到 2D Tensor
        Tensor<T> q_2d({seq_len_q, head_dim});
        Tensor<T> k_2d({seq_len_k, head_dim});
        Tensor<T> v_2d({seq_len_k, head_dim});

        for (size_t i = 0; i < seq_len_q; ++i) {
            for (size_t j = 0; j < head_dim; ++j) {
                q_2d.data[i * head_dim + j] = q_b.data[i * head_dim + j];
            }
        }
        for (size_t i = 0; i < seq_len_k; ++i) {
            for (size_t j = 0; j < head_dim; ++j) {
                k_2d.data[i * head_dim + j] = k_b.data[i * head_dim + j];
                v_2d.data[i * head_dim + j] = v_b.data[i * head_dim + j];
            }
        }

        // 转置 key: (seq_len_k, head_dim) -> (head_dim, seq_len_k)
        Tensor<T> k_t = transpose(k_2d);

        // matmul: (seq_len_q, head_dim) @ (head_dim, seq_len_k)
        Tensor<T> attn_scores = matmul(q_2d, k_t);

        // Scale
        float scale_val = (scale > 0.0f) ? scale : (1.0f / std::sqrt((float)head_dim));
        for (size_t i = 0; i < attn_scores.size(); ++i) {
            attn_scores.data[i] = Conv::from_float(
                Conv::to_float(attn_scores.data[i]) * scale_val
            );
        }

        // Mask (causal)
        if (is_causal) {
            Tensor<T> causal = causal_mask<T>(seq_len_q);
            for (size_t i = 0; i < attn_scores.size(); ++i) {
                attn_scores.data[i] = Conv::from_float(
                    Conv::to_float(attn_scores.data[i]) +
                    Conv::to_float(causal.data[i])
                );
            }
        } else if (mask != nullptr) {
            // 暂不支持 mask
        }

        // Softmax
        Tensor<T> attn_weights = softmax(attn_scores, 1);

        // attn_weights @ value: (seq_len_q, seq_len_k) @ (seq_len_k, head_dim)
        Tensor<T> out_2d = matmul(attn_weights, v_2d);

        // 写回 output
        for (size_t i = 0; i < seq_len_q; ++i) {
            for (size_t j = 0; j < head_dim; ++j) {
                output.data[b * seq_len_q * head_dim + i * head_dim + j] =
                    out_2d.data[i * head_dim + j];
            }
        }
    }

    return output;
}

// ============================================================
// Multi-Head Attention
// ============================================================
template<typename T>
Tensor<T> multi_head_attention(
    const Tensor<T>& query,
    const Tensor<T>& key,
    const Tensor<T>& value,
    const Tensor<T>* mask = nullptr,
    int num_heads = 8,
    float scale = -1.0f,
    bool is_causal = false,
    float dropout_p = 0.0f
) {
    // Remove warning
    (void)dropout_p;

    // 输入: (batch, seq_len, embed_dim)
    if (query.shape.size() != 3 || key.shape.size() != 3 || value.shape.size() != 3) {
        return Tensor<T>();
    }

    size_t batch = query.shape[0];
    size_t seq_len_q = query.shape[1];
    // size_t seq_len_k = key.shape[1];
    size_t embed_dim = query.shape[2];

    if (embed_dim % num_heads != 0) {
        return Tensor<T>();
    }

    // size_t head_dim = embed_dim / num_heads;

    // 如果单头，直接调用
    if (num_heads == 1) {
        return scaled_dot_product_attention(query, key, value, mask, scale, is_causal, dropout_p);
    }

    // 多头：reshape + 循环处理
    // 将 (batch, seq_len, embed_dim) 变成 (batch, num_heads, seq_len, head_dim)
    // 然后对每个 head 调用 scaled_dot_product_attention
    // 最后拼接

    Tensor<T> output({batch, seq_len_q, embed_dim});

    for (size_t b = 0; b < batch; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            // 提取当前 batch 和 head
            // 用 slice + reshape
            // 调用 scaled_dot_product_attention（现在支持 batch=1）
            // 写回 output
        }
    }

    return output;
}

} // namespace infer_train

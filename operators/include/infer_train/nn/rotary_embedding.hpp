#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/math/slice.hpp"
#include "infer_train/math/cat.hpp"
#include <cmath>
#include <vector>

namespace infer_train {

// ============================================================
// rotary_embedding: 旋转位置编码
// ============================================================
template<typename T>
Tensor<T> rotary_embedding(
    const Tensor<T>& x,
    const Tensor<T>& cos,
    const Tensor<T>& sin
) {
    // x: (batch, seq_len, head_dim)
    // cos, sin: (seq_len, head_dim/2)

    if (x.shape.size() != 3 || cos.shape.size() != 2 || sin.shape.size() != 2) {
        return Tensor<T>();
    }

    size_t batch = x.shape[0];
    size_t seq_len = x.shape[1];
    size_t head_dim = x.shape[2];

    if (head_dim % 2 != 0) return Tensor<T>();

    if (cos.shape[0] != seq_len || cos.shape[1] != head_dim / 2) {
        return Tensor<T>();
    }
    if (sin.shape[0] != seq_len || sin.shape[1] != head_dim / 2) {
        return Tensor<T>();
    }

    using Conv = DTypeConverter<T>;

    // 1. 将 x 分成两半
    Tensor<T> x1 = slice(x, 2, 0, head_dim / 2);  // (batch, seq_len, head_dim/2)
    Tensor<T> x2 = slice(x, 2, head_dim / 2, head_dim);

    Tensor<T> result(x.shape);

    // 2. 逐元素旋转
    // cos: (seq_len, head_dim/2) -> 广播到 (batch, seq_len, head_dim/2)
    // sin: (seq_len, head_dim/2) -> 广播到 (batch, seq_len, head_dim/2)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t h = 0; h < head_dim / 2; ++h) {
                size_t idx1 = b * seq_len * (head_dim / 2) + s * (head_dim / 2) + h;
                size_t idx2 = b * seq_len * (head_dim / 2) + s * (head_dim / 2) + h;
                size_t cos_idx = s * (head_dim / 2) + h;
                size_t sin_idx = s * (head_dim / 2) + h;

                float x1_val = Conv::to_float(x1.data[idx1]);
                float x2_val = Conv::to_float(x2.data[idx2]);
                float cos_val = Conv::to_float(cos.data[cos_idx]);
                float sin_val = Conv::to_float(sin.data[sin_idx]);

                // y1 = x1 * cos - x2 * sin
                float y1 = x1_val * cos_val - x2_val * sin_val;
                // y2 = x1 * sin + x2 * cos
                float y2 = x1_val * sin_val + x2_val * cos_val;

                // 写入结果
                size_t out1 = b * seq_len * head_dim + s * head_dim + h;
                size_t out2 = b * seq_len * head_dim + s * head_dim + h + head_dim / 2;
                result.data[out1] = Conv::from_float(y1);
                result.data[out2] = Conv::from_float(y2);
            }
        }
    }

    return result;
}

// ============================================================
// 方式1：输出参数（内部使用）
// ============================================================
template<typename T>
void precompute_rotary_embeddings_impl(
    size_t seq_len,
    size_t head_dim,
    float base,
    Tensor<T>& cos,
    Tensor<T>& sin
) {
    using Conv = DTypeConverter<T>;
    cos = Tensor<T>({seq_len, head_dim / 2});
    sin = Tensor<T>({seq_len, head_dim / 2});

    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < head_dim / 2; ++j) {
            float angle = static_cast<float>(i) / std::pow(base, 2.0f * j / head_dim);
            cos.data[i * (head_dim / 2) + j] = Conv::from_float(std::cos(angle));
            sin.data[i * (head_dim / 2) + j] = Conv::from_float(std::sin(angle));
        }
    }
}

// ============================================================
// 方式2：返回 pair（对外使用）
// ============================================================
template<typename T>
std::pair<Tensor<T>, Tensor<T>> precompute_rotary_embeddings(
    size_t seq_len,
    size_t head_dim,
    float base = 10000.0f
) {
    Tensor<T> cos, sin;
    precompute_rotary_embeddings_impl<T>(seq_len, head_dim, base, cos, sin);
    return {cos, sin};
}

} // namespace infer_train

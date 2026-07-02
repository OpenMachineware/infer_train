#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cmath>

namespace infer_train {

// ============================================================
// RMSNorm: Root Mean Square Normalization
// ============================================================
template<typename T>
Tensor<T> rmsnorm(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    float eps = 1e-6f
) {
    if (input.shape.empty()) return Tensor<T>();

    using Conv = DTypeConverter<T>;

    size_t last_dim = input.shape[input.shape.size() - 1];
    size_t batch = input.size() / last_dim;

    Tensor<T> result(input.shape);

    for (size_t b = 0; b < batch; ++b) {
        // 计算 RMS
        float sq_sum = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            float x = Conv::to_float(input.data[b * last_dim + i]);
            sq_sum += x * x;
        }
        float rms = std::sqrt(sq_sum / static_cast<float>(last_dim) + eps);
        float inv_rms = 1.0f / rms;

        // 归一化 + 权重
        for (size_t i = 0; i < last_dim; ++i) {
            float x = Conv::to_float(input.data[b * last_dim + i]);
            float w = Conv::to_float(weight.data[i]);
            float y = (x * inv_rms) * w;
            result.data[b * last_dim + i] = Conv::from_float(y);
        }
    }

    return result;
}

} // namespace infer_train

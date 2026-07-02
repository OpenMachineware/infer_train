#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cmath>
#include <vector>

namespace infer_train {

template<typename T>
Tensor<T> layernorm(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    const Tensor<T>& bias,
    float eps = 1e-5f
) {
    if (input.shape.empty()) {
        return Tensor<T>();
    }

    using Conv = DTypeConverter<T>;

    size_t last_dim = input.shape[input.shape.size() - 1];
    size_t batch = input.size() / last_dim;

    Tensor<T> result(input.shape);

    for (size_t b = 0; b < batch; ++b) {
        // 计算 mean
        float sum = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            float val = Conv::to_float(input.data[b * last_dim + i]);
            sum += val;
        }
        float mean = sum / static_cast<float>(last_dim);

        // 计算 var
        float sq_sum = 0.0f;
        for (size_t i = 0; i < last_dim; ++i) {
            float val = Conv::to_float(input.data[b * last_dim + i]);
            float diff = val - mean;
            sq_sum += diff * diff;
        }
        float var = sq_sum / static_cast<float>(last_dim);
        float inv_std = 1.0f / std::sqrt(var + eps);

        // 归一化 + affine
        for (size_t i = 0; i < last_dim; ++i) {
            float x = Conv::to_float(input.data[b * last_dim + i]);
            float gamma = Conv::to_float(weight.data[i]);
            float beta = Conv::to_float(bias.data[i]);
            float y = (x - mean) * inv_std * gamma + beta;
            result.data[b * last_dim + i] = Conv::from_float(y);
        }
    }

    return result;
}

} // namespace infer_train

#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cmath>

namespace infer_train {

// ============================================================
// BatchNorm2d（推理模式，已融合）
// ============================================================
template<typename T>
Tensor<T> batchnorm2d_inference(
    const Tensor<T>& input,
    const Tensor<T>& weight,      // gamma
    const Tensor<T>& bias,        // beta
    const Tensor<T>& running_mean,
    const Tensor<T>& running_var,
    float eps = 1e-5f
) {
    if (input.shape.size() != 4) {
        return Tensor<T>();
    }

    using Conv = DTypeConverter<T>;

    size_t N = input.shape[0];
    size_t C = input.shape[1];
    size_t H = input.shape[2];
    size_t W = input.shape[3];

    Tensor<T> output(input.shape);

    // 对每个通道
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            float mean = Conv::to_float(running_mean.data[c]);
            float var = Conv::to_float(running_var.data[c]);
            float gamma = Conv::to_float(weight.data[c]);
            float beta = Conv::to_float(bias.data[c]);

            float inv_std = 1.0f / std::sqrt(var + eps);
            float scale = gamma * inv_std;
            float shift = beta - gamma * mean * inv_std;

            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    float x = Conv::to_float(
                        input.data[n * C * H * W +
                                   c * H * W +
                                   h * W +
                                   w]
                    );
                    float y = x * scale + shift;
                    output.data[n * C * H * W +
                                c * H * W +
                                h * W +
                                w] = Conv::from_float(y);
                }
            }
        }
    }

    return output;
}

// ============================================================
// BatchNorm2d（训练模式）
// ============================================================
template<typename T>
Tensor<T> batchnorm2d_training(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    const Tensor<T>& bias,
    Tensor<T>& running_mean,
    Tensor<T>& running_var,
    float eps = 1e-5f,
    float momentum = 0.1f
) {
    (void)momentum;
    // 训练模式下需要计算均值和方差
    // 比较复杂，先实现推理模式
    return batchnorm2d_inference(input, weight, bias, running_mean, running_var, eps);
}

// ============================================================
// BatchNorm1d（推理模式）
// ============================================================
template<typename T>
Tensor<T> batchnorm1d_inference(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    const Tensor<T>& bias,
    const Tensor<T>& running_mean,
    const Tensor<T>& running_var,
    float eps = 1e-5f
) {
    if (input.shape.size() != 3) return Tensor<T>();

    using Conv = DTypeConverter<T>;

    size_t N = input.shape[0];
    size_t C = input.shape[1];
    size_t L = input.shape[2];

    Tensor<T> output(input.shape);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            float mean = Conv::to_float(running_mean.data[c]);
            float var = Conv::to_float(running_var.data[c]);
            float gamma = Conv::to_float(weight.data[c]);
            float beta = Conv::to_float(bias.data[c]);

            float inv_std = 1.0f / std::sqrt(var + eps);
            float scale = gamma * inv_std;
            float shift = beta - gamma * mean * inv_std;

            for (size_t l = 0; l < L; ++l) {
                float x = Conv::to_float(
                    input.data[n * C * L + c * L + l]
                );
                output.data[n * C * L + c * L + l] = Conv::from_float(x * scale + shift);
            }
        }
    }
    return output;
}

} // namespace infer_train

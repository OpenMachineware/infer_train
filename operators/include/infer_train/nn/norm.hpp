#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cmath>

namespace infer_train {

// ============================================================
// InstanceNorm2d
// ============================================================
template<typename T>
Tensor<T> instancenorm2d(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    const Tensor<T>& bias,
    float eps = 1e-5f
) {
    if (input.shape.size() != 4) return Tensor<T>();

    using Conv = DTypeConverter<T>;

    size_t N = input.shape[0];
    size_t C = input.shape[1];
    size_t H = input.shape[2];
    size_t W = input.shape[3];

    Tensor<T> output(input.shape);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            // 计算当前 (n, c) 的均值和方差
            float mean = 0.0f, sq_sum = 0.0f;
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    float x = Conv::to_float(
                        input.data[n * C * H * W + c * H * W + h * W + w]
                    );
                    mean += x;
                    sq_sum += x * x;
                }
            }
            float size = static_cast<float>(H * W);
            mean /= size;
            float var = sq_sum / size - mean * mean;
            float inv_std = 1.0f / std::sqrt(var + eps);
            float gamma = Conv::to_float(weight.data[c]);
            float beta = Conv::to_float(bias.data[c]);

            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    float x = Conv::to_float(
                        input.data[n * C * H * W + c * H * W + h * W + w]
                    );
                    float y = (x - mean) * inv_std * gamma + beta;
                    output.data[n * C * H * W + c * H * W + h * W + w] = Conv::from_float(y);
                }
            }
        }
    }
    return output;
}

// ============================================================
// GroupNorm
// ============================================================
template<typename T>
Tensor<T> groupnorm(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    const Tensor<T>& bias,
    int num_groups,
    float eps = 1e-5f
) {
    if (input.shape.size() != 4) return Tensor<T>();

    using Conv = DTypeConverter<T>;

    size_t N = input.shape[0];
    size_t C = input.shape[1];
    size_t H = input.shape[2];
    size_t W = input.shape[3];

    if (C % num_groups != 0) return Tensor<T>();

    size_t group_size = C / num_groups;
    Tensor<T> output(input.shape);

    for (size_t n = 0; n < N; ++n) {
        for (int g = 0; g < num_groups; ++g) {
            // 计算当前 (n, g) 的均值和方差
            float mean = 0.0f, sq_sum = 0.0f;
            size_t count = 0;
            for (size_t c = g * group_size; c < (g + 1) * group_size; ++c) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        float x = Conv::to_float(
                            input.data[n * C * H * W + c * H * W + h * W + w]
                        );
                        mean += x;
                        sq_sum += x * x;
                        count++;
                    }
                }
            }
            float size = static_cast<float>(count);
            mean /= size;
            float var = sq_sum / size - mean * mean;
            float inv_std = 1.0f / std::sqrt(var + eps);

            for (size_t c = g * group_size; c < (g + 1) * group_size; ++c) {
                float gamma = Conv::to_float(weight.data[c]);
                float beta = Conv::to_float(bias.data[c]);
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        float x = Conv::to_float(
                            input.data[n * C * H * W + c * H * W + h * W + w]
                        );
                        float y = (x - mean) * inv_std * gamma + beta;
                        output.data[n * C * H * W + c * H * W + h * W + w] = Conv::from_float(y);
                    }
                }
            }
        }
    }
    return output;
}

} // namespace infer_train

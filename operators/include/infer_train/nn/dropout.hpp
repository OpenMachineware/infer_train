#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <random>
#include <vector>

namespace infer_train {

// ============================================================
// Dropout（推理模式：直接返回输入）
// ============================================================
template<typename T>
Tensor<T> dropout_inference(const Tensor<T>& input, float /* p */ = 0.5f) {
    // 推理时 dropout 不做任何事
    return input;
}

// ============================================================
// Dropout（训练模式：随机掩码）
// ============================================================
template<typename T>
Tensor<T> dropout_training(
    const Tensor<T>& input,
    float p,
    uint32_t seed = 0
) {
    using Conv = DTypeConverter<T>;

    if (p <= 0.0f) {
        return input;
    }
    if (p >= 1.0f) {
        // 全部置零
        Tensor<T> output(input.shape);
        for (size_t i = 0; i < output.size(); ++i) {
            output.data[i] = Conv::from_float(0.0f);
        }
        return output;
    }

    // 随机数生成器
    static std::mt19937 rng(seed != 0 ? seed : std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // 创建掩码并应用到输出
    Tensor<T> output(input.shape);
    float scale = 1.0f / (1.0f - p);

    for (size_t i = 0; i < output.size(); ++i) {
        if (dist(rng) < p) {
            // 丢弃
            output.data[i] = Conv::from_float(0.0f);
        } else {
            // 保留并缩放
            float x = Conv::to_float(input.data[i]);
            output.data[i] = Conv::from_float(x * scale);
        }
    }

    return output;
}

} // namespace infer_train

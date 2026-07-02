#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <random>

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
// Dropout（训练模式：需要随机掩码，暂不实现）
// ============================================================
template<typename T>
Tensor<T> dropout_training(
    const Tensor<T>& input,
    float p,
    uint32_t seed = 0
) {
    // 训练模式需要生成随机掩码
    // 实现较复杂，先留空
    return Tensor<T>();
}

} // namespace infer_train

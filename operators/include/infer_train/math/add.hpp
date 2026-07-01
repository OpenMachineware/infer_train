#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"

namespace infer_train {

// ============================================================
// 浮点加法（模板统一处理 FP32/FP64/FP16/BF16）
// ============================================================
template<typename T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
    static_assert(!is_quantized<T>::value,
                  "Use quantized_add for quantized types");

    if (a.shape != b.shape) {
        return Tensor<T>();
    }

    using Conv = DTypeConverter<T>;
    Tensor<T> result(a.shape);

    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        result.data[i] = Conv::from_float(fa + fb);
    }

    return result;
}

// ============================================================
// 浮点标量加法
// ============================================================
template<typename T>
Tensor<T> add_scalar(const Tensor<T>& a, typename T::storage scalar) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(a.shape);
    float fs = Conv::to_float(scalar);

    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        result.data[i] = Conv::from_float(fa + fs);
    }

    return result;
}

// ============================================================
// 批量加法（多个张量相加）
// ============================================================
template<typename T>
Tensor<T> add_n(const std::vector<const Tensor<T>*>& tensors) {
    if (tensors.empty()) {
        return Tensor<T>();
    }

    const Tensor<T>& first = *tensors[0];
    for (const auto* t : tensors) {
        if (t->shape != first.shape) {
            return Tensor<T>();
        }
    }

    using Conv = DTypeConverter<T>;
    Tensor<T> result(first.shape);

    // 初始化为 0，累加所有张量
    for (size_t i = 0; i < result.size(); ++i) {
        float sum = 0.0f;
        for (const auto* t : tensors) {
            sum += Conv::to_float(t->data[i]);
        }
        result.data[i] = Conv::from_float(sum);
    }

    return result;
}

} // namespace infer_train

#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <algorithm>
#include <cmath>

namespace infer_train {

// ============================================================
// 浮点 ReLU（模板统一处理）
// ============================================================
template<typename T>
Tensor<T> relu(const Tensor<T>& input) {
    static_assert(!is_quantized<T>::value,
                  "Use quantized_relu for quantized types");

    using Conv = DTypeConverter<T>;
    Tensor<T> output(input.shape);

    for (size_t i = 0; i < output.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        output.data[i] = Conv::from_float(std::max(0.0f, x));
    }

    return output;
}

// ============================================================
// 浮点 Leaky ReLU
// ============================================================
template<typename T>
Tensor<T> leaky_relu(const Tensor<T>& input, float alpha = 0.01f) {
    static_assert(!is_quantized<T>::value,
                  "Use quantized_leaky_relu for quantized types");

    using Conv = DTypeConverter<T>;
    Tensor<T> output(input.shape);

    for (size_t i = 0; i < output.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        float out = (x > 0.0f) ? x : alpha * x;
        output.data[i] = Conv::from_float(out);
    }

    return output;
}

// ============================================================
// 浮点 ELU
// ============================================================
template<typename T>
Tensor<T> elu(const Tensor<T>& input, float alpha = 1.0f) {
    static_assert(!is_quantized<T>::value,
                  "Use quantized_elu for quantized types");

    using Conv = DTypeConverter<T>;
    Tensor<T> output(input.shape);

    for (size_t i = 0; i < output.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        float out = (x > 0.0f) ? x : alpha * (std::exp(x) - 1.0f);
        output.data[i] = Conv::from_float(out);
    }

    return output;
}

// ============================================================
// 浮点 GELU（Tanh 近似）
// ============================================================
template<typename T>
Tensor<T> gelu(const Tensor<T>& input) {
    static_assert(!is_quantized<T>::value,
                  "Use quantized_gelu for quantized types");

    using Conv = DTypeConverter<T>;
    Tensor<T> output(input.shape);

    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    for (size_t i = 0; i < output.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        float x3 = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + coeff * x3);
        float tanh_val = std::tanh(tanh_arg);
        float out = 0.5f * x * (1.0f + tanh_val);
        output.data[i] = Conv::from_float(out);
    }

    return output;
}

} // namespace infer_train

#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cmath>

namespace infer_train {

template<typename T>
Tensor<T> sqrt(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);

    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(std::sqrt(std::max(0.0f, x)));
    }

    return result;
}

template<typename T>
Tensor<T> rsqrt(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);

    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(1.0f / std::sqrt(std::max(0.0f, x)));
    }

    return result;
}

} // namespace infer_train

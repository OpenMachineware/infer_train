#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cmath>

namespace infer_train {

template<typename T>
Tensor<T> floor(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(std::floor(x));
    }
    return result;
}

template<typename T>
Tensor<T> ceil(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(std::ceil(x));
    }
    return result;
}

template<typename T>
Tensor<T> round(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(std::nearbyint(x));
    }
    return result;
}

} // namespace infer_train

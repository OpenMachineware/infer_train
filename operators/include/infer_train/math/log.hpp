#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cmath>

namespace infer_train {

template<typename T>
Tensor<T> log(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(std::log(std::max(1e-7f, x)));
    }
    return result;
}

template<typename T>
Tensor<T> log2(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(std::log2(std::max(1e-7f, x)));
    }
    return result;
}

template<typename T>
Tensor<T> log10(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(std::log10(std::max(1e-7f, x)));
    }
    return result;
}

} // namespace infer_train

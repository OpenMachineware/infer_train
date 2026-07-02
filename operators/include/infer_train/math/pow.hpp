#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cmath>

namespace infer_train {

template<typename T>
Tensor<T> pow(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) {
        return Tensor<T>();
    }

    using Conv = DTypeConverter<T>;
    Tensor<T> result(a.shape);

    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        result.data[i] = Conv::from_float(std::pow(fa, fb));
    }

    return result;
}

template<typename T>
Tensor<T> pow_scalar(const Tensor<T>& a, typename T::storage exponent) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(a.shape);
    float fe = Conv::to_float(exponent);

    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        result.data[i] = Conv::from_float(std::pow(fa, fe));
    }

    return result;
}

} // namespace infer_train

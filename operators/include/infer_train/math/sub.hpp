#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"

namespace infer_train {

template<typename T>
Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) {
        return Tensor<T>();
    }

    using Conv = DTypeConverter<T>;
    Tensor<T> result(a.shape);

    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        result.data[i] = Conv::from_float(fa - fb);
    }

    return result;
}

template<typename T>
Tensor<T> sub_scalar(const Tensor<T>& a, typename T::storage scalar) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(a.shape);
    float fs = Conv::to_float(scalar);

    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        result.data[i] = Conv::from_float(fa - fs);
    }

    return result;
}

template<typename T>
Tensor<T> scalar_sub(typename T::storage scalar, const Tensor<T>& a) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(a.shape);
    float fs = Conv::to_float(scalar);

    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        result.data[i] = Conv::from_float(fs - fa);
    }

    return result;
}

} // namespace infer_train

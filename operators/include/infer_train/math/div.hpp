#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cmath>

namespace infer_train {

template<typename T>
Tensor<T> div(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) {
        return Tensor<T>();
    }

    using Conv = DTypeConverter<T>;
    Tensor<T> result(a.shape);

    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        if (std::abs(fb) < 1e-7f) {
            result.data[i] = Conv::from_float(0.0f);
        } else {
            result.data[i] = Conv::from_float(fa / fb);
        }
    }

    return result;
}

template<typename T>
Tensor<T> div_scalar(const Tensor<T>& a, typename T::storage scalar) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(a.shape);
    float fs = Conv::to_float(scalar);

    if (std::abs(fs) < 1e-7f) {
        return Tensor<T>();
    }

    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        result.data[i] = Conv::from_float(fa / fs);
    }

    return result;
}

template<typename T>
Tensor<T> scalar_div(typename T::storage scalar, const Tensor<T>& a) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(a.shape);
    float fs = Conv::to_float(scalar);

    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        if (std::abs(fa) < 1e-7f) {
            result.data[i] = Conv::from_float(0.0f);
        } else {
            result.data[i] = Conv::from_float(fs / fa);
        }
    }

    return result;
}

} // namespace infer_train

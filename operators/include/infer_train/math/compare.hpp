#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"

namespace infer_train {

template<typename T>
Tensor<uint8_t> eq(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) return Tensor<uint8_t>();
    using Conv = DTypeConverter<T>;
    Tensor<uint8_t> result(a.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        result.data[i] = (std::abs(fa - fb) < 1e-6f) ? 1 : 0;
    }
    return result;
}

template<typename T>
Tensor<uint8_t> ne(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) return Tensor<uint8_t>();
    using Conv = DTypeConverter<T>;
    Tensor<uint8_t> result(a.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        result.data[i] = (std::abs(fa - fb) >= 1e-6f) ? 1 : 0;
    }
    return result;
}

template<typename T>
Tensor<uint8_t> gt(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) return Tensor<uint8_t>();
    using Conv = DTypeConverter<T>;
    Tensor<uint8_t> result(a.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        result.data[i] = (fa > fb) ? 1 : 0;
    }
    return result;
}

template<typename T>
Tensor<uint8_t> lt(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) return Tensor<uint8_t>();
    using Conv = DTypeConverter<T>;
    Tensor<uint8_t> result(a.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        result.data[i] = (fa < fb) ? 1 : 0;
    }
    return result;
}

template<typename T>
Tensor<uint8_t> ge(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) return Tensor<uint8_t>();
    using Conv = DTypeConverter<T>;
    Tensor<uint8_t> result(a.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        result.data[i] = (fa >= fb) ? 1 : 0;
    }
    return result;
}

template<typename T>
Tensor<uint8_t> le(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) return Tensor<uint8_t>();
    using Conv = DTypeConverter<T>;
    Tensor<uint8_t> result(a.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        result.data[i] = (fa <= fb) ? 1 : 0;
    }
    return result;
}

} // namespace infer_train

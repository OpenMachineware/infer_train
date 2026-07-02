#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <vector>
#include <cmath>

namespace infer_train {

template<typename T>
std::vector<uint8_t> eq(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) return {};
    using Conv = DTypeConverter<T>;
    std::vector<uint8_t> result(a.size());
    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        result[i] = (std::abs(fa - fb) < 1e-6f) ? 1 : 0;
    }
    return result;
}

template<typename T>
std::vector<uint8_t> ne(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) return {};
    using Conv = DTypeConverter<T>;
    std::vector<uint8_t> result(a.size());
    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        result[i] = (std::abs(fa - fb) >= 1e-6f) ? 1 : 0;
    }
    return result;
}

template<typename T>
std::vector<uint8_t> gt(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) return {};
    using Conv = DTypeConverter<T>;
    std::vector<uint8_t> result(a.size());
    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        result[i] = (fa > fb) ? 1 : 0;
    }
    return result;
}

template<typename T>
std::vector<uint8_t> lt(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) return {};
    using Conv = DTypeConverter<T>;
    std::vector<uint8_t> result(a.size());
    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        result[i] = (fa < fb) ? 1 : 0;
    }
    return result;
}

template<typename T>
std::vector<uint8_t> ge(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) return {};
    using Conv = DTypeConverter<T>;
    std::vector<uint8_t> result(a.size());
    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        result[i] = (fa >= fb) ? 1 : 0;
    }
    return result;
}

template<typename T>
std::vector<uint8_t> le(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape != b.shape) return {};
    using Conv = DTypeConverter<T>;
    std::vector<uint8_t> result(a.size());
    for (size_t i = 0; i < result.size(); ++i) {
        float fa = Conv::to_float(a.data[i]);
        float fb = Conv::to_float(b.data[i]);
        result[i] = (fa <= fb) ? 1 : 0;
    }
    return result;
}

// ============================================================
// I8 专用版本（直接比较存储值）
// ============================================================
inline std::vector<uint8_t> eq(const Tensor<I8>& a, const Tensor<I8>& b) {
    if (a.shape != b.shape) return {};
    std::vector<uint8_t> result(a.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = (a.data[i] == b.data[i]) ? 1 : 0;
    }
    return result;
}

inline std::vector<uint8_t> ne(const Tensor<I8>& a, const Tensor<I8>& b) {
    if (a.shape != b.shape) return {};
    std::vector<uint8_t> result(a.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = (a.data[i] != b.data[i]) ? 1 : 0;
    }
    return result;
}

inline std::vector<uint8_t> gt(const Tensor<I8>& a, const Tensor<I8>& b) {
    if (a.shape != b.shape) return {};
    std::vector<uint8_t> result(a.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = (a.data[i] > b.data[i]) ? 1 : 0;
    }
    return result;
}

inline std::vector<uint8_t> lt(const Tensor<I8>& a, const Tensor<I8>& b) {
    if (a.shape != b.shape) return {};
    std::vector<uint8_t> result(a.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = (a.data[i] < b.data[i]) ? 1 : 0;
    }
    return result;
}

inline std::vector<uint8_t> ge(const Tensor<I8>& a, const Tensor<I8>& b) {
    if (a.shape != b.shape) return {};
    std::vector<uint8_t> result(a.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = (a.data[i] >= b.data[i]) ? 1 : 0;
    }
    return result;
}

inline std::vector<uint8_t> le(const Tensor<I8>& a, const Tensor<I8>& b) {
    if (a.shape != b.shape) return {};
    std::vector<uint8_t> result(a.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = (a.data[i] <= b.data[i]) ? 1 : 0;
    }
    return result;
}

} // namespace infer_train

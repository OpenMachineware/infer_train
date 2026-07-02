#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace infer_train {

// ============================================================
// 内部工具：计算步长
// ============================================================
static inline std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// ============================================================
// sum
// ============================================================
template<typename T>
Tensor<T> sum(const Tensor<T>& input, const std::vector<int>& dims, bool keepdim = false) {
    using Conv = DTypeConverter<T>;
    
    // 简化：只支持全规约
    if (dims.empty()) {
        // 全规约
        float total = 0.0f;
        for (size_t i = 0; i < input.size(); ++i) {
            total += Conv::to_float(input.data[i]);
        }
        std::vector<size_t> shape;
        if (keepdim) {
            shape = input.shape;
            for (auto& d : shape) d = 1;
        }
        Tensor<T> result(shape);
        result.data[0] = Conv::from_float(total);
        return result;
    }
    
    // 简化：直接返回空
    return Tensor<T>();
}

// ============================================================
// mean
// ============================================================
template<typename T>
Tensor<T> mean(const Tensor<T>& input, const std::vector<int>& dims, bool keepdim = false) {
    using Conv = DTypeConverter<T>;
    
    if (dims.empty()) {
        float total = 0.0f;
        for (size_t i = 0; i < input.size(); ++i) {
            total += Conv::to_float(input.data[i]);
        }
        float avg = total / input.size();
        std::vector<size_t> shape;
        if (keepdim) {
            shape = input.shape;
            for (auto& d : shape) d = 1;
        }
        Tensor<T> result(shape);
        result.data[0] = Conv::from_float(avg);
        return result;
    }
    
    return Tensor<T>();
}

// ============================================================
// max (全规约)
// ============================================================
template<typename T>
Tensor<T> max_all(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    
    float max_val = Conv::to_float(input.data[0]);
    for (size_t i = 1; i < input.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        if (x > max_val) max_val = x;
    }
    
    Tensor<T> result({1});
    result.data[0] = Conv::from_float(max_val);
    return result;
}

// ============================================================
// min (全规约)
// ============================================================
template<typename T>
Tensor<T> min_all(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    
    float min_val = Conv::to_float(input.data[0]);
    for (size_t i = 1; i < input.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        if (x < min_val) min_val = x;
    }
    
    Tensor<T> result({1});
    result.data[0] = Conv::from_float(min_val);
    return result;
}

// ============================================================
// prod (全规约)
// ============================================================
template<typename T>
Tensor<T> prod_all(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    float prod = 1.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        prod *= Conv::to_float(input.data[i]);
    }
    Tensor<T> result({1});
    result.data[0] = Conv::from_float(prod);
    return result;
}

} // namespace infer_train

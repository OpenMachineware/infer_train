#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <algorithm>
#include <vector>

namespace infer_train {

// ============================================================
// argmax (沿指定维度)
// ============================================================
template<typename T>
Tensor<int64_t> argmax(const Tensor<T>& input, int dim = -1) {
    // 简单版：全规约
    using Conv = DTypeConverter<T>;
    float max_val = Conv::to_float(input.data[0]);
    size_t max_idx = 0;
    for (size_t i = 1; i < input.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        if (x > max_val) {
            max_val = x;
            max_idx = i;
        }
    }
    Tensor<int64_t> result({1});
    result.data[0] = static_cast<int64_t>(max_idx);
    return result;
}

// ============================================================
// argmin (沿指定维度)
// ============================================================
template<typename T>
Tensor<int64_t> argmin(const Tensor<T>& input, int dim = -1) {
    using Conv = DTypeConverter<T>;
    float min_val = Conv::to_float(input.data[0]);
    size_t min_idx = 0;
    for (size_t i = 1; i < input.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        if (x < min_val) {
            min_val = x;
            min_idx = i;
        }
    }
    Tensor<int64_t> result({1});
    result.data[0] = static_cast<int64_t>(min_idx);
    return result;
}

} // namespace infer_train

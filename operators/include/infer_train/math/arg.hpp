#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace infer_train {

// ============================================================
// argmax（全规约版，沿最后一维）
// ============================================================
template<typename T>
std::vector<int64_t> argmax(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    int64_t max_idx = 0;
    float max_val = Conv::to_float(input.data[0]);
    for (size_t i = 1; i < input.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        if (x > max_val) {
            max_val = x;
            max_idx = static_cast<int64_t>(i);
        }
    }
    return std::vector<int64_t>{max_idx};
}

template<typename T>
std::vector<int64_t> argmin(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    int64_t min_idx = 0;
    float min_val = Conv::to_float(input.data[0]);
    for (size_t i = 1; i < input.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        if (x < min_val) {
            min_val = x;
            min_idx = static_cast<int64_t>(i);
        }
    }
    return std::vector<int64_t>{min_idx};
}

// ============================================================
// I8 专用版本（直接比较存储值）
// ============================================================
inline std::vector<int64_t> argmax(const Tensor<I8>& input) {
    int64_t max_idx = 0;
    int8_t max_val = input.data[0];
    for (size_t i = 1; i < input.size(); ++i) {
        if (input.data[i] > max_val) {
            max_val = input.data[i];
            max_idx = static_cast<int64_t>(i);
        }
    }
    return std::vector<int64_t>{max_idx};
}

inline std::vector<int64_t> argmin(const Tensor<I8>& input) {
    int64_t min_idx = 0;
    int8_t min_val = input.data[0];
    for (size_t i = 1; i < input.size(); ++i) {
        if (input.data[i] < min_val) {
            min_val = input.data[i];
            min_idx = static_cast<int64_t>(i);
        }
    }
    return std::vector<int64_t>{min_idx};
}

} // namespace infer_train

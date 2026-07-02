#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cmath>
#include <vector>

namespace infer_train {

template<typename T>
Tensor<T> var(const Tensor<T>& input, bool unbiased = false) {
    using Conv = DTypeConverter<T>;

    // 计算均值
    float mean = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        mean += Conv::to_float(input.data[i]);
    }
    mean /= static_cast<float>(input.size());

    // 计算方差
    float sq_sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        float diff = x - mean;
        sq_sum += diff * diff;
    }

    float result_val = sq_sum / static_cast<float>(
        unbiased ? input.size() - 1 : input.size()
    );

    Tensor<T> result({1});
    result.data[0] = Conv::from_float(result_val);
    return result;
}

template<typename T>
Tensor<T> std(const Tensor<T>& input, bool unbiased = false) {
    Tensor<T> v = var(input, unbiased);
    using Conv = DTypeConverter<T>;
    Tensor<T> result({1});
    result.data[0] = Conv::from_float(
        std::sqrt(Conv::to_float(v.data[0]))
    );
    return result;
}

} // namespace infer_train

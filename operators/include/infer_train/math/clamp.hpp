#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <algorithm>

namespace infer_train {

template<typename T>
Tensor<T> clamp(
    const Tensor<T>& input,
    typename T::storage min_val,
    typename T::storage max_val
) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    float fmin = Conv::to_float(min_val);
    float fmax = Conv::to_float(max_val);
    
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(std::clamp(x, fmin, fmax));
    }
    
    return result;
}

template<typename T>
Tensor<T> clamp_min(const Tensor<T>& input, typename T::storage min_val) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    float fmin = Conv::to_float(min_val);
    
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(std::max(x, fmin));
    }
    
    return result;
}

template<typename T>
Tensor<T> clamp_max(const Tensor<T>& input, typename T::storage max_val) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    float fmax = Conv::to_float(max_val);
    
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(std::min(x, fmax));
    }
    
    return result;
}

} // namespace infer_train

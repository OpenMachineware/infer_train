#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <vector>

namespace infer_train {

// ============================================================
// where: condition ? true_val : false_val
// condition: 张量（bool 或 0/1）
// ============================================================
template<typename T>
Tensor<T> where(
    const std::vector<uint8_t>& condition,
    const std::vector<size_t>& condition_shape,
    const Tensor<T>& true_val,
    const Tensor<T>& false_val
) {
    if (condition_shape != true_val.shape || true_val.shape != false_val.shape) {
        return Tensor<T>();
    }
    if (condition.size() != true_val.size()) {
        return Tensor<T>();
    }

    Tensor<T> output(true_val.shape);

    for (size_t i = 0; i < output.size(); ++i) {
        if (condition[i] != 0) {
            output.data[i] = true_val.data[i];
        } else {
            output.data[i] = false_val.data[i];
        }
    }

    return output;
}

// 标量版本
template<typename T>
Tensor<T> where_scalar(
    const std::vector<uint8_t>& condition,
    const std::vector<size_t>& condition_shape,
    typename T::storage true_val,
    typename T::storage false_val
) {
    if (condition_shape.empty()) return Tensor<T>();

    using Conv = DTypeConverter<T>;
    Tensor<T> output(condition_shape);

    float tv = Conv::to_float(true_val);
    float fv = Conv::to_float(false_val);

    for (size_t i = 0; i < output.size(); ++i) {
        if (condition[i] != 0) {
            output.data[i] = Conv::from_float(tv);
        } else {
            output.data[i] = Conv::from_float(fv);
        }
    }

    return output;
}

} // namespace infer_train

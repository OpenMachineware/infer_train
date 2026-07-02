#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cmath>

namespace infer_train {

// ============================================================
// L1Loss: mean(|input - target|)
// ============================================================
template<typename T>
Tensor<T> l1_loss(
    const Tensor<T>& input,
    const Tensor<T>& target,
    bool reduction = true
) {
    if (input.shape != target.shape) return Tensor<T>();

    using Conv = DTypeConverter<T>;

    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        sum += std::abs(Conv::to_float(input.data[i]) - Conv::to_float(target.data[i]));
    }

    if (reduction) {
        sum /= static_cast<float>(input.size());
    }

    Tensor<T> result({1});
    result.data[0] = Conv::from_float(sum);
    return result;
}

} // namespace infer_train

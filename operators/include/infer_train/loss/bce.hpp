#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cmath>

namespace infer_train {

// ============================================================
// BCELoss: -[y * log(x) + (1-y) * log(1-x)]
// 输入需要在 [0,1] 范围内（sigmoid 后）
// ============================================================
template<typename T>
Tensor<T> bce_loss(
    const Tensor<T>& input,
    const Tensor<T>& target,
    bool reduction = true,
    float eps = 1e-7f
) {
    if (input.shape != target.shape) return Tensor<T>();

    using Conv = DTypeConverter<T>;

    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        float y = Conv::to_float(target.data[i]);
        x = std::clamp(x, eps, 1.0f - eps);  // 防止 log(0)
        sum -= (y * std::log(x) + (1.0f - y) * std::log(1.0f - x));
    }

    if (reduction) {
        sum /= static_cast<float>(input.size());
    }

    Tensor<T> result({1});
    result.data[0] = Conv::from_float(sum);
    return result;
}

} // namespace infer_train

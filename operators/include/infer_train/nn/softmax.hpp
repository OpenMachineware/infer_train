#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cmath>
#include <algorithm>

namespace infer_train {

// ============================================================
// 浮点 Softmax（模板统一处理）
// ============================================================
template<typename T>
Tensor<T> softmax(const Tensor<T>& input, int dim = -1) {
    static_assert(!is_quantized<T>::value,
                  "Use quantized_softmax for quantized types");

    if (input.shape.empty()) {
        return Tensor<T>();
    }

    // 默认最后一维
    if (dim < 0) {
        dim = static_cast<int>(input.shape.size()) - 1;
    }

    using Conv = DTypeConverter<T>;
    Tensor<T> output(input.shape);

    // 简化版：只支持 2D (batch, features)
    if (input.shape.size() == 2 && dim == 1) {
        size_t batch = input.shape[0];
        size_t features = input.shape[1];

        for (size_t b = 0; b < batch; ++b) {
            // 找最大值
            float max_val = Conv::to_float(input.data[b * features]);
            for (size_t f = 1; f < features; ++f) {
                float val = Conv::to_float(input.data[b * features + f]);
                max_val = std::max(max_val, val);
            }

            // exp(x - max)
            std::vector<float> exp_vals(features);
            float sum = 0.0f;
            for (size_t f = 0; f < features; ++f) {
                float x = Conv::to_float(input.data[b * features + f]);
                exp_vals[f] = std::exp(x - max_val);
                sum += exp_vals[f];
            }

            // 归一化
            for (size_t f = 0; f < features; ++f) {
                output.data[b * features + f] = Conv::from_float(exp_vals[f] / sum);
            }
        }
    } else {
        // 通用版本（较慢）
        // 这里可以后续优化
        return Tensor<T>();  // 暂时返回空
    }

    return output;
}

// ============================================================
// 浮点 LogSoftmax
// ============================================================
template<typename T>
Tensor<T> log_softmax(const Tensor<T>& input, int dim = -1) {
    Tensor<T> s = softmax(input, dim);
    using Conv = DTypeConverter<T>;
    Tensor<T> output(s.shape);

    for (size_t i = 0; i < output.size(); ++i) {
        output.data[i] = Conv::from_float(
            std::log(Conv::to_float(s.data[i]) + 1e-7f)
        );
    }

    return output;
}

} // namespace infer_train

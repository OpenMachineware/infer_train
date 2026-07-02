#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/nn/softmax.hpp"
#include <cmath>

namespace infer_train {

// ============================================================
// CrossEntropyLoss（分类）
// input: (batch, num_classes)
// target: (batch,) 类别索引
// ============================================================
template<typename T>
Tensor<T> cross_entropy_loss(
    const Tensor<T>& input,
    const std::vector<int64_t>& target,
    bool reduction = true
) {
    if (input.shape.size() != 2) return Tensor<T>();
    if (target.size() != input.shape[0]) return Tensor<T>();

    using Conv = DTypeConverter<T>;

    size_t batch = input.shape[0];
    size_t num_classes = input.shape[1];

    Tensor<T> loss({1});
    float total_loss = 0.0f;

    // log_softmax: log(exp(x_i) / sum(exp(x)))
    // 对每个样本
    for (size_t b = 0; b < batch; ++b) {
        // 找到最大值（数值稳定）
        float max_val = Conv::to_float(input.data[b * num_classes]);
        for (size_t c = 1; c < num_classes; ++c) {
            float x = Conv::to_float(input.data[b * num_classes + c]);
            if (x > max_val) max_val = x;
        }

        // 计算 exp 和 sum
        float sum_exp = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            float x = Conv::to_float(input.data[b * num_classes + c]);
            sum_exp += std::exp(x - max_val);
        }

        // log_softmax 和 loss
        int64_t label = target[b];
        if (label >= 0 && label < (int64_t)num_classes) {
            float log_softmax = (Conv::to_float(input.data[b * num_classes + label]) - max_val)
                               - std::log(sum_exp);
            total_loss -= log_softmax;
        }
    }

    if (reduction) {
        total_loss /= static_cast<float>(batch);
    }

    loss.data[0] = Conv::from_float(total_loss);
    return loss;
}

} // namespace infer_train

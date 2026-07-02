#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/math/matmul.hpp"

namespace infer_train {

    // ============================================================
    // Linear（全连接层）：y = x @ W.T + b
    // ============================================================
    template<typename T>
    Tensor<T> linear(
        const Tensor<T>& input,
        const Tensor<T>& weight,
        const Tensor<T>* bias = nullptr
    ) {
        // input: (N, ..., in_features)
        // weight: (out_features, in_features)
        // bias: (out_features,)

        if (input.shape.size() < 2 || weight.shape.size() != 2) {
            return Tensor<T>();
        }

        // 检查 input 最后一维是否等于 weight 的第二维
        size_t in_features = input.shape[input.shape.size() - 1];
        if (in_features != weight.shape[1]) {
            return Tensor<T>();
        }

        // 将 input 展平为 (N, in_features)
        size_t batch = 1;
        for (size_t i = 0; i < input.shape.size() - 1; ++i) {
            batch *= input.shape[i];
        }

        // 重塑 input 为 2D: (batch, in_features)
        // 为了简化，直接 matmul
        Tensor<T> input_2d;
        if (input.shape.size() != 2) {
            // 需要 reshape，这里简化处理
            // 实际应该用 reshape
            return Tensor<T>();
        }

        // y = x @ W.T
        // x: (batch, in_features), W: (out_features, in_features)
        // 需要 x @ W^T: (batch, out_features)
        // 先转置 weight
        Tensor<T> weight_t = transpose(weight);
        Tensor<T> output = matmul(input, weight_t);

        // 加上 bias
        if (bias != nullptr) {
            if (bias->shape.size() != 1 || bias->shape[0] != weight.shape[0]) {
                return Tensor<T>();
            }
            using Conv = DTypeConverter<T>;
            for (size_t i = 0; i < output.size(); ++i) {
                size_t out_idx = i % output.shape[output.shape.size() - 1];
                float b = Conv::to_float(bias->data[out_idx]);
                float val = Conv::to_float(output.data[i]);
                output.data[i] = Conv::from_float(val + b);
            }
        }

        return output;
    }

} // namespace infer_train

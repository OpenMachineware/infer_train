#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/nn/activation.hpp"

namespace infer_train {

// ============================================================
// SwiGLU: x * silu(gate)
// 输入: (batch, seq_len, hidden_size * 2)
// 输出: (batch, seq_len, hidden_size)
// ============================================================
template<typename T>
Tensor<T> swiglu(const Tensor<T>& input) {
    if (input.shape.size() != 3) return Tensor<T>();
    if (input.shape[2] % 2 != 0) return Tensor<T>();

    using Conv = DTypeConverter<T>;

    size_t batch = input.shape[0];
    size_t seq_len = input.shape[1];
    size_t hidden_size = input.shape[2] / 2;

    Tensor<T> output({batch, seq_len, hidden_size});

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t h = 0; h < hidden_size; ++h) {
                // gate = input[:, :, h]
                // x = input[:, :, h + hidden_size]
                float gate_val = Conv::to_float(
                    input.data[b * seq_len * 2 * hidden_size +
                               s * 2 * hidden_size + h]
                );
                float x_val = Conv::to_float(
                    input.data[b * seq_len * 2 * hidden_size +
                               s * 2 * hidden_size + h + hidden_size]
                );
                // silu(gate) * x
                float silu_val = gate_val / (1.0f + std::exp(-gate_val));
                float y = silu_val * x_val;
                output.data[b * seq_len * hidden_size +
                            s * hidden_size + h] = Conv::from_float(y);
            }
        }
    }

    return output;
}

} // namespace infer_train

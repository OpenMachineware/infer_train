#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <vector>
#include <cmath>

namespace infer_train {

// ============================================================
// AdamW 优化器状态
// ============================================================
template<typename T>
struct AdamWState {
    std::vector<Tensor<T>> m;  // 一阶矩
    std::vector<Tensor<T>> v;  // 二阶矩
    size_t step = 0;
};

// ============================================================
// AdamW 更新（weight_decay 在参数更新时做）
// ============================================================
template<typename T>
void adamw_update(
    std::vector<Tensor<T>>& params,
    std::vector<Tensor<T>>& grads,
    AdamWState<T>& state,
    float lr = 0.001f,
    float beta1 = 0.9f,
    float beta2 = 0.999f,
    float eps = 1e-8f,
    float weight_decay = 0.01f
) {
    using Conv = DTypeConverter<T>;

    state.step++;
    float step = static_cast<float>(state.step);

    for (size_t i = 0; i < params.size(); ++i) {
        Tensor<T>& param = params[i];
        Tensor<T>& grad = grads[i];

        if (param.shape != grad.shape) continue;

        if (state.m.size() <= i) {
            state.m.push_back(Tensor<T>(param.shape));
            state.v.push_back(Tensor<T>(param.shape));
        }

        Tensor<T>& m = state.m[i];
        Tensor<T>& v = state.v[i];

        float lr_t = lr * std::sqrt(1.0f - std::pow(beta2, step)) /
                         (1.0f - std::pow(beta1, step));

        for (size_t j = 0; j < param.size(); ++j) {
            float g = Conv::to_float(grad.data[j]);

            // Adam 更新
            float m_val = beta1 * Conv::to_float(m.data[j]) + (1.0f - beta1) * g;
            float v_val = beta2 * Conv::to_float(v.data[j]) + (1.0f - beta2) * g * g;

            m.data[j] = Conv::from_float(m_val);
            v.data[j] = Conv::from_float(v_val);

            float param_val = Conv::to_float(param.data[j]);

            // AdamW: weight_decay 在参数更新时做
            param_val -= lr_t * (m_val / (std::sqrt(v_val) + eps) + weight_decay * param_val);

            param.data[j] = Conv::from_float(param_val);
        }
    }
}

} // namespace infer_train

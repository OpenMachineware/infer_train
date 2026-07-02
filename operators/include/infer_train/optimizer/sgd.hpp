#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <vector>
#include <cmath>

namespace infer_train {

// ============================================================
// SGD 更新
// ============================================================
template<typename T>
void sgd_update(
    std::vector<Tensor<T>>& params,
    std::vector<Tensor<T>>& grads,
    float lr = 0.01f,
    float momentum = 0.0f,
    float weight_decay = 0.0f,
    bool nesterov = false
) {
    using Conv = DTypeConverter<T>;

    static std::vector<Tensor<T>> velocities;

    for (size_t i = 0; i < params.size(); ++i) {
        Tensor<T>& param = params[i];
        Tensor<T>& grad = grads[i];

        if (param.shape != grad.shape) continue;

        // 初始化 velocity
        if (velocities.size() <= i) {
            velocities.push_back(Tensor<T>(param.shape));
        }
        Tensor<T>& v = velocities[i];

        for (size_t j = 0; j < param.size(); ++j) {
            float g = Conv::to_float(grad.data[j]);

            // weight decay
            if (weight_decay > 0.0f) {
                g += weight_decay * Conv::to_float(param.data[j]);
            }

            float v_val;
            if (momentum > 0.0f) {
                v_val = momentum * Conv::to_float(v.data[j]) - lr * g;
                v.data[j] = Conv::from_float(v_val);

                if (nesterov) {
                    v_val = momentum * Conv::to_float(v.data[j]) - lr * g;
                }
            } else {
                v_val = -lr * g;
            }

            float param_val = Conv::to_float(param.data[j]) + v_val;
            param.data[j] = Conv::from_float(param_val);
        }
    }
}

} // namespace infer_train

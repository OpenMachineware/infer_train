#include "c_api_common.h"
#include "infer_train/optimizer.hpp"
#include <vector>

using namespace infer_train;

// ============================================================
// 优化器状态
// ============================================================
struct AdamStateWrapper {
    AdamState<F32> state_f32;
    AdamState<F64> state_f64;
    AdamState<F16> state_f16;
    AdamState<BF16> state_bf16;
    it_dtype_t dtype;
};

struct AdamWStateWrapper {
    AdamWState<F32> state_f32;
    AdamWState<F64> state_f64;
    AdamWState<F16> state_f16;
    AdamWState<BF16> state_bf16;
    it_dtype_t dtype;
};

// ============================================================
// sgd_update
// ============================================================
extern "C" void it_sgd_update(
    it_tensor** params,
    it_tensor** grads,
    size_t num_params,
    float lr,
    float momentum,
    float weight_decay,
    int nesterov
) {
    if (num_params == 0 || params == nullptr || grads == nullptr) return;

    it_dtype_t dtype = params[0]->dtype;

    switch (dtype) {
        case IT_DTYPE_F32: {
            std::vector<Tensor<F32>> p, g;
            for (size_t i = 0; i < num_params; ++i) {
                p.push_back(to_cpp_tensor<F32>(params[i]));
                g.push_back(to_cpp_tensor<F32>(grads[i]));
            }
            sgd_update(p, g, lr, momentum, weight_decay, nesterov != 0);
            // 写回数据
            for (size_t i = 0; i < num_params; ++i) {
                memcpy(params[i]->data, p[i].ptr(), p[i].size() * sizeof(float));
            }
            break;
        }
        case IT_DTYPE_F64: {
            std::vector<Tensor<F64>> p, g;
            for (size_t i = 0; i < num_params; ++i) {
                p.push_back(to_cpp_tensor<F64>(params[i]));
                g.push_back(to_cpp_tensor<F64>(grads[i]));
            }
            sgd_update(p, g, lr, momentum, weight_decay, nesterov != 0);
            for (size_t i = 0; i < num_params; ++i) {
                memcpy(params[i]->data, p[i].ptr(), p[i].size() * sizeof(double));
            }
            break;
        }
        // F16/BF16 类似
        default: break;
    }
}

// ============================================================
// adam_state_new / adam_state_free
// ============================================================
extern "C" void* it_adam_state_new(size_t num_params, const size_t* param_shapes, const size_t* param_ndims) {
    // 简化：返回 null，需要时再实现
    return nullptr;
}

extern "C" void it_adam_state_free(void* state) {
    // 简化
}

extern "C" void it_adam_update(
    it_tensor** params,
    it_tensor** grads,
    size_t num_params,
    void* state,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) {
    // 简化：暂时不实现完整状态管理
    (void)params; (void)grads; (void)num_params; (void)state;
    (void)lr; (void)beta1; (void)beta2; (void)eps; (void)weight_decay;
}

extern "C" void* it_adamw_state_new(size_t num_params, const size_t* param_shapes, const size_t* param_ndims) {
    return nullptr;
}

extern "C" void it_adamw_state_free(void* state) {
    // 简化
}

extern "C" void it_adamw_update(
    it_tensor** params,
    it_tensor** grads,
    size_t num_params,
    void* state,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
) {
    // 简化：暂时不实现完整状态管理
    (void)params; (void)grads; (void)num_params; (void)state;
    (void)lr; (void)beta1; (void)beta2; (void)eps; (void)weight_decay;
}

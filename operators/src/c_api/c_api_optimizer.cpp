#include "c_api_common.h"
#include "infer_train/optimizer.hpp"
#include <vector>
#include <cstring>

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
    size_t num_params;
};

struct AdamWStateWrapper {
    AdamWState<F32> state_f32;
    AdamWState<F64> state_f64;
    AdamWState<F16> state_f16;
    AdamWState<BF16> state_bf16;
    it_dtype_t dtype;
    size_t num_params;
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
        case IT_DTYPE_F16: {
            std::vector<Tensor<F16>> p, g;
            for (size_t i = 0; i < num_params; ++i) {
                p.push_back(to_cpp_tensor<F16>(params[i]));
                g.push_back(to_cpp_tensor<F16>(grads[i]));
            }
            sgd_update(p, g, lr, momentum, weight_decay, nesterov != 0);
            for (size_t i = 0; i < num_params; ++i) {
                memcpy(params[i]->data, p[i].ptr(), p[i].size() * sizeof(uint16_t));
            }
            break;
        }
        case IT_DTYPE_BF16: {
            std::vector<Tensor<BF16>> p, g;
            for (size_t i = 0; i < num_params; ++i) {
                p.push_back(to_cpp_tensor<BF16>(params[i]));
                g.push_back(to_cpp_tensor<BF16>(grads[i]));
            }
            sgd_update(p, g, lr, momentum, weight_decay, nesterov != 0);
            for (size_t i = 0; i < num_params; ++i) {
                memcpy(params[i]->data, p[i].ptr(), p[i].size() * sizeof(uint16_t));
            }
            break;
        }
        default: break;
    }
}

// ============================================================
// adam_state_new
// ============================================================
extern "C" void* it_adam_state_new(
    size_t num_params,
    const size_t* param_shapes,
    const size_t* param_ndims
) {
    if (num_params == 0 || param_shapes == nullptr || param_ndims == nullptr) {
        return nullptr;
    }

    AdamStateWrapper* wrapper = new AdamStateWrapper();
    wrapper->num_params = num_params;
    wrapper->dtype = IT_DTYPE_F32;  // 默认 F32，实际需要从参数推断

    // 初始化每个参数的状态
    size_t offset = 0;
    for (size_t i = 0; i < num_params; ++i) {
        size_t ndim = param_ndims[i];
        std::vector<size_t> shape(param_shapes + offset, param_shapes + offset + ndim);
        offset += ndim;

        // 根据 dtype 初始化对应类型的状态
        // 这里简化：只初始化 F32，实际需要根据参数类型选择
        wrapper->state_f32.m.push_back(Tensor<F32>(shape));
        wrapper->state_f32.v.push_back(Tensor<F32>(shape));
    }
    wrapper->state_f32.step = 0;

    return wrapper;
}

// ============================================================
// adam_state_free
// ============================================================
extern "C" void it_adam_state_free(void* state) {
    if (state) {
        delete static_cast<AdamStateWrapper*>(state);
    }
}

// ============================================================
// adam_update
// ============================================================
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
    if (num_params == 0 || params == nullptr || grads == nullptr || state == nullptr) {
        return;
    }

    AdamStateWrapper* wrapper = static_cast<AdamStateWrapper*>(state);

    // 根据 dtype 选择更新路径
    it_dtype_t dtype = params[0]->dtype;

    switch (dtype) {
        case IT_DTYPE_F32: {
            std::vector<Tensor<F32>> p, g;
            for (size_t i = 0; i < num_params; ++i) {
                p.push_back(to_cpp_tensor<F32>(params[i]));
                g.push_back(to_cpp_tensor<F32>(grads[i]));
            }
            // 使用 wrapper 中的状态
            adam_update(p, g, wrapper->state_f32, lr, beta1, beta2, eps, weight_decay);
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
            adam_update(p, g, wrapper->state_f64, lr, beta1, beta2, eps, weight_decay);
            for (size_t i = 0; i < num_params; ++i) {
                memcpy(params[i]->data, p[i].ptr(), p[i].size() * sizeof(double));
            }
            break;
        }
        case IT_DTYPE_F16: {
            std::vector<Tensor<F16>> p, g;
            for (size_t i = 0; i < num_params; ++i) {
                p.push_back(to_cpp_tensor<F16>(params[i]));
                g.push_back(to_cpp_tensor<F16>(grads[i]));
            }
            adam_update(p, g, wrapper->state_f16, lr, beta1, beta2, eps, weight_decay);
            for (size_t i = 0; i < num_params; ++i) {
                memcpy(params[i]->data, p[i].ptr(), p[i].size() * sizeof(uint16_t));
            }
            break;
        }
        case IT_DTYPE_BF16: {
            std::vector<Tensor<BF16>> p, g;
            for (size_t i = 0; i < num_params; ++i) {
                p.push_back(to_cpp_tensor<BF16>(params[i]));
                g.push_back(to_cpp_tensor<BF16>(grads[i]));
            }
            adam_update(p, g, wrapper->state_bf16, lr, beta1, beta2, eps, weight_decay);
            for (size_t i = 0; i < num_params; ++i) {
                memcpy(params[i]->data, p[i].ptr(), p[i].size() * sizeof(uint16_t));
            }
            break;
        }
        default: break;
    }
}

// ============================================================
// adamw_state_new
// ============================================================
extern "C" void* it_adamw_state_new(
    size_t num_params,
    const size_t* param_shapes,
    const size_t* param_ndims
) {
    if (num_params == 0 || param_shapes == nullptr || param_ndims == nullptr) {
        return nullptr;
    }

    AdamWStateWrapper* wrapper = new AdamWStateWrapper();
    wrapper->num_params = num_params;
    wrapper->dtype = IT_DTYPE_F32;

    size_t offset = 0;
    for (size_t i = 0; i < num_params; ++i) {
        size_t ndim = param_ndims[i];
        std::vector<size_t> shape(param_shapes + offset, param_shapes + offset + ndim);
        offset += ndim;

        wrapper->state_f32.m.push_back(Tensor<F32>(shape));
        wrapper->state_f32.v.push_back(Tensor<F32>(shape));
    }
    wrapper->state_f32.step = 0;

    return wrapper;
}

// ============================================================
// adamw_state_free
// ============================================================
extern "C" void it_adamw_state_free(void* state) {
    if (state) {
        delete static_cast<AdamWStateWrapper*>(state);
    }
}

// ============================================================
// adamw_update
// ============================================================
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
    if (num_params == 0 || params == nullptr || grads == nullptr || state == nullptr) {
        return;
    }

    AdamWStateWrapper* wrapper = static_cast<AdamWStateWrapper*>(state);
    it_dtype_t dtype = params[0]->dtype;

    switch (dtype) {
        case IT_DTYPE_F32: {
            std::vector<Tensor<F32>> p, g;
            for (size_t i = 0; i < num_params; ++i) {
                p.push_back(to_cpp_tensor<F32>(params[i]));
                g.push_back(to_cpp_tensor<F32>(grads[i]));
            }
            adamw_update(p, g, wrapper->state_f32, lr, beta1, beta2, eps, weight_decay);
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
            adamw_update(p, g, wrapper->state_f64, lr, beta1, beta2, eps, weight_decay);
            for (size_t i = 0; i < num_params; ++i) {
                memcpy(params[i]->data, p[i].ptr(), p[i].size() * sizeof(double));
            }
            break;
        }
        case IT_DTYPE_F16: {
            std::vector<Tensor<F16>> p, g;
            for (size_t i = 0; i < num_params; ++i) {
                p.push_back(to_cpp_tensor<F16>(params[i]));
                g.push_back(to_cpp_tensor<F16>(grads[i]));
            }
            adamw_update(p, g, wrapper->state_f16, lr, beta1, beta2, eps, weight_decay);
            for (size_t i = 0; i < num_params; ++i) {
                memcpy(params[i]->data, p[i].ptr(), p[i].size() * sizeof(uint16_t));
            }
            break;
        }
        case IT_DTYPE_BF16: {
            std::vector<Tensor<BF16>> p, g;
            for (size_t i = 0; i < num_params; ++i) {
                p.push_back(to_cpp_tensor<BF16>(params[i]));
                g.push_back(to_cpp_tensor<BF16>(grads[i]));
            }
            adamw_update(p, g, wrapper->state_bf16, lr, beta1, beta2, eps, weight_decay);
            for (size_t i = 0; i < num_params; ++i) {
                memcpy(params[i]->data, p[i].ptr(), p[i].size() * sizeof(uint16_t));
            }
            break;
        }
        default: break;
    }
}

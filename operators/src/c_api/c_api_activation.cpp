#include "infer_train/c_interface.h"
#include "infer_train/nn.hpp"
#include "c_api_common.h"

using namespace infer_train;

// ============================================================
// 分派宏：一元浮点激活函数
// ============================================================
#define DISPATCH_ACTIVATION_UNARY(op) \
    switch (input->dtype) { \
        case IT_DTYPE_F32: { \
            auto t = to_cpp_tensor<F32>(input); \
            auto result = op(t); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto t = to_cpp_tensor<F64>(input); \
            auto result = op(t); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto t = to_cpp_tensor<F16>(input); \
            auto result = op(t); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto t = to_cpp_tensor<BF16>(input); \
            auto result = op(t); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: return nullptr; \
    }

#define DISPATCH_ACTIVATION_UNARY_WITH(op, arg) \
    switch (input->dtype) { \
        case IT_DTYPE_F32: { \
            auto t = to_cpp_tensor<F32>(input); \
            auto result = op(t, arg); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto t = to_cpp_tensor<F64>(input); \
            auto result = op(t, (double)arg); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto t = to_cpp_tensor<F16>(input); \
            auto result = op(t, (float)arg); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto t = to_cpp_tensor<BF16>(input); \
            auto result = op(t, (float)arg); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: return nullptr; \
    }

// ============================================================
// 激活函数
// ============================================================
extern "C" it_tensor* it_relu(const it_tensor* input) {
    DISPATCH_ACTIVATION_UNARY(relu);
}

extern "C" it_tensor* it_leaky_relu(const it_tensor* input, float alpha) {
    DISPATCH_ACTIVATION_UNARY_WITH(leaky_relu, alpha);
}

extern "C" it_tensor* it_elu(const it_tensor* input, float alpha) {
    DISPATCH_ACTIVATION_UNARY_WITH(elu, alpha);
}

extern "C" it_tensor* it_gelu(const it_tensor* input) {
    DISPATCH_ACTIVATION_UNARY(gelu);
}

extern "C" it_tensor* it_sigmoid(const it_tensor* input) {
    DISPATCH_ACTIVATION_UNARY(sigmoid);
}

extern "C" it_tensor* it_tanh(const it_tensor* input) {
    DISPATCH_ACTIVATION_UNARY(tanh);
}

extern "C" it_tensor* it_silu(const it_tensor* input) {
    DISPATCH_ACTIVATION_UNARY(silu);
}

extern "C" it_tensor* it_hard_swish(const it_tensor* input) {
    DISPATCH_ACTIVATION_UNARY(hard_swish);
}

extern "C" it_tensor* it_hard_sigmoid(const it_tensor* input) {
    DISPATCH_ACTIVATION_UNARY(hard_sigmoid);
}

// ============================================================
// Softmax（带 dim 参数）
// ============================================================
extern "C" it_tensor* it_softmax(const it_tensor* input, int dim) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = softmax(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = softmax(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = softmax(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = softmax(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_log_softmax(const it_tensor* input, int dim) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = log_softmax(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = log_softmax(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = log_softmax(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = log_softmax(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

#include "infer_train/c_interface.h"
#include "infer_train/nn_q.hpp"
#include "c_api_common.h"

using namespace infer_train;

// ============================================================
// 量化激活函数（只支持 I8）
// ============================================================
#define DISPATCH_QUANTIZED_ACTIVATION_UNARY(op) \
    if (input->dtype != IT_DTYPE_I8) return nullptr; \
    auto t = to_cpp_tensor_with_params<I8>(input); \
    auto result = op(t); \
    return from_cpp_tensor(result, IT_DTYPE_I8);

#define DISPATCH_QUANTIZED_ACTIVATION_UNARY_WITH(op, arg) \
    if (input->dtype != IT_DTYPE_I8) return nullptr; \
    auto t = to_cpp_tensor_with_params<I8>(input); \
    auto result = op(t, arg); \
    return from_cpp_tensor(result, IT_DTYPE_I8);

extern "C" it_tensor* it_quantized_relu(const it_tensor* input) {
    DISPATCH_QUANTIZED_ACTIVATION_UNARY(quantized_relu);
}

extern "C" it_tensor* it_quantized_leaky_relu(const it_tensor* input, float alpha) {
    DISPATCH_QUANTIZED_ACTIVATION_UNARY_WITH(quantized_leaky_relu, alpha);
}

extern "C" it_tensor* it_quantized_elu(const it_tensor* input, float alpha) {
    DISPATCH_QUANTIZED_ACTIVATION_UNARY_WITH(quantized_elu, alpha);
}

extern "C" it_tensor* it_quantized_gelu(const it_tensor* input) {
    DISPATCH_QUANTIZED_ACTIVATION_UNARY(quantized_gelu);
}

extern "C" it_tensor* it_quantized_sigmoid(const it_tensor* input) {
    DISPATCH_QUANTIZED_ACTIVATION_UNARY(quantized_sigmoid);
}

extern "C" it_tensor* it_quantized_tanh(const it_tensor* input) {
    DISPATCH_QUANTIZED_ACTIVATION_UNARY(quantized_tanh);
}

extern "C" it_tensor* it_quantized_silu(const it_tensor* input) {
    DISPATCH_QUANTIZED_ACTIVATION_UNARY(quantized_silu);
}

extern "C" it_tensor* it_quantized_hard_swish(const it_tensor* input) {
    DISPATCH_QUANTIZED_ACTIVATION_UNARY(quantized_hard_swish);
}

extern "C" it_tensor* it_quantized_hard_sigmoid(const it_tensor* input) {
    DISPATCH_QUANTIZED_ACTIVATION_UNARY(quantized_hard_sigmoid);
}

extern "C" it_tensor* it_quantized_softmax(const it_tensor* input, int dim) {
    if (input->dtype != IT_DTYPE_I8) return nullptr;
    auto t = to_cpp_tensor_with_params<I8>(input);
    auto result = quantized_softmax(t, dim);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

extern "C" it_tensor* it_quantized_log_softmax(const it_tensor* input, int dim) {
    if (input->dtype != IT_DTYPE_I8) return nullptr;
    auto t = to_cpp_tensor_with_params<I8>(input);
    auto result = quantized_log_softmax(t, dim);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

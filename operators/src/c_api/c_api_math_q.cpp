#include "c_api_common.h"
#include "infer_train/math_q.hpp"

using namespace infer_train;

// ============================================================
// 量化数学算子（只支持 I8）
// ============================================================
#define DISPATCH_QUANTIZED_BINARY(op) \
    if (a->dtype != IT_DTYPE_I8 || b->dtype != IT_DTYPE_I8) return nullptr; \
    auto ta = to_cpp_tensor_with_params<I8>(a); \
    auto tb = to_cpp_tensor_with_params<I8>(b); \
    auto result = op(ta, tb); \
    return from_cpp_tensor(result, IT_DTYPE_I8);

#define DISPATCH_QUANTIZED_UNARY(op) \
    if (input->dtype != IT_DTYPE_I8) return nullptr; \
    auto t = to_cpp_tensor_with_params<I8>(input); \
    auto result = op(t); \
    return from_cpp_tensor(result, IT_DTYPE_I8);

extern "C" it_tensor* it_quantized_add(const it_tensor* a, const it_tensor* b) {
    DISPATCH_QUANTIZED_BINARY(quantized_add);
}

extern "C" it_tensor* it_quantized_sub(const it_tensor* a, const it_tensor* b) {
    DISPATCH_QUANTIZED_BINARY(quantized_sub);
}

extern "C" it_tensor* it_quantized_mul(const it_tensor* a, const it_tensor* b) {
    DISPATCH_QUANTIZED_BINARY(quantized_mul);
}

extern "C" it_tensor* it_quantized_div(const it_tensor* a, const it_tensor* b) {
    DISPATCH_QUANTIZED_BINARY(quantized_div);
}

extern "C" it_tensor* it_quantized_exp(const it_tensor* input) {
    DISPATCH_QUANTIZED_UNARY(quantized_exp);
}

extern "C" it_tensor* it_quantized_sqrt(const it_tensor* input) {
    DISPATCH_QUANTIZED_UNARY(quantized_sqrt);
}

extern "C" it_tensor* it_quantized_abs(const it_tensor* input) {
    DISPATCH_QUANTIZED_UNARY(quantized_abs);
}

extern "C" it_tensor* it_quantized_neg(const it_tensor* input) {
    DISPATCH_QUANTIZED_UNARY(quantized_neg);
}

extern "C" it_tensor* it_quantized_clamp(const it_tensor* input, float min_val, float max_val) {
    if (input->dtype != IT_DTYPE_I8) return nullptr;
    auto t = to_cpp_tensor_with_params<I8>(input);
    auto result = quantized_clamp(t, (int8_t)min_val, (int8_t)max_val);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

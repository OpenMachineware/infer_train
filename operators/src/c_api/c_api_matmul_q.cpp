#include "infer_train/c_interface.h"
#include "infer_train/math_q.hpp"
#include "c_api_common.h"

using namespace infer_train;

extern "C" it_tensor* it_quantized_matmul(const it_tensor* a, const it_tensor* b) {
    if (a->dtype != IT_DTYPE_I8 || b->dtype != IT_DTYPE_I8) return nullptr;
    auto ta = to_cpp_tensor_with_params<I8>(a);
    auto tb = to_cpp_tensor_with_params<I8>(b);
    auto result = quantized_matmul(ta, tb);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

extern "C" it_tensor* it_quantized_vec_matmul(const it_tensor* vec, const it_tensor* mat) {
    if (vec->dtype != IT_DTYPE_I8 || mat->dtype != IT_DTYPE_I8) return nullptr;
    auto tv = to_cpp_tensor_with_params<I8>(vec);
    auto tm = to_cpp_tensor_with_params<I8>(mat);
    auto result = quantized_vec_matmul(tv, tm);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

extern "C" it_tensor* it_quantized_transpose(const it_tensor* input) {
    if (input->dtype != IT_DTYPE_I8) return nullptr;
    auto t = to_cpp_tensor_with_params<I8>(input);
    auto result = quantized_transpose(t);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

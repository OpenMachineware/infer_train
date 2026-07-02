#include "infer_train/c_interface.h"
#include "infer_train/math_q.hpp"
#include "c_api_common.h"
#include <vector>

using namespace infer_train;

static inline std::vector<int> dims_to_vector(const int* dims, size_t ndim) {
    return std::vector<int>(dims, dims + ndim);
}

extern "C" it_tensor* it_quantized_sum(const it_tensor* input, const int* dims, size_t ndim, int keepdim) {
    if (input->dtype != IT_DTYPE_I8) return nullptr;
    auto d = dims_to_vector(dims, ndim);
    auto t = to_cpp_tensor_with_params<I8>(input);
    auto result = quantized_sum(t, d, keepdim != 0);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

extern "C" it_tensor* it_quantized_mean(const it_tensor* input, const int* dims, size_t ndim, int keepdim) {
    if (input->dtype != IT_DTYPE_I8) return nullptr;
    auto d = dims_to_vector(dims, ndim);
    auto t = to_cpp_tensor_with_params<I8>(input);
    auto result = quantized_mean(t, d, keepdim != 0);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

extern "C" it_tensor* it_quantized_max_all(const it_tensor* input) {
    if (input->dtype != IT_DTYPE_I8) return nullptr;
    auto t = to_cpp_tensor_with_params<I8>(input);
    auto result = quantized_max_all(t);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

extern "C" it_tensor* it_quantized_min_all(const it_tensor* input) {
    if (input->dtype != IT_DTYPE_I8) return nullptr;
    auto t = to_cpp_tensor_with_params<I8>(input);
    auto result = quantized_min_all(t);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

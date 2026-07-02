#include "c_api_common.h"
#include "infer_train/math.hpp"
#include <vector>

using namespace infer_train;

// ============================================================
// 辅助：将 C 数组转为 vector
// ============================================================
static inline std::vector<int> dims_to_vector(const int* dims, size_t ndim) {
    return std::vector<int>(dims, dims + ndim);
}

#define DISPATCH_REDUCE(op, dims, ndim, keepdim) \
    auto d = dims_to_vector(dims, ndim); \
    switch (input->dtype) { \
        case IT_DTYPE_F32: { \
            auto t = to_cpp_tensor<F32>(input); \
            auto result = op(t, d, keepdim != 0); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto t = to_cpp_tensor<F64>(input); \
            auto result = op(t, d, keepdim != 0); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto t = to_cpp_tensor<F16>(input); \
            auto result = op(t, d, keepdim != 0); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto t = to_cpp_tensor<BF16>(input); \
            auto result = op(t, d, keepdim != 0); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: return nullptr; \
    }

#define DISPATCH_REDUCE_UNARY(op) \
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

// ============================================================
// 浮点规约
// ============================================================
extern "C" it_tensor* it_sum(const it_tensor* input, const int* dims, size_t ndim, int keepdim) {
    DISPATCH_REDUCE(sum, dims, ndim, keepdim);
}

extern "C" it_tensor* it_mean(const it_tensor* input, const int* dims, size_t ndim, int keepdim) {
    DISPATCH_REDUCE(mean, dims, ndim, keepdim);
}

extern "C" it_tensor* it_max_all(const it_tensor* input) {
    DISPATCH_REDUCE_UNARY(max_all);
}

extern "C" it_tensor* it_min_all(const it_tensor* input) {
    DISPATCH_REDUCE_UNARY(min_all);
}

extern "C" it_tensor* it_prod_all(const it_tensor* input) {
    DISPATCH_REDUCE_UNARY(prod_all);
}

extern "C" it_tensor* it_var(const it_tensor* input, int unbiased) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = var(t, unbiased != 0);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = var(t, unbiased != 0);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = var(t, unbiased != 0);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = var(t, unbiased != 0);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_std(const it_tensor* input, int unbiased) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = infer_train::std(t, unbiased != 0);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = infer_train::std(t, unbiased != 0);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = infer_train::std(t, unbiased != 0);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = infer_train::std(t, unbiased != 0);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

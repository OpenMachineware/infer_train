#include "infer_train/c_interface.h"
#include "infer_train/math.hpp"
#include "infer_train/math_q.hpp"
#include "c_api_common.h"
#include <vector>

using namespace infer_train;

// ============================================================
// 辅助：将 C 数组转为 vector
// ============================================================
static inline std::vector<int> dims_to_vector(const int* dims, size_t ndim) {
    return std::vector<int>(dims, dims + ndim);
}

// ============================================================
// 浮点规约
// ============================================================
extern "C" it_tensor* it_sum(const it_tensor* input, const int* dims, size_t ndim, int keepdim) {
    auto d = dims_to_vector(dims, ndim);
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = sum(t, d, keepdim != 0);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = sum(t, d, keepdim != 0);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = sum(t, d, keepdim != 0);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = sum(t, d, keepdim != 0);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_mean(const it_tensor* input, const int* dims, size_t ndim, int keepdim) {
    auto d = dims_to_vector(dims, ndim);
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = mean(t, d, keepdim != 0);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = mean(t, d, keepdim != 0);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = mean(t, d, keepdim != 0);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = mean(t, d, keepdim != 0);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_max_all(const it_tensor* input) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = max_all(t);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = max_all(t);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = max_all(t);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = max_all(t);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_min_all(const it_tensor* input) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = min_all(t);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = min_all(t);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = min_all(t);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = min_all(t);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

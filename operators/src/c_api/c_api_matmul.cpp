#include "infer_train/c_interface.h"
#include "infer_train/math.hpp"
#include "infer_train/math_q.hpp"
#include "c_api_common.h"

using namespace infer_train;

// ============================================================
// 浮点矩阵算子
// ============================================================
#define DISPATCH_MATMUL_BINARY(op) \
    switch (a->dtype) { \
        case IT_DTYPE_F32: { \
            auto ta = to_cpp_tensor<F32>(a); \
            auto tb = to_cpp_tensor<F32>(b); \
            auto result = op(ta, tb); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto ta = to_cpp_tensor<F64>(a); \
            auto tb = to_cpp_tensor<F64>(b); \
            auto result = op(ta, tb); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto ta = to_cpp_tensor<F16>(a); \
            auto tb = to_cpp_tensor<F16>(b); \
            auto result = op(ta, tb); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto ta = to_cpp_tensor<BF16>(a); \
            auto tb = to_cpp_tensor<BF16>(b); \
            auto result = op(ta, tb); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: return nullptr; \
    }

#define DISPATCH_MATMUL_UNARY(op) \
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

extern "C" it_tensor* it_matmul(const it_tensor* a, const it_tensor* b) {
    DISPATCH_MATMUL_BINARY(matmul);
}

extern "C" it_tensor* it_batch_matmul(const it_tensor* a, const it_tensor* b) {
    DISPATCH_MATMUL_BINARY(batch_matmul);
}

extern "C" it_tensor* it_vec_matmul(const it_tensor* vec, const it_tensor* mat) {
    switch (vec->dtype) {
        case IT_DTYPE_F32: {
            auto tv = to_cpp_tensor<F32>(vec);
            auto tm = to_cpp_tensor<F32>(mat);
            auto result = vec_matmul(tv, tm);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto tv = to_cpp_tensor<F64>(vec);
            auto tm = to_cpp_tensor<F64>(mat);
            auto result = vec_matmul(tv, tm);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto tv = to_cpp_tensor<F16>(vec);
            auto tm = to_cpp_tensor<F16>(mat);
            auto result = vec_matmul(tv, tm);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto tv = to_cpp_tensor<BF16>(vec);
            auto tm = to_cpp_tensor<BF16>(mat);
            auto result = vec_matmul(tv, tm);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_transpose(const it_tensor* input) {
    DISPATCH_MATMUL_UNARY(transpose);
}

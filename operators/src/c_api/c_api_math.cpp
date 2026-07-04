#include "c_api_common.h"
#include "infer_train/math.hpp"

using namespace infer_train;

// ============================================================
// 分派宏：二元浮点数学算子
// ============================================================
#define DISPATCH_MATH_BINARY(op) \
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

// ============================================================
// 分派宏：一元浮点数学算子
// ============================================================
#define DISPATCH_MATH_UNARY(op) \
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
// 二元算子
// ============================================================
extern "C" it_tensor* it_add(const it_tensor* a, const it_tensor* b) {
    DISPATCH_MATH_BINARY(add);
}

extern "C" it_tensor* it_sub(const it_tensor* a, const it_tensor* b) {
    DISPATCH_MATH_BINARY(sub);
}

extern "C" it_tensor* it_mul(const it_tensor* a, const it_tensor* b) {
    DISPATCH_MATH_BINARY(mul);
}

extern "C" it_tensor* it_div(const it_tensor* a, const it_tensor* b) {
    DISPATCH_MATH_BINARY(div);
}

extern "C" it_tensor* it_pow(const it_tensor* a, const it_tensor* b) {
    DISPATCH_MATH_BINARY(pow);
}

// ============================================================
// 一元算子
// ============================================================
extern "C" it_tensor* it_exp(const it_tensor* input) {
    DISPATCH_MATH_UNARY(exp);
}

extern "C" it_tensor* it_sqrt(const it_tensor* input) {
    DISPATCH_MATH_UNARY(sqrt);
}

extern "C" it_tensor* it_log(const it_tensor* input) {
    DISPATCH_MATH_UNARY(log);
}

extern "C" it_tensor* it_log2(const it_tensor* input) {
    DISPATCH_MATH_UNARY(log2);
}

extern "C" it_tensor* it_log10(const it_tensor* input) {
    DISPATCH_MATH_UNARY(log10);
}

extern "C" it_tensor* it_abs(const it_tensor* input) {
    DISPATCH_MATH_UNARY(abs);
}

extern "C" it_tensor* it_neg(const it_tensor* input) {
    DISPATCH_MATH_UNARY(neg);
}

extern "C" it_tensor* it_clamp(const it_tensor* input, float min_val, float max_val) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = clamp(t, min_val, max_val);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = clamp(t, (double)min_val, (double)max_val);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = clamp(t, (uint16_t)min_val, (uint16_t)max_val);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = clamp(t, (uint16_t)min_val, (uint16_t)max_val);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_floor(const it_tensor* input) {
    DISPATCH_MATH_UNARY(floor);
}

extern "C" it_tensor* it_ceil(const it_tensor* input) {
    DISPATCH_MATH_UNARY(ceil);
}

extern "C" it_tensor* it_round(const it_tensor* input) {
    DISPATCH_MATH_UNARY(round);
}

// ============================================================
// 标量版本
// ============================================================
extern "C" it_tensor* it_add_scalar(const it_tensor* a, float scalar) {
    switch (a->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(a);
            auto result = add_scalar(t, scalar);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(a);
            auto result = add_scalar(t, (double)scalar);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(a);
            auto result = add_scalar(t, (uint16_t)scalar);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(a);
            auto result = add_scalar(t, (uint16_t)scalar);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_sub_scalar(const it_tensor* a, float scalar) {
    switch (a->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(a);
            auto result = sub_scalar(t, scalar);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(a);
            auto result = sub_scalar(t, (double)scalar);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(a);
            auto result = sub_scalar(t, (uint16_t)scalar);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(a);
            auto result = sub_scalar(t, (uint16_t)scalar);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_mul_scalar(const it_tensor* a, float scalar) {
    switch (a->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(a);
            auto result = mul_scalar(t, scalar);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(a);
            auto result = mul_scalar(t, (double)scalar);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(a);
            auto result = mul_scalar(t, (uint16_t)scalar);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(a);
            auto result = mul_scalar(t, (uint16_t)scalar);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_div_scalar(const it_tensor* a, float scalar) {
    switch (a->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(a);
            auto result = div_scalar(t, scalar);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(a);
            auto result = div_scalar(t, (double)scalar);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(a);
            auto result = div_scalar(t, (uint16_t)scalar);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(a);
            auto result = div_scalar(t, (uint16_t)scalar);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_pow_scalar(const it_tensor* a, float exponent) {
    switch (a->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(a);
            auto result = pow_scalar(t, exponent);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(a);
            auto result = pow_scalar(t, (double)exponent);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(a);
            auto result = pow_scalar(t, (uint16_t)exponent);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(a);
            auto result = pow_scalar(t, (uint16_t)exponent);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_scalar_sub(float scalar, const it_tensor* a) {
    switch (a->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(a);
            auto result = scalar_sub(scalar, t);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(a);
            auto result = scalar_sub((double)scalar, t);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(a);
            auto result = scalar_sub((uint16_t)scalar, t);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(a);
            auto result = scalar_sub((uint16_t)scalar, t);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_scalar_div(float scalar, const it_tensor* a) {
    switch (a->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(a);
            auto result = scalar_div(scalar, t);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(a);
            auto result = scalar_div((double)scalar, t);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(a);
            auto result = scalar_div((uint16_t)scalar, t);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(a);
            auto result = scalar_div((uint16_t)scalar, t);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

#define IMPL_COMPARE_OP(OP_NAME) \
extern "C" it_tensor* it_##OP_NAME(const it_tensor* a, const it_tensor* b) { \
    if (!a || !b) return nullptr; \
    if (a->dtype != b->dtype) return nullptr; \
    switch (a->dtype) { \
        case IT_DTYPE_F32: { \
            auto ta = to_cpp_tensor<F32>(a); \
            auto tb = to_cpp_tensor<F32>(b); \
            auto result = OP_NAME(ta, tb); \
            it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor)); \
            ct->ndim = a->ndim; \
            ct->shape = (size_t*)malloc(a->ndim * sizeof(size_t)); \
            memcpy(ct->shape, a->shape, a->ndim * sizeof(size_t)); \
            ct->data = malloc(result.size() * sizeof(uint8_t)); \
            memcpy(ct->data, result.data(), result.size() * sizeof(uint8_t)); \
            ct->elem_size = sizeof(uint8_t); \
            ct->dtype = IT_DTYPE_U8; \
            ct->scale = 1.0f; \
            ct->zero_point = 0.0f; \
            ct->owns_memory = true; \
            return ct; \
        } \
        case IT_DTYPE_F64: { \
            auto ta = to_cpp_tensor<F64>(a); \
            auto tb = to_cpp_tensor<F64>(b); \
            auto result = OP_NAME(ta, tb); \
            it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor)); \
            ct->ndim = a->ndim; \
            ct->shape = (size_t*)malloc(a->ndim * sizeof(size_t)); \
            memcpy(ct->shape, a->shape, a->ndim * sizeof(size_t)); \
            ct->data = malloc(result.size() * sizeof(uint8_t)); \
            memcpy(ct->data, result.data(), result.size() * sizeof(uint8_t)); \
            ct->elem_size = sizeof(uint8_t); \
            ct->dtype = IT_DTYPE_U8; \
            ct->scale = 1.0f; \
            ct->zero_point = 0.0f; \
            ct->owns_memory = true; \
            return ct; \
        } \
        case IT_DTYPE_F16: { \
            auto ta = to_cpp_tensor<F16>(a); \
            auto tb = to_cpp_tensor<F16>(b); \
            auto result = OP_NAME(ta, tb); \
            it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor)); \
            ct->ndim = a->ndim; \
            ct->shape = (size_t*)malloc(a->ndim * sizeof(size_t)); \
            memcpy(ct->shape, a->shape, a->ndim * sizeof(size_t)); \
            ct->data = malloc(result.size() * sizeof(uint8_t)); \
            memcpy(ct->data, result.data(), result.size() * sizeof(uint8_t)); \
            ct->elem_size = sizeof(uint8_t); \
            ct->dtype = IT_DTYPE_U8; \
            ct->scale = 1.0f; \
            ct->zero_point = 0.0f; \
            ct->owns_memory = true; \
            return ct; \
        } \
        case IT_DTYPE_BF16: { \
            auto ta = to_cpp_tensor<BF16>(a); \
            auto tb = to_cpp_tensor<BF16>(b); \
            auto result = OP_NAME(ta, tb); \
            it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor)); \
            ct->ndim = a->ndim; \
            ct->shape = (size_t*)malloc(a->ndim * sizeof(size_t)); \
            memcpy(ct->shape, a->shape, a->ndim * sizeof(size_t)); \
            ct->data = malloc(result.size() * sizeof(uint8_t)); \
            memcpy(ct->data, result.data(), result.size() * sizeof(uint8_t)); \
            ct->elem_size = sizeof(uint8_t); \
            ct->dtype = IT_DTYPE_U8; \
            ct->scale = 1.0f; \
            ct->zero_point = 0.0f; \
            ct->owns_memory = true; \
            return ct; \
        } \
        case IT_DTYPE_I8: { \
            auto ta = to_cpp_tensor<I8>(a); \
            auto tb = to_cpp_tensor<I8>(b); \
            auto result = OP_NAME(ta, tb); \
            it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor)); \
            ct->ndim = a->ndim; \
            ct->shape = (size_t*)malloc(a->ndim * sizeof(size_t)); \
            memcpy(ct->shape, a->shape, a->ndim * sizeof(size_t)); \
            ct->data = malloc(result.size() * sizeof(uint8_t)); \
            memcpy(ct->data, result.data(), result.size() * sizeof(uint8_t)); \
            ct->elem_size = sizeof(uint8_t); \
            ct->dtype = IT_DTYPE_U8; \
            ct->scale = 1.0f; \
            ct->zero_point = 0.0f; \
            ct->owns_memory = true; \
            return ct; \
        } \
        default: return nullptr; \
    } \
}

// 展开六个比较函数
IMPL_COMPARE_OP(eq)
IMPL_COMPARE_OP(ne)
IMPL_COMPARE_OP(gt)
IMPL_COMPARE_OP(lt)
IMPL_COMPARE_OP(ge)
IMPL_COMPARE_OP(le)

extern "C" it_tensor* it_reshape(
    const it_tensor* input,
    const size_t* new_shape,
    size_t ndim
) {
    if (!input || !new_shape || ndim == 0) return nullptr;

    std::vector<size_t> shape_vec(new_shape, new_shape + ndim);

    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = reshape(t, shape_vec);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = reshape(t, shape_vec);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = reshape(t, shape_vec);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = reshape(t, shape_vec);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        case IT_DTYPE_I8: {
            auto t = to_cpp_tensor<I8>(input);
            auto result = reshape(t, shape_vec);
            return from_cpp_tensor(result, IT_DTYPE_I8);
        }
        default: return nullptr;
    }
}

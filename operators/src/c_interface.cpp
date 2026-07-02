#include "infer_train/c_interface.h"
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/math/add.hpp"
#include "infer_train/math/matmul.hpp"
#include "infer_train/math_q.hpp"
#include "infer_train/nn/relu.hpp"
#include "infer_train/nn/softmax.hpp"
#include "infer_train/nn_q.hpp"
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace infer_train;

// ============================================================
// Tensor 包装（不透明类型）
// ============================================================
struct it_tensor {
    void* data;
    size_t* shape;
    size_t ndim;
    size_t elem_size;
    it_dtype_t dtype;
    float scale;
    float zero_point;
    bool owns_memory;
};

// ============================================================
// 类型转换辅助
// ============================================================
template<typename T>
Tensor<T> to_cpp_tensor(const it_tensor* ct) {
    std::vector<size_t> shape_vec(ct->shape, ct->shape + ct->ndim);
    return Tensor<T>(
        static_cast<const typename T::storage*>(ct->data),
        shape_vec
    );
}

template<typename T>
Tensor<T> to_cpp_tensor_with_params(const it_tensor* ct) {
    Tensor<T> t = to_cpp_tensor<T>(ct);
    t.scale = ct->scale;
    t.zero_point = ct->zero_point;
    return t;
}

template<typename T>
it_tensor* from_cpp_tensor(const Tensor<T>& t, it_dtype_t dtype) {
    it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor));
    ct->ndim = t.shape.size();
    ct->shape = (size_t*)malloc(ct->ndim * sizeof(size_t));
    for (size_t i = 0; i < ct->ndim; ++i) {
        ct->shape[i] = t.shape[i];
    }
    size_t bytes = t.size() * sizeof(typename T::storage);
    ct->data = malloc(bytes);
    memcpy(ct->data, t.ptr(), bytes);
    ct->elem_size = sizeof(typename T::storage);
    ct->dtype = dtype;
    ct->scale = t.scale;
    ct->zero_point = t.zero_point;
    ct->owns_memory = true;
    return ct;
}

// ============================================================
// Tensor 生命周期
// ============================================================
extern "C" it_tensor* it_tensor_new(
    const void* data,
    const size_t* shape,
    size_t ndim,
    it_dtype_t dtype
) {
    it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor));
    ct->ndim = ndim;
    ct->shape = (size_t*)malloc(ndim * sizeof(size_t));
    memcpy(ct->shape, shape, ndim * sizeof(size_t));

    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i) size *= shape[i];

    size_t elem_size = 0;
    switch (dtype) {
        case IT_DTYPE_F32: elem_size = sizeof(float); break;
        case IT_DTYPE_F64: elem_size = sizeof(double); break;
        case IT_DTYPE_F16:
        case IT_DTYPE_BF16: elem_size = sizeof(uint16_t); break;
        case IT_DTYPE_I8: elem_size = sizeof(int8_t); break;
        default: free(ct->shape); free(ct); return nullptr;
    }

    ct->data = malloc(size * elem_size);
    memcpy(ct->data, data, size * elem_size);
    ct->elem_size = elem_size;
    ct->dtype = dtype;
    ct->scale = 1.0f;
    ct->zero_point = 0.0f;
    ct->owns_memory = true;
    return ct;
}

extern "C" it_tensor* it_tensor_empty(
    const size_t* shape,
    size_t ndim,
    it_dtype_t dtype
) {
    it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor));
    ct->ndim = ndim;
    ct->shape = (size_t*)malloc(ndim * sizeof(size_t));
    memcpy(ct->shape, shape, ndim * sizeof(size_t));

    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i) size *= shape[i];

    size_t elem_size = 0;
    switch (dtype) {
        case IT_DTYPE_F32: elem_size = sizeof(float); break;
        case IT_DTYPE_F64: elem_size = sizeof(double); break;
        case IT_DTYPE_F16:
        case IT_DTYPE_BF16: elem_size = sizeof(uint16_t); break;
        case IT_DTYPE_I8: elem_size = sizeof(int8_t); break;
        default: free(ct->shape); free(ct); return nullptr;
    }

    ct->data = calloc(size, elem_size);
    ct->elem_size = elem_size;
    ct->dtype = dtype;
    ct->scale = 1.0f;
    ct->zero_point = 0.0f;
    ct->owns_memory = true;
    return ct;
}

extern "C" void it_tensor_free(it_tensor* ct) {
    if (ct) {
        if (ct->owns_memory) {
            free(ct->data);
            free(ct->shape);
        }
        free(ct);
    }
}

// ============================================================
// 带量化参数的创建
// ============================================================
extern "C" it_tensor* it_tensor_new_quantized(
    const void* data,
    const size_t* shape,
    size_t ndim,
    it_dtype_t dtype,
    float scale,
    float zero_point
) {
    it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor));
    ct->ndim = ndim;
    ct->shape = (size_t*)malloc(ndim * sizeof(size_t));
    memcpy(ct->shape, shape, ndim * sizeof(size_t));

    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i) size *= shape[i];

    size_t elem_size = 0;
    switch (dtype) {
        case IT_DTYPE_F32: elem_size = sizeof(float); break;
        case IT_DTYPE_F64: elem_size = sizeof(double); break;
        case IT_DTYPE_F16:
        case IT_DTYPE_BF16: elem_size = sizeof(uint16_t); break;
        case IT_DTYPE_I8: elem_size = sizeof(int8_t); break;
        default: free(ct->shape); free(ct); return nullptr;
    }

    ct->data = malloc(size * elem_size);
    memcpy(ct->data, data, size * elem_size);
    ct->elem_size = elem_size;
    ct->dtype = dtype;
    ct->scale = scale;
    ct->zero_point = zero_point;
    ct->owns_memory = true;
    return ct;
}

extern "C" it_tensor* it_tensor_empty_quantized(
    const size_t* shape,
    size_t ndim,
    it_dtype_t dtype,
    float scale,
    float zero_point
) {
    it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor));
    ct->ndim = ndim;
    ct->shape = (size_t*)malloc(ndim * sizeof(size_t));
    memcpy(ct->shape, shape, ndim * sizeof(size_t));

    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i) size *= shape[i];

    size_t elem_size = 0;
    switch (dtype) {
        case IT_DTYPE_F32: elem_size = sizeof(float); break;
        case IT_DTYPE_F64: elem_size = sizeof(double); break;
        case IT_DTYPE_F16:
        case IT_DTYPE_BF16: elem_size = sizeof(uint16_t); break;
        case IT_DTYPE_I8: elem_size = sizeof(int8_t); break;
        default: free(ct->shape); free(ct); return nullptr;
    }

    ct->data = calloc(size, elem_size);
    ct->elem_size = elem_size;
    ct->dtype = dtype;
    ct->scale = scale;
    ct->zero_point = zero_point;
    ct->owns_memory = true;
    return ct;
}

extern "C" void it_tensor_set_quant_params(
    it_tensor* tensor,
    float scale,
    float zero_point
) {
    if (tensor) {
        tensor->scale = scale;
        tensor->zero_point = zero_point;
    }
}

// ============================================================
// Tensor 属性
// ============================================================
extern "C" size_t it_tensor_ndim(const it_tensor* tensor) {
    return tensor->ndim;
}

extern "C" const size_t* it_tensor_shape(const it_tensor* tensor) {
    return tensor->shape;
}

extern "C" const void* it_tensor_data(const it_tensor* tensor) {
    return tensor->data;
}

extern "C" void* it_tensor_mutable_data(it_tensor* tensor) {
    return tensor->data;
}

extern "C" it_dtype_t it_tensor_dtype(const it_tensor* tensor) {
    return tensor->dtype;
}

extern "C" size_t it_tensor_size(const it_tensor* tensor) {
    size_t n = 1;
    for (size_t i = 0; i < tensor->ndim; ++i) n *= tensor->shape[i];
    return n;
}

extern "C" size_t it_tensor_elem_size(const it_tensor* tensor) {
    return tensor->elem_size;
}

extern "C" float it_tensor_scale(const it_tensor* tensor) {
    return tensor->scale;
}

extern "C" float it_tensor_zero_point(const it_tensor* tensor) {
    return tensor->zero_point;
}

// ============================================================
// 宏：二元浮点算子（a, b 两个参数）
// ============================================================
#define DISPATCH_FP_BINARY(op) \
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
        default: \
            return nullptr; \
    }

// ============================================================
// 宏：一元浮点算子（参数名 x）
// ============================================================
#define DISPATCH_FP_UNARY(op, x) \
    switch (x->dtype) { \
        case IT_DTYPE_F32: { \
            auto t = to_cpp_tensor<F32>(x); \
            auto result = op(t); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto t = to_cpp_tensor<F64>(x); \
            auto result = op(t); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto t = to_cpp_tensor<F16>(x); \
            auto result = op(t); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto t = to_cpp_tensor<BF16>(x); \
            auto result = op(t); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: \
            return nullptr; \
    }

// ============================================================
// 宏：一元浮点算子带额外参数（参数名 x）
// ============================================================
#define DISPATCH_FP_UNARY_WITH(op, x, arg) \
    switch (x->dtype) { \
        case IT_DTYPE_F32: { \
            auto t = to_cpp_tensor<F32>(x); \
            auto result = op(t, arg); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto t = to_cpp_tensor<F64>(x); \
            auto result = op(t, arg); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto t = to_cpp_tensor<F16>(x); \
            auto result = op(t, arg); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto t = to_cpp_tensor<BF16>(x); \
            auto result = op(t, arg); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: \
            return nullptr; \
    }

// ============================================================
// 数学算子
// ============================================================
extern "C" it_tensor* it_add(const it_tensor* a, const it_tensor* b) {
    if (a->dtype == IT_DTYPE_I8 && b->dtype == IT_DTYPE_I8) {
        auto ta = to_cpp_tensor_with_params<I8>(a);
        auto tb = to_cpp_tensor_with_params<I8>(b);
        auto result = quantized_add<I8>(ta, tb);
        return from_cpp_tensor(result, IT_DTYPE_I8);
    }
    DISPATCH_FP_BINARY(add);
    return nullptr;
}

extern "C" it_tensor* it_add_scalar(const it_tensor* a, float scalar) {
    DISPATCH_FP_UNARY_WITH(add_scalar, a, scalar);
    return nullptr;
}

extern "C" it_tensor* it_add_n(const it_tensor** tensors, size_t n) {
    if (n == 0) return nullptr;

    it_dtype_t dtype = tensors[0]->dtype;

    switch (dtype) {
        case IT_DTYPE_F32: {
            std::vector<Tensor<F32>> temp;
            for (size_t i = 0; i < n; ++i) {
                temp.push_back(to_cpp_tensor<F32>(tensors[i]));
            }
            std::vector<const Tensor<F32>*> ptrs;
            for (auto& t : temp) ptrs.push_back(&t);
            auto result = add_n<F32>(ptrs);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            std::vector<Tensor<F64>> temp;
            for (size_t i = 0; i < n; ++i) {
                temp.push_back(to_cpp_tensor<F64>(tensors[i]));
            }
            std::vector<const Tensor<F64>*> ptrs;
            for (auto& t : temp) ptrs.push_back(&t);
            auto result = add_n<F64>(ptrs);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            std::vector<Tensor<F16>> temp;
            for (size_t i = 0; i < n; ++i) {
                temp.push_back(to_cpp_tensor<F16>(tensors[i]));
            }
            std::vector<const Tensor<F16>*> ptrs;
            for (auto& t : temp) ptrs.push_back(&t);
            auto result = add_n<F16>(ptrs);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            std::vector<Tensor<BF16>> temp;
            for (size_t i = 0; i < n; ++i) {
                temp.push_back(to_cpp_tensor<BF16>(tensors[i]));
            }
            std::vector<const Tensor<BF16>*> ptrs;
            for (auto& t : temp) ptrs.push_back(&t);
            auto result = add_n<BF16>(ptrs);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default:
            return nullptr;
    }
}

extern "C" it_tensor* it_matmul(const it_tensor* a, const it_tensor* b) {
    if (a->dtype == IT_DTYPE_I8 && b->dtype == IT_DTYPE_I8) {
        auto ta = to_cpp_tensor_with_params<I8>(a);
        auto tb = to_cpp_tensor_with_params<I8>(b);
        auto result = quantized_matmul<I8>(ta, tb);
        return from_cpp_tensor(result, IT_DTYPE_I8);
    }
    DISPATCH_FP_BINARY(matmul);
    return nullptr;
}

extern "C" it_tensor* it_batch_matmul(const it_tensor* a, const it_tensor* b) {
    DISPATCH_FP_BINARY(batch_matmul);
    return nullptr;
}

extern "C" it_tensor* it_vec_matmul(const it_tensor* vec, const it_tensor* mat) {
    if (vec->dtype == IT_DTYPE_I8 && mat->dtype == IT_DTYPE_I8) {
        auto tv = to_cpp_tensor_with_params<I8>(vec);
        auto tm = to_cpp_tensor_with_params<I8>(mat);
        auto result = quantized_vec_matmul<I8>(tv, tm);
        return from_cpp_tensor(result, IT_DTYPE_I8);
    }

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
        default:
            return nullptr;
    }
}

extern "C" it_tensor* it_transpose(const it_tensor* a) {
    DISPATCH_FP_UNARY(transpose, a);
    return nullptr;
}

// ============================================================
// 激活函数
// ============================================================
extern "C" it_tensor* it_relu(const it_tensor* input) {
    if (input->dtype == IT_DTYPE_I8) {
        auto t = to_cpp_tensor_with_params<I8>(input);
        auto result = quantized_relu<I8>(t);
        return from_cpp_tensor(result, IT_DTYPE_I8);
    }
    DISPATCH_FP_UNARY(relu, input);
    return nullptr;
}

extern "C" it_tensor* it_leaky_relu(const it_tensor* input, float alpha) {
    if (input->dtype == IT_DTYPE_I8) {
        auto t = to_cpp_tensor_with_params<I8>(input);
        auto result = quantized_leaky_relu<I8>(t, alpha);
        return from_cpp_tensor(result, IT_DTYPE_I8);
    }
    DISPATCH_FP_UNARY_WITH(leaky_relu, input, alpha);
    return nullptr;
}

extern "C" it_tensor* it_elu(const it_tensor* input, float alpha) {
    if (input->dtype == IT_DTYPE_I8) {
        auto t = to_cpp_tensor_with_params<I8>(input);
        auto result = quantized_elu<I8>(t, alpha);
        return from_cpp_tensor(result, IT_DTYPE_I8);
    }
    DISPATCH_FP_UNARY_WITH(elu, input, alpha);
    return nullptr;
}

extern "C" it_tensor* it_gelu(const it_tensor* input) {
    if (input->dtype == IT_DTYPE_I8) {
        auto t = to_cpp_tensor_with_params<I8>(input);
        auto result = quantized_gelu<I8>(t);
        return from_cpp_tensor(result, IT_DTYPE_I8);
    }
    DISPATCH_FP_UNARY(gelu, input);
    return nullptr;
}

// ============================================================
// 归一化
// ============================================================
extern "C" it_tensor* it_softmax(const it_tensor* input, int dim) {
    if (input->dtype == IT_DTYPE_I8) {
        auto t = to_cpp_tensor_with_params<I8>(input);
        auto result = quantized_softmax<I8>(t, dim);
        return from_cpp_tensor(result, IT_DTYPE_I8);
    }
    DISPATCH_FP_UNARY_WITH(softmax, input, dim);
    return nullptr;
}

extern "C" it_tensor* it_log_softmax(const it_tensor* input, int dim) {
    if (input->dtype == IT_DTYPE_I8) {
        auto t = to_cpp_tensor_with_params<I8>(input);
        auto result = quantized_log_softmax<I8>(t, dim);
        return from_cpp_tensor(result, IT_DTYPE_I8);
    }
    DISPATCH_FP_UNARY_WITH(log_softmax, input, dim);
    return nullptr;
}

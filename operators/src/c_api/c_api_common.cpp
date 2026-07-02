#include "infer_train/c_interface.h"
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cstdlib>
#include <cstring>

using namespace infer_train;

// ============================================================
// Tensor 结构（不透明类型）
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
// 辅助转换函数
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

extern "C" it_tensor* it_tensor_new_quantized(
    const void* data,
    const size_t* shape,
    size_t ndim,
    it_dtype_t dtype,
    float scale,
    float zero_point
) {
    it_tensor* ct = it_tensor_new(data, shape, ndim, dtype);
    if (ct) {
        ct->scale = scale;
        ct->zero_point = zero_point;
    }
    return ct;
}

extern "C" it_tensor* it_tensor_empty_quantized(
    const size_t* shape,
    size_t ndim,
    it_dtype_t dtype,
    float scale,
    float zero_point
) {
    it_tensor* ct = it_tensor_empty(shape, ndim, dtype);
    if (ct) {
        ct->scale = scale;
        ct->zero_point = zero_point;
    }
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

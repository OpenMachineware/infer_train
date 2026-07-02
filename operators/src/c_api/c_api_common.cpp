#include "c_api_common.h"

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

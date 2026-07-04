#pragma once
#include "infer_train/c_interface.h"
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace infer_train;

// ============================================================
// C Tensor 结构（不透明类型的实际定义）
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
// 转换辅助函数
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

// 特化：从 std::vector 创建 it_tensor（用于 where/embedding 等）
template<typename T>
it_tensor* from_vector_to_tensor(
    const std::vector<T>& data,
    const std::vector<size_t>& shape,
    it_dtype_t dtype
) {
    it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor));
    ct->ndim = shape.size();
    ct->shape = (size_t*)malloc(ct->ndim * sizeof(size_t));
    for (size_t i = 0; i < ct->ndim; ++i) {
        ct->shape[i] = shape[i];
    }
    size_t bytes = data.size() * sizeof(T);
    ct->data = malloc(bytes);
    memcpy(ct->data, data.data(), bytes);
    ct->elem_size = sizeof(T);
    ct->dtype = dtype;
    ct->scale = 1.0f;
    ct->zero_point = 0.0f;
    ct->owns_memory = true;
    return ct;
}

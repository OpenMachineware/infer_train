#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cstdlib>
#include <cstring>

using namespace infer_train;

// ============================================================
// C Tensor 结构
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
Tensor<T> to_cpp_tensor(const it_tensor* ct);

template<typename T>
Tensor<T> to_cpp_tensor_with_params(const it_tensor* ct);

template<typename T>
it_tensor* from_cpp_tensor(const Tensor<T>& t, it_dtype_t dtype);

#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// 数据类型枚举
// ============================================================
typedef enum {
    IT_DTYPE_F32 = 0,
    IT_DTYPE_F64 = 1,
    IT_DTYPE_F16 = 2,
    IT_DTYPE_BF16 = 3,
    IT_DTYPE_I8 = 4,
} it_dtype_t;

// ============================================================
// 不透明类型
// ============================================================
typedef struct it_tensor it_tensor_t;

// ============================================================
// Tensor 生命周期
// ============================================================
it_tensor_t* it_tensor_new(
    const void* data,
    const size_t* shape,
    size_t ndim,
    it_dtype_t dtype
);

it_tensor_t* it_tensor_empty(
    const size_t* shape,
    size_t ndim,
    it_dtype_t dtype
);

// ============================================================
// 带量化参数的创建
// ============================================================
it_tensor_t* it_tensor_new_quantized(
    const void* data,
    const size_t* shape,
    size_t ndim,
    it_dtype_t dtype,
    float scale,
    float zero_point
);

it_tensor_t* it_tensor_empty_quantized(
    const size_t* shape,
    size_t ndim,
    it_dtype_t dtype,
    float scale,
    float zero_point
);

void it_tensor_set_quant_params(
    it_tensor_t* tensor,
    float scale,
    float zero_point
);

void it_tensor_free(it_tensor_t* tensor);

// ============================================================
// Tensor 属性
// ============================================================
size_t it_tensor_ndim(const it_tensor_t* tensor);
const size_t* it_tensor_shape(const it_tensor_t* tensor);
const void* it_tensor_data(const it_tensor_t* tensor);
void* it_tensor_mutable_data(it_tensor_t* tensor);
it_dtype_t it_tensor_dtype(const it_tensor_t* tensor);
size_t it_tensor_size(const it_tensor_t* tensor);
size_t it_tensor_elem_size(const it_tensor_t* tensor);

// 量化张量特有
float it_tensor_scale(const it_tensor_t* tensor);
float it_tensor_zero_point(const it_tensor_t* tensor);

// ============================================================
// 数学算子
// ============================================================
it_tensor_t* it_add(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_add_scalar(const it_tensor_t* a, float scalar);
it_tensor_t* it_add_n(const it_tensor_t** tensors, size_t n);

it_tensor_t* it_matmul(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_batch_matmul(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_vec_matmul(const it_tensor_t* vec, const it_tensor_t* mat);
it_tensor_t* it_transpose(const it_tensor_t* a);

// ============================================================
// 激活函数
// ============================================================
it_tensor_t* it_relu(const it_tensor_t* input);
it_tensor_t* it_leaky_relu(const it_tensor_t* input, float alpha);
it_tensor_t* it_elu(const it_tensor_t* input, float alpha);
it_tensor_t* it_gelu(const it_tensor_t* input);

// ============================================================
// 归一化
// ============================================================
it_tensor_t* it_softmax(const it_tensor_t* input, int dim);
it_tensor_t* it_log_softmax(const it_tensor_t* input, int dim);

#ifdef __cplusplus
}
#endif

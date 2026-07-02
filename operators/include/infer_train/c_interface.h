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
float it_tensor_scale(const it_tensor_t* tensor);
float it_tensor_zero_point(const it_tensor_t* tensor);

// ============================================================
// 数学算子（浮点）
// ============================================================
it_tensor_t* it_add(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_sub(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_mul(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_div(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_pow(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_exp(const it_tensor_t* input);
it_tensor_t* it_sqrt(const it_tensor_t* input);
it_tensor_t* it_abs(const it_tensor_t* input);
it_tensor_t* it_neg(const it_tensor_t* input);
it_tensor_t* it_clamp(const it_tensor_t* input, float min_val, float max_val);

it_tensor_t* it_add_scalar(const it_tensor_t* a, float scalar);
it_tensor_t* it_sub_scalar(const it_tensor_t* a, float scalar);
it_tensor_t* it_mul_scalar(const it_tensor_t* a, float scalar);
it_tensor_t* it_div_scalar(const it_tensor_t* a, float scalar);
it_tensor_t* it_pow_scalar(const it_tensor_t* a, float exponent);
it_tensor_t* it_scalar_sub(float scalar, const it_tensor_t* a);
it_tensor_t* it_scalar_div(float scalar, const it_tensor_t* a);

// ============================================================
// 规约算子
// ============================================================
it_tensor_t* it_sum(const it_tensor_t* input, const int* dims, size_t ndim, int keepdim);
it_tensor_t* it_mean(const it_tensor_t* input, const int* dims, size_t ndim, int keepdim);
it_tensor_t* it_max_all(const it_tensor_t* input);
it_tensor_t* it_min_all(const it_tensor_t* input);

// ============================================================
// 矩阵算子
// ============================================================
it_tensor_t* it_matmul(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_batch_matmul(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_vec_matmul(const it_tensor_t* vec, const it_tensor_t* mat);
it_tensor_t* it_transpose(const it_tensor_t* a);

// ============================================================
// 量化数学算子
// ============================================================
it_tensor_t* it_quantized_add(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_quantized_sub(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_quantized_mul(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_quantized_div(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_quantized_exp(const it_tensor_t* input);
it_tensor_t* it_quantized_sqrt(const it_tensor_t* input);
it_tensor_t* it_quantized_abs(const it_tensor_t* input);
it_tensor_t* it_quantized_neg(const it_tensor_t* input);
it_tensor_t* it_quantized_clamp(const it_tensor_t* input, float min_val, float max_val);

// ============================================================
// 量化规约
// ============================================================
it_tensor_t* it_quantized_sum(const it_tensor_t* input, const int* dims, size_t ndim, int keepdim);
it_tensor_t* it_quantized_mean(const it_tensor_t* input, const int* dims, size_t ndim, int keepdim);
it_tensor_t* it_quantized_max_all(const it_tensor_t* input);
it_tensor_t* it_quantized_min_all(const it_tensor_t* input);

// ============================================================
// 量化矩阵算子
// ============================================================
it_tensor_t* it_quantized_matmul(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_quantized_vec_matmul(const it_tensor_t* vec, const it_tensor_t* mat);
it_tensor_t* it_quantized_transpose(const it_tensor_t* a);

// ============================================================
// NN 算子（浮点）
// ============================================================
it_tensor_t* it_conv2d(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    int stride,
    int padding,
    int dilation,
    int groups
);

it_tensor_t* it_maxpool2d(
    const it_tensor_t* input,
    int kernel_size,
    int stride,
    int padding
);

it_tensor_t* it_avgpool2d(
    const it_tensor_t* input,
    int kernel_size,
    int stride,
    int padding
);

it_tensor_t* it_batchnorm2d_inference(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    const it_tensor_t* running_mean,
    const it_tensor_t* running_var,
    float eps
);

it_tensor_t* it_layernorm(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    float eps
);

it_tensor_t* it_linear(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias
);

it_tensor_t* it_conv1d(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    int stride,
    int padding,
    int dilation,
    int groups
);

it_tensor_t* it_conv3d(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    int stride,
    int padding,
    int dilation,
    int groups
);

it_tensor_t* it_maxpool1d(
    const it_tensor_t* input,
    int kernel_size,
    int stride,
    int padding
);

it_tensor_t* it_maxpool3d(
    const it_tensor_t* input,
    int kernel_size,
    int stride,
    int padding
);

it_tensor_t* it_avgpool1d(
    const it_tensor_t* input,
    int kernel_size,
    int stride,
    int padding
);

it_tensor_t* it_avgpool3d(
    const it_tensor_t* input,
    int kernel_size,
    int stride,
    int padding
);

it_tensor_t* it_batchnorm1d_inference(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    const it_tensor_t* running_mean,
    const it_tensor_t* running_var,
    float eps
);

it_tensor_t* it_instancenorm2d(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    float eps
);

it_tensor_t* it_groupnorm(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    int num_groups,
    float eps
);

it_tensor_t* it_dropout(
    const it_tensor_t* input,
    float p
);

it_tensor_t* it_embedding(
    const it_tensor_t* indices,
    const it_tensor_t* weight,
    int padding_idx
);

// ============================================================
// 激活函数（浮点）
// ============================================================
it_tensor_t* it_relu(const it_tensor_t* input);
it_tensor_t* it_leaky_relu(const it_tensor_t* input, float alpha);
it_tensor_t* it_elu(const it_tensor_t* input, float alpha);
it_tensor_t* it_gelu(const it_tensor_t* input);
it_tensor_t* it_sigmoid(const it_tensor_t* input);
it_tensor_t* it_tanh(const it_tensor_t* input);
it_tensor_t* it_silu(const it_tensor_t* input);
it_tensor_t* it_hard_swish(const it_tensor_t* input);
it_tensor_t* it_hard_sigmoid(const it_tensor_t* input);
it_tensor_t* it_softmax(const it_tensor_t* input, int dim);
it_tensor_t* it_log_softmax(const it_tensor_t* input, int dim);

// ============================================================
// 量化 NN 算子
// ============================================================
it_tensor_t* it_quantized_conv2d(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    int stride,
    int padding,
    int dilation,
    int groups
);

it_tensor_t* it_quantized_maxpool2d(
    const it_tensor_t* input,
    int kernel_size,
    int stride,
    int padding
);

it_tensor_t* it_quantized_avgpool2d(
    const it_tensor_t* input,
    int kernel_size,
    int stride,
    int padding
);

it_tensor_t* it_quantized_batchnorm2d_inference(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    const it_tensor_t* running_mean,
    const it_tensor_t* running_var,
    float eps
);

it_tensor_t* it_quantized_layernorm(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    float eps
);

it_tensor_t* it_quantized_linear(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias
);

it_tensor_t* it_quantized_conv1d(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    int stride,
    int padding,
    int dilation,
    int groups
);

it_tensor_t* it_quantized_conv3d(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    int stride,
    int padding,
    int dilation,
    int groups
);

it_tensor_t* it_quantized_maxpool1d(
    const it_tensor_t* input,
    int kernel_size,
    int stride,
    int padding
);

it_tensor_t* it_quantized_maxpool3d(
    const it_tensor_t* input,
    int kernel_size,
    int stride,
    int padding
);

it_tensor_t* it_quantized_avgpool1d(
    const it_tensor_t* input,
    int kernel_size,
    int stride,
    int padding
);

it_tensor_t* it_quantized_avgpool3d(
    const it_tensor_t* input,
    int kernel_size,
    int stride,
    int padding
);

it_tensor_t* it_quantized_batchnorm1d_inference(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    const it_tensor_t* running_mean,
    const it_tensor_t* running_var,
    float eps
);

it_tensor_t* it_quantized_instancenorm2d(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    float eps
);

it_tensor_t* it_quantized_groupnorm(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    int num_groups,
    float eps
);

it_tensor_t* it_quantized_dropout(
    const it_tensor_t* input,
    float p
);

it_tensor_t* it_quantized_embedding(
    const it_tensor_t* indices,
    const it_tensor_t* weight,
    int padding_idx
);

// ============================================================
// 量化激活函数
// ============================================================
it_tensor_t* it_quantized_relu(const it_tensor_t* input);
it_tensor_t* it_quantized_leaky_relu(const it_tensor_t* input, float alpha);
it_tensor_t* it_quantized_elu(const it_tensor_t* input, float alpha);
it_tensor_t* it_quantized_gelu(const it_tensor_t* input);
it_tensor_t* it_quantized_sigmoid(const it_tensor_t* input);
it_tensor_t* it_quantized_tanh(const it_tensor_t* input);
it_tensor_t* it_quantized_silu(const it_tensor_t* input);
it_tensor_t* it_quantized_hard_swish(const it_tensor_t* input);
it_tensor_t* it_quantized_hard_sigmoid(const it_tensor_t* input);
it_tensor_t* it_quantized_softmax(const it_tensor_t* input, int dim);
it_tensor_t* it_quantized_log_softmax(const it_tensor_t* input, int dim);


#ifdef __cplusplus
}
#endif

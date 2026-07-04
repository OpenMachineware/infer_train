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
    IT_DTYPE_I64 = 5,
    IT_DTYPE_U8 = 6,
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
it_tensor_t* it_log(const it_tensor_t* input);
it_tensor_t* it_log2(const it_tensor_t* input);
it_tensor_t* it_log10(const it_tensor_t* input);
it_tensor_t* it_abs(const it_tensor_t* input);
it_tensor_t* it_neg(const it_tensor_t* input);
it_tensor_t* it_clamp(const it_tensor_t* input, float min_val, float max_val);
it_tensor_t* it_floor(const it_tensor_t* input);
it_tensor_t* it_ceil(const it_tensor_t* input);
it_tensor_t* it_round(const it_tensor_t* input);
it_tensor_t* it_eq(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_ne(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_gt(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_lt(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_ge(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_le(const it_tensor_t* a, const it_tensor_t* b);

it_tensor_t* it_add_scalar(const it_tensor_t* a, float scalar);
it_tensor_t* it_sub_scalar(const it_tensor_t* a, float scalar);
it_tensor_t* it_mul_scalar(const it_tensor_t* a, float scalar);
it_tensor_t* it_div_scalar(const it_tensor_t* a, float scalar);
it_tensor_t* it_pow_scalar(const it_tensor_t* a, float exponent);
it_tensor_t* it_scalar_sub(float scalar, const it_tensor_t* a);
it_tensor_t* it_scalar_div(float scalar, const it_tensor_t* a);

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
// 规约算子
// ============================================================
it_tensor_t* it_sum(const it_tensor_t* input, const int* dims, size_t ndim, int keepdim);
it_tensor_t* it_mean(const it_tensor_t* input, const int* dims, size_t ndim, int keepdim);
it_tensor_t* it_max_all(const it_tensor_t* input);
it_tensor_t* it_min_all(const it_tensor_t* input);
it_tensor_t* it_prod_all(const it_tensor_t* input);
it_tensor_t* it_var(const it_tensor_t* input, int unbiased);
it_tensor_t* it_std(const it_tensor_t* input, int unbiased);
it_tensor_t* it_argmax(const it_tensor_t* input);
it_tensor_t* it_argmin(const it_tensor_t* input);
void it_topk(
    const it_tensor_t* input,
    size_t k,
    int dim,
    int largest,
    it_tensor_t** values,
    it_tensor_t** indices
);

it_tensor_t* it_quantized_sum(const it_tensor_t* input, const int* dims, size_t ndim, int keepdim);
it_tensor_t* it_quantized_mean(const it_tensor_t* input, const int* dims, size_t ndim, int keepdim);
it_tensor_t* it_quantized_max_all(const it_tensor_t* input);
it_tensor_t* it_quantized_min_all(const it_tensor_t* input);


// ============================================================
// 矩阵算子
// ============================================================
it_tensor_t* it_matmul(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_batch_matmul(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_vec_matmul(const it_tensor_t* vec, const it_tensor_t* mat);
it_tensor_t* it_transpose(const it_tensor_t* input);

it_tensor_t* it_quantized_matmul(const it_tensor_t* a, const it_tensor_t* b);
it_tensor_t* it_quantized_vec_matmul(const it_tensor_t* vec, const it_tensor_t* mat);
it_tensor_t* it_quantized_transpose(const it_tensor_t* input);

// ============================================================
// 张量操作
// ============================================================
it_tensor_t* it_slice(const it_tensor_t* input, int dim, int start, int end, int step);
it_tensor_t* it_cat(const it_tensor_t** tensors, size_t n, int dim);
it_tensor_t* it_gather(const it_tensor_t* input, const int64_t* indices, size_t indices_len, const size_t* indices_shape, size_t indices_ndim, int dim);
it_tensor_t* it_scatter(
    const it_tensor_t* input,
    const int64_t* indices,
    size_t indices_len,
    const size_t* indices_shape,
    size_t indices_ndim,
    const it_tensor_t* src,
    int dim
);
void it_sort(
    const it_tensor_t* input,
    int dim,
    int ascending,
    it_tensor_t** values,
    it_tensor_t** indices
);

// ============================================================
// 复杂算子
// ============================================================
it_tensor_t* it_cumsum(const it_tensor_t* input, int dim);
it_tensor_t* it_cumprod(const it_tensor_t* input, int dim);
it_tensor_t* it_where(
    const uint8_t* condition,
    const size_t* condition_shape,
    size_t condition_ndim,
    const it_tensor_t* true_val,
    const it_tensor_t* false_val
);

// ============================================================
// NN 算子（浮点）
// ============================================================
it_tensor_t* it_conv1d(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    int stride,
    int padding,
    int dilation,
    int groups
);
it_tensor_t* it_conv2d(
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

it_tensor_t* it_maxpool1d(const it_tensor_t* input, int kernel_size, int stride, int padding);
it_tensor_t* it_maxpool2d(const it_tensor_t* input, int kernel_size, int stride, int padding);
it_tensor_t* it_maxpool3d(const it_tensor_t* input, int kernel_size, int stride, int padding);
it_tensor_t* it_avgpool1d(const it_tensor_t* input, int kernel_size, int stride, int padding);
it_tensor_t* it_avgpool2d(const it_tensor_t* input, int kernel_size, int stride, int padding);
it_tensor_t* it_avgpool3d(const it_tensor_t* input, int kernel_size, int stride, int padding);

it_tensor_t* it_batchnorm1d_inference(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    const it_tensor_t* running_mean,
    const it_tensor_t* running_var,
    float eps
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
it_tensor_t* it_rmsnorm(
    const it_tensor_t* input,
    const it_tensor_t* weight,
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

it_tensor_t* it_linear(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias
);
it_tensor_t* it_embedding(
    const int64_t* indices,
    size_t num_indices,
    const it_tensor_t* weight,
    int padding_idx
);
it_tensor_t* it_dropout(
    const it_tensor_t* input,
    float p,
    int training,
    uint32_t seed
);

// ============================================================
// NN 算子（量化）
// ============================================================
it_tensor_t* it_quantized_conv1d(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    int stride,
    int padding,
    int dilation,
    int groups
);
it_tensor_t* it_quantized_conv2d(
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
it_tensor_t* it_quantized_maxpool1d(const it_tensor_t* input, int kernel_size, int stride, int padding);
it_tensor_t* it_quantized_maxpool2d(const it_tensor_t* input, int kernel_size, int stride, int padding);
it_tensor_t* it_quantized_maxpool3d(const it_tensor_t* input, int kernel_size, int stride, int padding);
it_tensor_t* it_quantized_avgpool1d(const it_tensor_t* input, int kernel_size, int stride, int padding);
it_tensor_t* it_quantized_avgpool2d(const it_tensor_t* input, int kernel_size, int stride, int padding);
it_tensor_t* it_quantized_avgpool3d(const it_tensor_t* input, int kernel_size, int stride, int padding);
it_tensor_t* it_quantized_batchnorm1d_inference(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias,
    const it_tensor_t* running_mean,
    const it_tensor_t* running_var,
    float eps
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
it_tensor_t* it_quantized_rmsnorm(
    const it_tensor_t* input,
    const it_tensor_t* weight,
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

it_tensor_t* it_quantized_linear(
    const it_tensor_t* input,
    const it_tensor_t* weight,
    const it_tensor_t* bias
);
it_tensor_t* it_quantized_embedding(
    const int64_t* indices,
    size_t num_indices,
    const it_tensor_t* weight,
    int padding_idx
);
it_tensor_t* it_quantized_dropout(const it_tensor_t* input, float p);

// ============================================================
// 激活函数（浮点）
// ============================================================
it_tensor_t* it_relu(const it_tensor_t* input);
it_tensor_t* it_leaky_relu(const it_tensor_t* input, float alpha);
it_tensor_t* it_elu(const it_tensor_t* input, float alpha);
it_tensor_t* it_gelu(const it_tensor_t* input);
it_tensor_t* it_relu6(const it_tensor_t* input);
it_tensor_t* it_sigmoid(const it_tensor_t* input);
it_tensor_t* it_tanh(const it_tensor_t* input);
it_tensor_t* it_silu(const it_tensor_t* input);
it_tensor_t* it_hard_swish(const it_tensor_t* input);
it_tensor_t* it_hard_sigmoid(const it_tensor_t* input);
it_tensor_t* it_softplus(const it_tensor_t* input, float beta, float threshold);
it_tensor_t* it_softshrink(const it_tensor_t* input, float lambda);
it_tensor_t* it_celu(const it_tensor_t* input, float alpha);
it_tensor_t* it_softmax(const it_tensor_t* input, int dim);
it_tensor_t* it_log_softmax(const it_tensor_t* input, int dim);

// ============================================================
// 激活函数（量化）
// ============================================================
it_tensor_t* it_quantized_relu(const it_tensor_t* input);
it_tensor_t* it_quantized_leaky_relu(const it_tensor_t* input, float alpha);
it_tensor_t* it_quantized_elu(const it_tensor_t* input, float alpha);
it_tensor_t* it_quantized_gelu(const it_tensor_t* input);
it_tensor_t* it_quantized_relu6(const it_tensor_t* input);
it_tensor_t* it_quantized_sigmoid(const it_tensor_t* input);
it_tensor_t* it_quantized_tanh(const it_tensor_t* input);
it_tensor_t* it_quantized_silu(const it_tensor_t* input);
it_tensor_t* it_quantized_hard_swish(const it_tensor_t* input);
it_tensor_t* it_quantized_hard_sigmoid(const it_tensor_t* input);
it_tensor_t* it_quantized_softmax(const it_tensor_t* input, int dim);
it_tensor_t* it_quantized_log_softmax(const it_tensor_t* input, int dim);

// ============================================================
// 注意力算子
// ============================================================
it_tensor_t* it_scaled_dot_product_attention(
    const it_tensor_t* query,
    const it_tensor_t* key,
    const it_tensor_t* value,
    const it_tensor_t* mask,
    float scale,
    int is_causal,
    float dropout_p
);
it_tensor_t* it_multi_head_attention(
    const it_tensor_t* query,
    const it_tensor_t* key,
    const it_tensor_t* value,
    const it_tensor_t* mask,
    int num_heads,
    float scale,
    int is_causal,
    float dropout_p
);
it_tensor_t* it_rotary_embedding(
    const it_tensor_t* x,
    const it_tensor_t* cos,
    const it_tensor_t* sin
);

it_tensor_t* it_quantized_scaled_dot_product_attention(
    const it_tensor_t* query,
    const it_tensor_t* key,
    const it_tensor_t* value,
    const it_tensor_t* mask,
    float scale,
    int is_causal,
    float dropout_p
);
it_tensor_t* it_quantized_multi_head_attention(
    const it_tensor_t* query,
    const it_tensor_t* key,
    const it_tensor_t* value,
    const it_tensor_t* mask,
    int num_heads,
    float scale,
    int is_causal,
    float dropout_p
);
it_tensor_t* it_quantized_rotary_embedding(
    const it_tensor_t* x,
    const it_tensor_t* cos,
    const it_tensor_t* sin
);

// ============================================================
// 损失函数
// ============================================================
it_tensor_t* it_cross_entropy_loss(
    const it_tensor_t* input,
    const int64_t* target,
    size_t batch_size,
    int reduction
);
it_tensor_t* it_mse_loss(
    const it_tensor_t* input,
    const it_tensor_t* target,
    int reduction
);
it_tensor_t* it_l1_loss(
    const it_tensor_t* input,
    const it_tensor_t* target,
    int reduction
);
it_tensor_t* it_bce_loss(
    const it_tensor_t* input,
    const it_tensor_t* target,
    int reduction,
    float eps
);

// ============================================================
// 优化器
// ============================================================
void it_sgd_update(
    it_tensor_t** params,
    it_tensor_t** grads,
    size_t num_params,
    float lr,
    float momentum,
    float weight_decay,
    int nesterov
);

void it_adam_update(
    it_tensor_t** params,
    it_tensor_t** grads,
    size_t num_params,
    void* state,  // 不透明状态
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
);

void it_adamw_update(
    it_tensor_t** params,
    it_tensor_t** grads,
    size_t num_params,
    void* state,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
);

// ============================================================
// 优化器状态管理
// ============================================================
void* it_adam_state_new(
    size_t num_params,
    const size_t* param_shapes,
    const size_t* param_ndims
);

void it_adam_state_free(void* state);

void it_adam_update(
    it_tensor_t** params,
    it_tensor_t** grads,
    size_t num_params,
    void* state,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
);

void* it_adamw_state_new(
    size_t num_params,
    const size_t* param_shapes,
    const size_t* param_ndims
);

void it_adamw_state_free(void* state);

void it_adamw_update(
    it_tensor_t** params,
    it_tensor_t** grads,
    size_t num_params,
    void* state,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay
);

#ifdef __cplusplus
}
#endif

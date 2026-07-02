#include "c_api_common.h"
#include "infer_train/nn_q.hpp"

using namespace infer_train;

// ============================================================
// 量化卷积
// ============================================================
extern "C" it_tensor* it_quantized_conv1d(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    if (input->dtype != IT_DTYPE_I8 || weight->dtype != IT_DTYPE_I8) return nullptr;
    auto ti = to_cpp_tensor_with_params<I8>(input);
    auto tw = to_cpp_tensor_with_params<I8>(weight);
    Tensor<I8> tb; bool has_bias = (bias != nullptr);
    if (has_bias) tb = to_cpp_tensor_with_params<I8>(bias);
    auto result = quantized_conv1d(ti, tw, has_bias ? &tb : nullptr, stride, padding, dilation, groups);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

extern "C" it_tensor* it_quantized_conv2d(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    if (input->dtype != IT_DTYPE_I8 || weight->dtype != IT_DTYPE_I8) return nullptr;
    auto ti = to_cpp_tensor_with_params<I8>(input);
    auto tw = to_cpp_tensor_with_params<I8>(weight);
    Tensor<I8> tb; bool has_bias = (bias != nullptr);
    if (has_bias) tb = to_cpp_tensor_with_params<I8>(bias);
    auto result = quantized_conv2d(ti, tw, has_bias ? &tb : nullptr, stride, padding, dilation, groups);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

extern "C" it_tensor* it_quantized_conv3d(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    if (input->dtype != IT_DTYPE_I8 || weight->dtype != IT_DTYPE_I8) return nullptr;
    auto ti = to_cpp_tensor_with_params<I8>(input);
    auto tw = to_cpp_tensor_with_params<I8>(weight);
    Tensor<I8> tb; bool has_bias = (bias != nullptr);
    if (has_bias) tb = to_cpp_tensor_with_params<I8>(bias);
    auto result = quantized_conv3d(ti, tw, has_bias ? &tb : nullptr, stride, padding, dilation, groups);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

// ============================================================
// 量化池化
// ============================================================
#define DISPATCH_QUANTIZED_POOL_1D(op) \
if (input->dtype != IT_DTYPE_I8) return nullptr; \
auto t = to_cpp_tensor_with_params<I8>(input); \
auto result = op(t, kernel_size, stride, padding); \
return from_cpp_tensor(result, IT_DTYPE_I8);

#define DISPATCH_QUANTIZED_POOL_2D(op) DISPATCH_QUANTIZED_POOL_1D(op)
#define DISPATCH_QUANTIZED_POOL_3D(op) DISPATCH_QUANTIZED_POOL_1D(op)

extern "C" it_tensor* it_quantized_maxpool1d(const it_tensor* input, int kernel_size, int stride, int padding) {
    DISPATCH_QUANTIZED_POOL_1D(quantized_maxpool1d);
}

extern "C" it_tensor* it_quantized_maxpool2d(const it_tensor* input, int kernel_size, int stride, int padding) {
    DISPATCH_QUANTIZED_POOL_2D(quantized_maxpool2d);
}

extern "C" it_tensor* it_quantized_maxpool3d(const it_tensor* input, int kernel_size, int stride, int padding) {
    DISPATCH_QUANTIZED_POOL_3D(quantized_maxpool3d);
}

extern "C" it_tensor* it_quantized_avgpool1d(const it_tensor* input, int kernel_size, int stride, int padding) {
    DISPATCH_QUANTIZED_POOL_1D(quantized_avgpool1d);
}

extern "C" it_tensor* it_quantized_avgpool2d(const it_tensor* input, int kernel_size, int stride, int padding) {
    DISPATCH_QUANTIZED_POOL_2D(quantized_avgpool2d);
}

extern "C" it_tensor* it_quantized_avgpool3d(const it_tensor* input, int kernel_size, int stride, int padding) {
    DISPATCH_QUANTIZED_POOL_3D(quantized_avgpool3d);
}

// ============================================================
// 量化归一化
// ============================================================
extern "C" it_tensor* it_quantized_batchnorm1d_inference(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    const it_tensor* running_mean,
    const it_tensor* running_var,
    float eps
) {
    if (input->dtype != IT_DTYPE_I8) return nullptr;
    auto ti = to_cpp_tensor_with_params<I8>(input);
    auto tw = to_cpp_tensor_with_params<I8>(weight);
    auto tb = to_cpp_tensor_with_params<I8>(bias);
    auto tm = to_cpp_tensor_with_params<I8>(running_mean);
    auto tv = to_cpp_tensor_with_params<I8>(running_var);
    auto result = quantized_batchnorm1d_inference(ti, tw, tb, tm, tv, eps);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

extern "C" it_tensor* it_quantized_batchnorm2d_inference(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    const it_tensor* running_mean,
    const it_tensor* running_var,
    float eps
) {
    if (input->dtype != IT_DTYPE_I8) return nullptr;
    auto ti = to_cpp_tensor_with_params<I8>(input);
    auto tw = to_cpp_tensor_with_params<I8>(weight);
    auto tb = to_cpp_tensor_with_params<I8>(bias);
    auto tm = to_cpp_tensor_with_params<I8>(running_mean);
    auto tv = to_cpp_tensor_with_params<I8>(running_var);
    auto result = quantized_batchnorm2d_inference(ti, tw, tb, tm, tv, eps);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

extern "C" it_tensor* it_quantized_layernorm(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    float eps
) {
    if (input->dtype != IT_DTYPE_I8) return nullptr;
    auto ti = to_cpp_tensor_with_params<I8>(input);
    auto tw = to_cpp_tensor_with_params<I8>(weight);
    auto tb = to_cpp_tensor_with_params<I8>(bias);
    auto result = quantized_layernorm(ti, tw, tb, eps);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

extern "C" it_tensor* it_quantized_rmsnorm(
    const it_tensor* input,
    const it_tensor* weight,
    float eps
) {
    if (input->dtype != IT_DTYPE_I8) return nullptr;
    auto ti = to_cpp_tensor_with_params<I8>(input);
    auto tw = to_cpp_tensor_with_params<I8>(weight);
    auto result = quantized_rmsnorm(ti, tw, eps);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

// ============================================================
// 量化 linear
// ============================================================
extern "C" it_tensor* it_quantized_linear(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias
) {
    if (input->dtype != IT_DTYPE_I8 || weight->dtype != IT_DTYPE_I8) return nullptr;
    auto ti = to_cpp_tensor_with_params<I8>(input);
    auto tw = to_cpp_tensor_with_params<I8>(weight);
    Tensor<I8> tb; bool has_bias = (bias != nullptr);
    if (has_bias) tb = to_cpp_tensor_with_params<I8>(bias);
    auto result = quantized_linear(ti, tw, has_bias ? &tb : nullptr);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

// ============================================================
// 量化 embedding
// ============================================================
extern "C" it_tensor* it_quantized_embedding(
    const int64_t* indices,
    size_t num_indices,
    const it_tensor* weight,
    int padding_idx
) {
    if (weight->dtype != IT_DTYPE_I8) return nullptr;
    std::vector<int64_t> idx_vec(indices, indices + num_indices);
    auto tw = to_cpp_tensor_with_params<I8>(weight);
    auto result = quantized_embedding(idx_vec, tw, padding_idx);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

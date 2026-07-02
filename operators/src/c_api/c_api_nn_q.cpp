#include "infer_train/c_interface.h"
#include "infer_train/nn_q.hpp"
#include "c_api_common.h"

using namespace infer_train;

// ============================================================
// 量化 NN 算子（只支持 I8）
// ============================================================
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

extern "C" it_tensor* it_quantized_maxpool2d(
    const it_tensor* input,
    int kernel_size,
    int stride,
    int padding
) {
    if (input->dtype != IT_DTYPE_I8) return nullptr;
    auto t = to_cpp_tensor_with_params<I8>(input);
    auto result = quantized_maxpool2d(t, kernel_size, stride, padding);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

extern "C" it_tensor* it_quantized_avgpool2d(
    const it_tensor* input,
    int kernel_size,
    int stride,
    int padding
) {
    if (input->dtype != IT_DTYPE_I8) return nullptr;
    auto t = to_cpp_tensor_with_params<I8>(input);
    auto result = quantized_avgpool2d(t, kernel_size, stride, padding);
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

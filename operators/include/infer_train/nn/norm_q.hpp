#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/nn/norm.hpp"

namespace infer_train {

// ============================================================
// 量化 InstanceNorm2d
// ============================================================
template<typename T>
Tensor<T> quantized_instancenorm2d(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    const Tensor<T>& bias,
    float eps = 1e-5f
) {
    static_assert(is_quantized<T>::value,
                  "quantized_instancenorm2d only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> fp32_weight = dequantize(weight);
    Tensor<F32> fp32_bias = dequantize(bias);

    Tensor<F32> result_fp32 = instancenorm2d<F32>(
        fp32_input, fp32_weight, fp32_bias, eps
    );

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

// ============================================================
// 量化 GroupNorm
// ============================================================
template<typename T>
Tensor<T> quantized_groupnorm(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    const Tensor<T>& bias,
    int num_groups,
    float eps = 1e-5f
) {
    static_assert(is_quantized<T>::value,
                  "quantized_groupnorm only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> fp32_weight = dequantize(weight);
    Tensor<F32> fp32_bias = dequantize(bias);

    Tensor<F32> result_fp32 = groupnorm<F32>(
        fp32_input, fp32_weight, fp32_bias, num_groups, eps
    );

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train

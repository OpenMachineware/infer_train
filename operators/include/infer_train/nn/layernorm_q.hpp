#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/nn/layernorm.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_layernorm(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    const Tensor<T>& bias,
    float eps = 1e-5f
) {
    static_assert(is_quantized<T>::value,
                  "quantized_layernorm only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> fp32_weight = dequantize(weight);
    Tensor<F32> fp32_bias = dequantize(bias);

    Tensor<F32> result_fp32 = layernorm<F32>(
        fp32_input, fp32_weight, fp32_bias, eps
    );

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train

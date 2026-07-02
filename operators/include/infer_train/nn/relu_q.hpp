#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/nn/relu.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_relu(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_relu only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = relu<F32>(fp32_input);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_leaky_relu(const Tensor<T>& input, float alpha = 0.01f) {
    static_assert(is_quantized<T>::value,
                  "quantized_leaky_relu only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = leaky_relu<F32>(fp32_input, alpha);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_elu(const Tensor<T>& input, float alpha = 1.0f) {
    static_assert(is_quantized<T>::value,
                  "quantized_elu only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = elu<F32>(fp32_input, alpha);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_gelu(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_gelu only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = gelu<F32>(fp32_input);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train

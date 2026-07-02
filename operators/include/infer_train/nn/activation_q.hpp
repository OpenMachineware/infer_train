#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/nn/activation.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_sigmoid(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_sigmoid only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = sigmoid<F32>(fp32_input);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_tanh(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_tanh only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = tanh<F32>(fp32_input);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_silu(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_silu only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = silu<F32>(fp32_input);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_hard_swish(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_hard_swish only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = hard_swish<F32>(fp32_input);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_hard_sigmoid(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_hard_sigmoid only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = hard_sigmoid<F32>(fp32_input);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

// ============================================================
// 量化 Softplus
// ============================================================
template<typename T>
Tensor<T> quantized_softplus(const Tensor<T>& input, float beta = 1.0f, float threshold = 20.0f) {
    static_assert(is_quantized<T>::value,
                  "quantized_softplus only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = softplus<F32>(fp32_input, beta, threshold);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

// ============================================================
// 量化 Softshrink
// ============================================================
template<typename T>
Tensor<T> quantized_softshrink(const Tensor<T>& input, float lambda = 0.5f) {
    static_assert(is_quantized<T>::value,
                  "quantized_softshrink only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = softshrink<F32>(fp32_input, lambda);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

// ============================================================
// 量化 CELU
// ============================================================
template<typename T>
Tensor<T> quantized_celu(const Tensor<T>& input, float alpha = 1.0f) {
    static_assert(is_quantized<T>::value,
                  "quantized_celu only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = celu<F32>(fp32_input, alpha);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

// nn/activation_q.hpp
template<typename T>
Tensor<T> quantized_relu6(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_relu6 only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = relu6<F32>(fp32_input);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train

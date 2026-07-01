#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/math/quantized.hpp"
#include "infer_train/nn/relu.hpp"
#include "infer_train/nn/softmax.hpp"
#include <algorithm>
#include <cmath>

namespace infer_train {

// ============================================================
// 量化 ReLU
// ============================================================
template<typename T>
Tensor<T> quantized_relu(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_relu only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> fp32_output = relu<F32>(fp32_input);

    float out_scale, out_zero_point;
    compute_scale_zero_point(fp32_output, out_scale, out_zero_point);
    return quantize<T>(fp32_output, out_scale, out_zero_point);
}

// ============================================================
// 量化 Leaky ReLU
// ============================================================
template<typename T>
Tensor<T> quantized_leaky_relu(const Tensor<T>& input, float alpha = 0.01f) {
    static_assert(is_quantized<T>::value,
                  "quantized_leaky_relu only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> fp32_output = leaky_relu<F32>(fp32_input, alpha);

    float out_scale, out_zero_point;
    compute_scale_zero_point(fp32_output, out_scale, out_zero_point);
    return quantize<T>(fp32_output, out_scale, out_zero_point);
}

// ============================================================
// 量化 ELU
// ============================================================
template<typename T>
Tensor<T> quantized_elu(const Tensor<T>& input, float alpha = 1.0f) {
    static_assert(is_quantized<T>::value,
                  "quantized_elu only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> fp32_output = elu<F32>(fp32_input, alpha);

    float out_scale, out_zero_point;
    compute_scale_zero_point(fp32_output, out_scale, out_zero_point);
    return quantize<T>(fp32_output, out_scale, out_zero_point);
}

// ============================================================
// 量化 GELU
// ============================================================
template<typename T>
Tensor<T> quantized_gelu(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_gelu only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> fp32_output = gelu<F32>(fp32_input);

    float out_scale, out_zero_point;
    compute_scale_zero_point(fp32_output, out_scale, out_zero_point);
    return quantize<T>(fp32_output, out_scale, out_zero_point);
}

// ============================================================
// 量化 Softmax
// ============================================================
template<typename T>
Tensor<T> quantized_softmax(const Tensor<T>& input, int dim = -1) {
    static_assert(is_quantized<T>::value,
                  "quantized_softmax only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> fp32_output = softmax<F32>(fp32_input, dim);

    float out_scale, out_zero_point;
    compute_scale_zero_point(fp32_output, out_scale, out_zero_point);
    return quantize<T>(fp32_output, out_scale, out_zero_point);
}

// ============================================================
// 量化 LogSoftmax
// ============================================================
template<typename T>
Tensor<T> quantized_log_softmax(const Tensor<T>& input, int dim = -1) {
    static_assert(is_quantized<T>::value,
                  "quantized_log_softmax only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> fp32_output = log_softmax<F32>(fp32_input, dim);

    float out_scale, out_zero_point;
    compute_scale_zero_point(fp32_output, out_scale, out_zero_point);
    return quantize<T>(fp32_output, out_scale, out_zero_point);
}

} // namespace infer_train

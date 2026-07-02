#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/nn/softmax.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_softmax(const Tensor<T>& input, int dim = -1) {
    static_assert(is_quantized<T>::value,
                  "quantized_softmax only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = softmax<F32>(fp32_input, dim);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_log_softmax(const Tensor<T>& input, int dim = -1) {
    static_assert(is_quantized<T>::value,
                  "quantized_log_softmax only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = log_softmax<F32>(fp32_input, dim);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train

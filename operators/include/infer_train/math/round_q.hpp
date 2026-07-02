#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/math/round.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_floor(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_floor only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = floor<F32>(fp32_input);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_ceil(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_ceil only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = ceil<F32>(fp32_input);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_round(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_round only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = round<F32>(fp32_input);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train

#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/math/log.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_log(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_log only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = log<F32>(fp32_input);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_log2(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_log2 only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = log2<F32>(fp32_input);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_log10(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_log10 only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = log10<F32>(fp32_input);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train

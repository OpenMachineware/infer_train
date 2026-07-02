#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/math/std_var.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_var(const Tensor<T>& input, bool unbiased = false) {
    static_assert(is_quantized<T>::value,
                  "quantized_var only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = var<F32>(fp32_input, unbiased);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_std(const Tensor<T>& input, bool unbiased = false) {
    static_assert(is_quantized<T>::value,
                  "quantized_std only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = std<F32>(fp32_input, unbiased);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train

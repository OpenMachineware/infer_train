#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/math/exp.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_exp(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_exp only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = exp<F32>(fp32_input);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_expm1(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_expm1 only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = expm1<F32>(fp32_input);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train

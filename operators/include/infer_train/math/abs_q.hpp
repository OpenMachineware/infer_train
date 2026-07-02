#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/math/abs.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_abs(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_abs only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = abs<F32>(fp32_input);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_neg(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_neg only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = neg<F32>(fp32_input);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train

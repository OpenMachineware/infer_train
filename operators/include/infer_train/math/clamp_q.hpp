#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/math/clamp.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_clamp(
    const Tensor<T>& input,
    typename T::storage min_val,
    typename T::storage max_val
) {
    static_assert(is_quantized<T>::value,
                  "quantized_clamp only works with quantized types");

    using Conv = QuantizedConverter<T>;
    Tensor<F32> fp32_input = dequantize(input);

    float fmin = Conv::dequantize(min_val, 1.0f, 0.0f);
    float fmax = Conv::dequantize(max_val, 1.0f, 0.0f);

    Tensor<F32> result_fp32 = clamp<F32>(fp32_input, fmin, fmax);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train

#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/math/reduce.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_sum(
    const Tensor<T>& input,
    const std::vector<int>& dims,
    bool keepdim = false
) {
    static_assert(is_quantized<T>::value,
                  "quantized_sum only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = sum<F32>(fp32_input, dims, keepdim);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_mean(
    const Tensor<T>& input,
    const std::vector<int>& dims,
    bool keepdim = false
) {
    static_assert(is_quantized<T>::value,
                  "quantized_mean only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = mean<F32>(fp32_input, dims, keepdim);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_max_all(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_max_all only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = max_all<F32>(fp32_input);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_min_all(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_min_all only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = min_all<F32>(fp32_input);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train

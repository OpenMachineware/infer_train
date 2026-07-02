#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/nn/pooling.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_maxpool2d(
    const Tensor<T>& input,
    int kernel_size,
    int stride = -1,
    int padding = 0
) {
    static_assert(is_quantized<T>::value,
                  "quantized_maxpool2d only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = maxpool2d<F32>(
        fp32_input, kernel_size, stride, padding
    );

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_avgpool2d(
    const Tensor<T>& input,
    int kernel_size,
    int stride = -1,
    int padding = 0
) {
    static_assert(is_quantized<T>::value,
                  "quantized_avgpool2d only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = avgpool2d<F32>(
        fp32_input, kernel_size, stride, padding
    );

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train

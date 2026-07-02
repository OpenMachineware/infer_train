#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/nn/pool.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_maxpool1d(
    const Tensor<T>& input,
    int kernel_size,
    int stride = -1,
    int padding = 0
) {
    static_assert(is_quantized<T>::value,
                  "quantized_maxpool1d only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = maxpool1d<F32>(fp32_input, kernel_size, stride, padding);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_maxpool3d(
    const Tensor<T>& input,
    int kernel_size,
    int stride = -1,
    int padding = 0
) {
    static_assert(is_quantized<T>::value,
                  "quantized_maxpool3d only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = maxpool3d<F32>(fp32_input, kernel_size, stride, padding);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_avgpool1d(
    const Tensor<T>& input,
    int kernel_size,
    int stride = -1,
    int padding = 0
) {
    static_assert(is_quantized<T>::value,
                  "quantized_avgpool1d only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = avgpool1d<F32>(fp32_input, kernel_size, stride, padding);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

template<typename T>
Tensor<T> quantized_avgpool3d(
    const Tensor<T>& input,
    int kernel_size,
    int stride = -1,
    int padding = 0
) {
    static_assert(is_quantized<T>::value,
                  "quantized_avgpool3d only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = avgpool3d<F32>(fp32_input, kernel_size, stride, padding);
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train

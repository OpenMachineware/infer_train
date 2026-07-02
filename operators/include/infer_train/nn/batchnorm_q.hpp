#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/nn/batchnorm.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_batchnorm2d_inference(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    const Tensor<T>& bias,
    const Tensor<T>& running_mean,
    const Tensor<T>& running_var,
    float eps = 1e-5f
) {
    static_assert(is_quantized<T>::value,
                  "quantized_batchnorm2d_inference only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> fp32_weight = dequantize(weight);
    Tensor<F32> fp32_bias = dequantize(bias);
    Tensor<F32> fp32_mean = dequantize(running_mean);
    Tensor<F32> fp32_var = dequantize(running_var);

    Tensor<F32> result_fp32 = batchnorm2d_inference<F32>(
        fp32_input, fp32_weight, fp32_bias,
        fp32_mean, fp32_var, eps
    );

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

// ============================================================
// batchnorm1d 量化版本
// ============================================================
// operators/include/infer_train/nn/batchnorm_q.hpp

template<typename T>
Tensor<T> quantized_batchnorm1d_inference(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    const Tensor<T>& bias,
    const Tensor<T>& running_mean,
    const Tensor<T>& running_var,
    float eps = 1e-5f
) {
    static_assert(is_quantized<T>::value,
                  "quantized_batchnorm1d_inference only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> fp32_weight = dequantize(weight);
    Tensor<F32> fp32_bias = dequantize(bias);
    Tensor<F32> fp32_mean = dequantize(running_mean);
    Tensor<F32> fp32_var = dequantize(running_var);

    Tensor<F32> result_fp32 = batchnorm1d_inference<F32>(
        fp32_input, fp32_weight, fp32_bias,
        fp32_mean, fp32_var, eps
    );

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}



} // namespace infer_train

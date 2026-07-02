#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/nn/rmsnorm.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_rmsnorm(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    float eps = 1e-6f
) {
    static_assert(is_quantized<T>::value,
                  "quantized_rmsnorm only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> fp32_weight = dequantize(weight);

    Tensor<F32> result_fp32 = rmsnorm<F32>(fp32_input, fp32_weight, eps);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train

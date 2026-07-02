#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/math/arg.hpp"

namespace infer_train {

template<typename T>
Tensor<int64_t> quantized_argmax(const Tensor<T>& input, int dim = -1) {
    static_assert(is_quantized<T>::value,
                  "quantized_argmax only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    return argmax<F32>(fp32_input, dim);
}

template<typename T>
Tensor<int64_t> quantized_argmin(const Tensor<T>& input, int dim = -1) {
    static_assert(is_quantized<T>::value,
                  "quantized_argmin only works with quantized types");
    Tensor<F32> fp32_input = dequantize(input);
    return argmin<F32>(fp32_input, dim);
}

} // namespace infer_train

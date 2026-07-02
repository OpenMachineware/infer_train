#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/math/add.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_add(const Tensor<T>& a, const Tensor<T>& b) {
    static_assert(is_quantized<T>::value,
                  "quantized_add only works with quantized types");

    if (a.shape != b.shape) {
        return Tensor<T>();
    }

    Tensor<F32> a_fp32 = dequantize(a);
    Tensor<F32> b_fp32 = dequantize(b);
    Tensor<F32> result_fp32 = add<F32>(a_fp32, b_fp32);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train
